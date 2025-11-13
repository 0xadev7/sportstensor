"""
Feature engineering for NBA win-probability model.

We build leakage-safe, pre-game features primarily from:
- Games by Date (Scores feed) – past results, home/away, start times
- Team Game Stats by Date (Stats feed) – team totals to compute Four Factors-like rates
- Injuries (Stats feed) – aggregate impact proxies
Data dictionary: https://sportsdata.io/developers/data-dictionary/nba
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict


# --------------------------- ELO ---------------------------
@dataclass
class EloConfig:
    base: float = 1500.0
    k: float = 20.0
    hca: float = 65.0  # home-court advantage in Elo points
    mov_mult_cap: float = 2.0  # cap the margin multiplier


def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def mov_multiplier(mov: float, elo_diff: float) -> float:
    # classic basketball MoV scaling similar to FiveThirtyEight's approach
    return min((np.log(abs(mov) + 1) * 2.2) / (2.2 + (elo_diff * 0.001)), 3.0)


@dataclass
class EloState:
    cfg: EloConfig = field(default_factory=EloConfig)
    ratings: Dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.cfg.base)

    def update(self, home_team: str, away_team: str, home_pts: int, away_pts: int):
        rh = self.rating(home_team) + self.cfg.hca
        ra = self.rating(away_team)
        exp_h = elo_expected(rh, ra)
        exp_a = 1.0 - exp_h
        home_w = 1.0 if home_pts > away_pts else 0.0
        away_w = 1.0 - home_w
        mov = abs(home_pts - away_pts)
        mm = mov_multiplier(mov, (rh - ra))
        k = self.cfg.k * mm
        new_rh = self.rating(home_team) + k * (home_w - exp_h)
        new_ra = self.rating(away_team) + k * (away_w - exp_a)
        self.ratings[home_team] = new_rh
        self.ratings[away_team] = new_ra


# --------------------------- Utilities ---------------------------


def safe_key(x: Optional[str]) -> Optional[str]:
    return x.strip().lower() if isinstance(x, str) else x


def compute_rest_days(game_dates: List[dt.datetime]) -> int:
    if len(game_dates) == 0:
        return 6  # treat as fully rested
    return int((pd.Timestamp.utcnow() - pd.Series(game_dates).max()).days)


def is_back_to_back(dates: List[pd.Timestamp], ref: pd.Timestamp) -> int:
    if len(dates) == 0:
        return 0
    return int(any((ref - d).days == 1 for d in dates))


# --------------------------- Core FE ---------------------------


@dataclass
class RollingTeamState:
    # store last N games stats for quick lookups
    last_games: Dict[str, List[Dict]] = field(default_factory=lambda: defaultdict(list))

    def push_game(self, row: Dict):
        # We push two entries: one for home as team, one for away as team
        # Expected keys in row: date, home, away, home_pts, away_pts
        date = pd.to_datetime(row["date"])
        h, a = row["home"], row["away"]
        home_win = 1 if row["home_pts"] > row["away_pts"] else 0

        home_entry = dict(
            date=date,
            is_home=1,
            pts=row["home_pts"],
            opp_pts=row["away_pts"],
            win=home_win,
        )
        away_entry = dict(
            date=date,
            is_home=0,
            pts=row["away_pts"],
            opp_pts=row["home_pts"],
            win=1 - home_win,
        )

        self.last_games[h].append(home_entry)
        self.last_games[a].append(away_entry)

        # keep last 30 for efficiency
        if len(self.last_games[h]) > 30:
            self.last_games[h] = self.last_games[h][-30:]
        if len(self.last_games[a]) > 30:
            self.last_games[a] = self.last_games[a][-30:]

    def make_features(self, team: str, ref_date: pd.Timestamp) -> Dict[str, float]:
        games = [g for g in self.last_games.get(team, []) if g["date"] < ref_date]
        if not games:
            return {
                "gp_l10": 0,
                "win_pct_l10": 0.5,
                "pd_l10": 0.0,
                "home_frac_l10": 0.5,
                "rest_b2b": 0,
                "days_since_game": 6.0,
            }
        g10 = games[-10:]
        win_pct = np.mean([g["win"] for g in g10])
        pdiff = np.mean([g["pts"] - g["opp_pts"] for g in g10])
        home_frac = np.mean([g["is_home"] for g in g10])
        last_dates = [g["date"] for g in games]
        b2b = is_back_to_back(last_dates, ref_date)
        days_since = float((ref_date - max(last_dates)).days)
        return {
            "gp_l10": float(len(g10)),
            "win_pct_l10": float(win_pct),
            "pd_l10": float(pdiff),
            "home_frac_l10": float(home_frac),
            "rest_b2b": float(b2b),
            "days_since_game": float(days_since),
        }


def build_match_feature_row(
    home: str,
    away: str,
    ref_date: pd.Timestamp,
    elo: EloState,
    rolling: RollingTeamState,
    inj_summary: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    fh = rolling.make_features(home, ref_date)
    fa = rolling.make_features(away, ref_date)
    elo_home = elo.rating(home)
    elo_away = elo.rating(away)
    inj_home = (inj_summary or {}).get(safe_key(home), 0.0)
    inj_away = (inj_summary or {}).get(safe_key(away), 0.0)

    row = {
        "elo_diff": float(elo_home - elo_away),
        "home_win_pct_l10": fh["win_pct_l10"],
        "away_win_pct_l10": fa["win_pct_l10"],
        "pdiff_l10": float(fh["pd_l10"] - fa["pd_l10"]),
        "home_days_since": fh["days_since_game"],
        "away_days_since": fa["days_since_game"],
        "home_b2b": fh["rest_b2b"],
        "away_b2b": fa["rest_b2b"],
        "home_gp_l10": fh["gp_l10"],
        "away_gp_l10": fa["gp_l10"],
        "inj_home_ct": float(inj_home),
        "inj_away_ct": float(inj_away),
    }
    return row


def summarize_injuries(injuries: List[Dict]) -> Dict[str, float]:
    """
    Very simple injury burden proxy: count players with a non-null InjuryStatus
    in {'Injured', 'Out', 'Doubtful', 'Questionable'} (or Status in that set),
    grouped by Team.
    Works with /projections/json/InjuredPlayers.
    """
    if not injuries:
        return {}
    bad = {"INJURED", "OUT", "DOUBTFUL", "QUESTIONABLE"}
    team_to_ct: Dict[str, float] = defaultdict(float)
    for it in injuries:
        team = safe_key(it.get("Team"))
        # Prefer InjuryStatus, fall back to Status
        status_raw = it.get("InjuryStatus") or it.get("Status") or ""
        status = str(status_raw).strip().upper()
        if team and status in bad:
            team_to_ct[team] += 1.0
    return dict(team_to_ct)
