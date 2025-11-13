"""
Feature engineering for NBA win-probability model.

We build leakage-safe, pre-game features primarily from:
- Games by Date (for matchups and results)
- Rolling team performance (last 10 games)
- Elo-style ratings
- Simple injury burden counts

All features are designed to be computable both historically (training) and
at inference time given SportsDataIO feeds.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd


# --------------------------- ELO ---------------------------


@dataclass
class EloConfig:
    base: float = 1500.0
    k: float = 20.0
    hca: float = 65.0  # home-court advantage in Elo points
    mov_mult_cap: float = 3.0  # cap the margin multiplier


def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def mov_multiplier(mov: float, elo_diff: float) -> float:
    # Classic basketball MoV scaling similar to 538's Elo adjustments.
    return min(
        (np.log(abs(mov) + 1.0) * 2.2) / (2.2 + (elo_diff * 0.001)),
        3.0,
    )


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
        mm = mov_multiplier(mov, rh - ra)
        k = self.cfg.k * mm

        new_rh_base = self.rating(home_team) + k * (home_w - exp_h)
        new_ra_base = self.rating(away_team) + k * (away_w - exp_a)

        self.ratings[home_team] = new_rh_base
        self.ratings[away_team] = new_ra_base


# --------------------------- Rolling team state ---------------------------


def safe_key(x: Optional[str]) -> Optional[str]:
    return x.strip().lower() if isinstance(x, str) else x


def is_back_to_back(dates: List[pd.Timestamp], ref: pd.Timestamp) -> int:
    if not dates:
        return 0
    return int(any((ref - d).days == 1 for d in dates))


@dataclass
class RollingTeamState:
    """
    Holds a rolling window of recent games for each team.
    Stored as per-team lists of simple dicts.
    """

    last_games: Dict[str, List[Dict]] = field(default_factory=lambda: defaultdict(list))

    def push_game(self, row: Dict):
        """
        `row`:
        {
            "date": pd.Timestamp,
            "home": str,
            "away": str,
            "home_pts": int,
            "away_pts": int,
        }
        """
        date = pd.to_datetime(row["date"])
        home = row["home"]
        away = row["away"]
        home_pts = int(row["home_pts"])
        away_pts = int(row["away_pts"])

        home_win = 1 if home_pts > away_pts else 0

        home_entry = dict(
            date=date,
            is_home=1,
            pts=home_pts,
            opp_pts=away_pts,
            win=home_win,
        )
        away_entry = dict(
            date=date,
            is_home=0,
            pts=away_pts,
            opp_pts=home_pts,
            win=1 - home_win,
        )

        self.last_games[home].append(home_entry)
        self.last_games[away].append(away_entry)

        # Keep last 30 per team for efficiency
        if len(self.last_games[home]) > 30:
            self.last_games[home] = self.last_games[home][-30:]
        if len(self.last_games[away]) > 30:
            self.last_games[away] = self.last_games[away][-30:]

    def make_features(self, team: str, ref_date: pd.Timestamp) -> Dict[str, float]:
        """
        Build features for `team` as of `ref_date` (exclusive), i.e. only
        games with date < ref_date are considered.
        """
        games = [g for g in self.last_games.get(team, []) if g["date"] < ref_date]
        if not games:
            # Reasonable priors for unseen team
            return {
                "gp_l10": 0.0,
                "win_pct_l10": 0.5,
                "pd_l10": 0.0,
                "home_frac_l10": 0.5,
                "rest_b2b": 0.0,
                "days_since_game": 6.0,
            }

        g10 = games[-10:]
        win_pct = float(np.mean([g["win"] for g in g10]))
        pdiff = float(np.mean([g["pts"] - g["opp_pts"] for g in g10]))
        home_frac = float(np.mean([g["is_home"] for g in g10]))
        last_dates = [g["date"] for g in games]
        b2b = float(is_back_to_back(last_dates, ref_date))
        days_since = float((ref_date - max(last_dates)).days)

        return {
            "gp_l10": float(len(g10)),
            "win_pct_l10": win_pct,
            "pd_l10": pdiff,
            "home_frac_l10": home_frac,
            "rest_b2b": b2b,
            "days_since_game": days_since,
        }


# --------------------------- Injuries ---------------------------


def summarize_injuries(injuries: List[Dict]) -> Dict[str, float]:
    """
    Very simple injury burden proxy: count players with a meaningful
    InjuryStatus / Status per team.

    Compatible with /projections/json/InjuredPlayers, where rows commonly have:
    - Team
    - InjuryStatus (Injured, Doubtful, Questionable, etc.)
    - Status (sometimes also set)
    """
    if not injuries:
        return {}

    bad = {"INJURED", "OUT", "DOUBTFUL", "QUESTIONABLE"}
    team_to_ct: Dict[str, float] = defaultdict(float)

    for it in injuries:
        team = safe_key(it.get("Team"))
        if not team:
            continue
        status_raw = it.get("InjuryStatus") or it.get("Status") or ""
        status = str(status_raw).strip().upper()
        if status in bad:
            team_to_ct[team] += 1.0

    return dict(team_to_ct)


# --------------------------- Match feature row ---------------------------


def build_match_feature_row(
    home: str,
    away: str,
    ref_date: pd.Timestamp,
    elo: EloState,
    rolling: RollingTeamState,
    inj_summary: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build a single, leakage-safe feature row for (home, away) as of ref_date.
    """

    inj_summary = inj_summary or {}

    fh = rolling.make_features(home, ref_date)
    fa = rolling.make_features(away, ref_date)

    elo_home = elo.rating(home)
    elo_away = elo.rating(away)

    inj_home = inj_summary.get(safe_key(home), 0.0)
    inj_away = inj_summary.get(safe_key(away), 0.0)

    row = {
        "elo_diff": float(elo_home - elo_away),
        "home_win_pct_l10": float(fh["win_pct_l10"]),
        "away_win_pct_l10": float(fa["win_pct_l10"]),
        "pdiff_l10": float(fh["pd_l10"] - fa["pd_l10"]),
        "home_days_since": float(fh["days_since_game"]),
        "away_days_since": float(fa["days_since_game"]),
        "home_b2b": float(fh["rest_b2b"]),
        "away_b2b": float(fa["rest_b2b"]),
        "home_gp_l10": float(fh["gp_l10"]),
        "away_gp_l10": float(fa["gp_l10"]),
        "inj_home_ct": float(inj_home),
        "inj_away_ct": float(inj_away),
    }

    return row
