"""
End-to-end updater for NBA win-probability model using SportsDataIO.

Usage (environment must set SPORTSDATAIO_API_KEY):

    python nba_update.py --start 2022-10-01 --end 2025-07-01

This script:
- Pulls Games by Date (and InjuredPlayers, if desired).
- Builds rolling features with Elo and last-10 form.
- Trains + calibrates a gradient boosted classifier.
- Saves the model to ./models/nba_winprob.joblib
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
from typing import Dict, List, Any, Set

import pandas as pd

from nba_sportsdataio_client import SportsDataIOClient
from nba_features import (
    EloState,
    RollingTeamState,
    summarize_injuries,
    build_match_feature_row,
)


def normalize_team_name(name: str) -> str:
    return name.strip()


async def fetch_range_build_dataset(start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Walk from start..end (inclusive) by date, pulling games and building
    a labeled dataset of completed games with pre-game features.

    For each final game:
    - Features are computed using current Elo/Rolling state (before the game).
    - Then Elo/Rolling are updated with the result.
    """
    async with SportsDataIOClient() as cli:
        rolling = RollingTeamState()
        elo = EloState()
        rows: List[Dict[str, Any]] = []

        async for day in cli.iterate_dates(start, end):
            games = await cli.games_by_date(day) or []
            injuries = await cli.injuries() or []
            inj_summary = summarize_injuries(injuries)

            for g in games:
                status = str(g.get("Status") or "").strip()
                if not status.startswith("Final"):
                    continue

                home_raw = g.get("HomeTeam") or g.get("HomeTeamName") or ""
                away_raw = g.get("AwayTeam") or g.get("AwayTeamName") or ""
                home = normalize_team_name(home_raw)
                away = normalize_team_name(away_raw)

                dt_str = g.get("DateTime") or g.get("Day")
                if not dt_str:
                    continue

                try:
                    gd = pd.to_datetime(dt_str)
                except Exception:
                    continue

                home_pts = int(g.get("HomeTeamScore") or 0)
                away_pts = int(g.get("AwayTeamScore") or 0)
                if home_pts == 0 and away_pts == 0:
                    continue

                ref_ts = pd.to_datetime(gd)

                # Pre-game features: use state before updating with this game.
                feat = build_match_feature_row(
                    home=home,
                    away=away,
                    ref_date=ref_ts,
                    elo=elo,
                    rolling=rolling,
                    inj_summary=inj_summary,
                )
                feat["date"] = ref_ts
                feat["home"] = home
                feat["away"] = away
                feat["home_win"] = int(home_pts > away_pts)
                rows.append(feat)

                # Now update state with the result (post-game)
                rolling.push_game(
                    {
                        "date": ref_ts,
                        "home": home,
                        "away": away,
                        "home_pts": home_pts,
                        "away_pts": away_pts,
                    }
                )
                elo.update(home, away, home_pts, away_pts)

        return pd.DataFrame(rows)


def seasons_from_range(start: dt.date, end: dt.date) -> List[int]:
    """
    Convert a date range to a list of SportsDataIO NBA season keys.
    Season key is the year the season starts, e.g. 2024 for 2024-25.
    """
    seasons: Set[int] = set()
    cur = start
    while cur <= end:
        yr = cur.year if cur.month >= 7 else cur.year - 1
        seasons.add(yr)
        cur += dt.timedelta(days=60)
    return sorted(seasons)


async def async_main(args: argparse.Namespace) -> None:
    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date()

    df = await fetch_range_build_dataset(start, end)
    if df.empty:
        raise SystemExit("No data fetched; check API access and date range.")

    from nba_model import train_model, save_model

    pipe, meta, metrics = train_model(
        df,
        seasons_from_range(start, end),
    )
    path = save_model(pipe, meta)
    print("Saved model:", path)
    print("Metrics:", metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
