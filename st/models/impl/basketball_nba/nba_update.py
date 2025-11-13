"""
End-to-end updater for NBA win-probability model using SportsDataIO.

Usage (environment must set SPORTSDATAIO_API_KEY):
    python nba_update.py --start 2022-10-01 --end 2025-07-01 --seasons 2023 2024 2025

This script:
- Pulls Games by Date + Team Game Stats by Date + Injuries
- Builds rolling features with Elo and last-10 form
- Trains + calibrates a gradient boosted classifier
- Saves the model to models/nba_winprob.joblib
"""

from __future__ import annotations

import argparse, asyncio, datetime as dt, os, json
from typing import Dict, List, Any
import pandas as pd

from nba_sportsdataio_client import SportsDataIOClient
from nba_features import (
    EloState,
    RollingTeamState,
    summarize_injuries,
    build_match_feature_row,
)
from nba_model import train_model, save_model


def normalize_team_name(name: str) -> str:
    # SportsDataIO returns City+Name (e.g., "Los Angeles Lakers") or "Lakers" in some tables; we normalize to City + Nickname.
    return name.strip()


async def fetch_range_build_dataset(start: dt.date, end: dt.date) -> pd.DataFrame:
    async with SportsDataIOClient() as cli:
        rolling = RollingTeamState()
        elo = EloState()
        rows = []
        async for day in cli.iterate_dates(start, end):
            games = await cli.games_by_date(day) or []
            stats = await cli.team_game_stats_by_date(day) or []
            injuries = await cli.injuries() or []
            inj_summary = summarize_injuries(injuries)
            # index stats by (Team, Opponent, HomeAway) if needed later
            for g in games:
                if g.get("Status") not in (
                    "Final",
                    "F inal",
                    "F",
                    "FinalOvertime",
                    "Scheduled",
                    "InProgress",
                ):
                    # still include scheduled for feature generation for future inference use
                    pass
                # expected keys from Games endpoint: HomeTeam, AwayTeam, HomeTeamScore, AwayTeamScore, DateTime
                home = normalize_team_name(
                    g["HomeTeam"] if "HomeTeam" in g else g.get("HomeTeamName", "")
                )
                away = normalize_team_name(
                    g["AwayTeam"] if "AwayTeam" in g else g.get("AwayTeamName", "")
                )
                dt_str = g.get("DateTime") or g.get("Day")
                if not dt_str:
                    continue
                gd = pd.to_datetime(dt_str)
                home_pts = int(g.get("HomeTeamScore") or 0)
                away_pts = int(g.get("AwayTeamScore") or 0)

                # Only update elo/rolling on completed games (scores present)
                if (home_pts or away_pts) and g.get("Status", "").startswith("Final"):
                    rolling.push_game(
                        {
                            "date": gd,
                            "home": home,
                            "away": away,
                            "home_pts": home_pts,
                            "away_pts": away_pts,
                        }
                    )
                    elo.update(home, away, home_pts, away_pts)
                    # create a labeled row (features as of pre-game: use reference = tipoff time)
                    feat = build_match_feature_row(
                        home, away, gd, elo, rolling, inj_summary
                    )
                    feat["date"] = gd
                    feat["home"] = home
                    feat["away"] = away
                    feat["home_win"] = int(home_pts > away_pts)
                    rows.append(feat)
        return pd.DataFrame(rows)


def seasons_from_range(start: dt.date, end: dt.date) -> List[int]:
    # SportsDataIO season key: year season starts
    out = set()
    cur = start
    while cur <= end:
        yr = cur.year if cur.month >= 7 else cur.year - 1
        out.add(yr)
        cur += dt.timedelta(days=60)
    return sorted(out)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    args = parser.parse_args()

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date()
    df = await fetch_range_build_dataset(start, end)
    if df.empty:
        raise SystemExit("No data fetched. Check API access and date range.")
    pipe, meta, metrics = train_model(df, seasons_from_range(start, end))
    path = save_model(pipe, meta)
    print("Saved model:", path)
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import pandas as pd

    asyncio.run(main())
