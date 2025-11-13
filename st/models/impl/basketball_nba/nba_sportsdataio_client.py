"""
SportsDataIO async client for NBA data (scores, stats, injuries, odds).

This client uses aiohttp and follows SportsDataIO's authentication model:
- Pass your API key via the "Ocp-Apim-Subscription-Key" header OR as ?key= query param.
Docs (general): https://sportsdata.io/developers/api-documentation/nba
Data dictionary (tables/endpoints referenced in docstrings): https://sportsdata.io/developers/data-dictionary/nba
"""

from __future__ import annotations
import os
import asyncio
import aiohttp
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple
import dotenv

dotenv.load_dotenv()

DEFAULT_TIMEOUT = 10
RETRY_BACKOFF = [0.5, 1.0, 2.0, 4.0]

SCORES_BASE = "https://api.sportsdata.io/v3/nba/scores/json"
STATS_BASE = "https://api.sportsdata.io/v3/nba/stats/json"
ODDS_BASE = "https://api.sportsdata.io/v3/nba/odds/json"
PROJ_BASE = "https://api.sportsdata.io/v3/nba/projections/json"


def _fmt_date(d: dt.date | dt.datetime) -> str:
    if isinstance(d, dt.datetime):
        d = d.date()
    return d.strftime("%Y-%m-%d")


class SportsDataIOClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = (
            api_key
            or os.getenv("SPORTSDATAIO_API_KEY")
            or os.getenv("SPORTS_DATA_IO_KEY")
            or os.getenv("SPORTSDATAIO_KEY")
        )
        if not self.api_key:
            raise ValueError("Set SPORTSDATAIO_API_KEY in env")
        self._session = session
        self.timeout = timeout

    async def __aenter__(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()
            self._session = None

    async def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if params is None:
            params = {}
        # Either header or query param works; we use header and keep ?key for redundancy
        params.setdefault("key", self.api_key)
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        last_err = None
        for backoff in [0.0] + RETRY_BACKOFF:
            if backoff:
                await asyncio.sleep(backoff)
            try:
                assert self._session is not None, "ClientSession not initialized"
                async with self._session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    # many SDIO endpoints return 204 for no content on off days
                    if resp.status in (204, 202):
                        return None
                    try:
                        txt = await resp.text()
                    except Exception:
                        txt = ""
                    last_err = RuntimeError(f"GET {url} -> {resp.status} {txt[:200]}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
        raise last_err

    # ------------------ Scores/Schedules ------------------
    async def games_by_date(
        self, date: dt.date | dt.datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """Games - by Date (Scores feed). Data dictionary shows 'Games - by Date [Live & Final]' and 'Final' tables.
        Endpoint shape (typical): /v3/nba/scores/json/GamesByDate/{date}
        """
        url = f"{SCORES_BASE}/GamesByDate/{_fmt_date(date)}"
        return await self._get(url)

    async def teams(self) -> Optional[List[Dict[str, Any]]]:
        """Teams metadata (Scores feed): /v3/nba/scores/json/teams"""
        url = f"{SCORES_BASE}/Teams"
        return await self._get(url)

    # ------------------ Team/Player Stats ------------------
    async def team_game_stats_by_date(
        self, date: dt.date | dt.datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """Team Game Stats - by Date (Stats feed): /v3/nba/stats/json/TeamGameStatsByDate/{date}"""
        url = f"{STATS_BASE}/TeamGameStatsByDate/{_fmt_date(date)}"
        return await self._get(url)

    async def player_game_stats_by_date(
        self, date: dt.date | dt.datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """Player Game Stats - by Date (Stats feed): /v3/nba/stats/json/PlayerGameStatsByDate/{date}"""
        url = f"{STATS_BASE}/PlayerGameStatsByDate/{_fmt_date(date)}"
        return await self._get(url)

    async def team_game_logs_by_season(
        self, season: int, season_type: str = "Regular"
    ) -> Optional[List[Dict[str, Any]]]:
        """Team Game Logs - by Season: /v3/nba/stats/json/TeamGameLogs/{season}?SeasonType=Regular
        Notes: Season is the year in which the season starts, e.g. 2024 for 2024-2025.
        """
        url = f"{STATS_BASE}/TeamGameLogs/{season}"
        params = {"SeasonType": season_type, "key": self.api_key}
        return await self._get(url, params=params)

    async def injuries(self) -> Optional[List[Dict[str, Any]]]:
        """Injuries (current): /v3/nba/projections/json/InjuredPlayers"""
        url = f"{PROJ_BASE}/InjuredPlayers"
        return await self._get(url)

    # ------------------ Betting/Odds ------------------
    async def game_odds_by_date(
        self, date: dt.date | dt.datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """(If your plan includes betting): Game Odds - by Date (common pattern): /v3/nba/odds/json/GameOddsByDate/{date}"""
        url = f"{ODDS_BASE}/GameOddsByDate/{_fmt_date(date)}"
        return await self._get(url)

    # ------------------ Utilities ------------------
    async def iterate_dates(self, start: dt.date, end: dt.date):
        """Inclusive date range generator"""
        cur = start
        while cur <= end:
            yield cur
            cur += dt.timedelta(days=1)
