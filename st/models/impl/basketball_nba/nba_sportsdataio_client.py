"""
SportsDataIO async client for NBA data (scores, stats, injuries, odds).

This client uses aiohttp and follows SportsDataIO's authentication model:
- Pass your API key via the "Ocp-Apim-Subscription-Key" header OR as ?key= query param.

Docs (general, for reference):
- NBA API docs + Data Dictionary
"""

from __future__ import annotations

import os
import asyncio
import datetime as dt
from typing import Any, Dict, List, Optional, AsyncIterator

import aiohttp


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
            raise ValueError("Set SPORTSDATAIO_API_KEY in environment.")
        self._session = session
        self.timeout = timeout

    async def __aenter__(self) -> "SportsDataIOClient":
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if params is None:
            params = {}
        params.setdefault("key", self.api_key)
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        last_err: Optional[Exception] = None

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
                    if resp.status in (202, 204):
                        return None
                    try:
                        txt = await resp.text()
                    except Exception:
                        txt = ""
                    last_err = RuntimeError(f"GET {url} -> {resp.status} {txt[:200]!r}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e

        raise last_err or RuntimeError(f"GET {url} failed without response")

    # ------------------ Scores / metadata ------------------

    async def games_by_date(
        self, date: dt.date | dt.datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Games - by Date (Scores feed).
        Endpoint: /v3/nba/scores/json/GamesByDate/{date}
        """
        url = f"{SCORES_BASE}/GamesByDate/{_fmt_date(date)}"
        return await self._get(url)

    async def teams(self) -> Optional[List[Dict[str, Any]]]:
        """
        Teams metadata (Scores feed).
        Endpoint: /v3/nba/scores/json/Teams
        """
        url = f"{SCORES_BASE}/Teams"
        return await self._get(url)

    # ------------------ Stats ------------------

    async def team_game_stats_by_date(
        self, date: dt.date | dt.datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Team Game Stats - by Date (Stats feed).
        Endpoint: /v3/nba/stats/json/TeamGameStatsByDate/{date}
        """
        url = f"{STATS_BASE}/TeamGameStatsByDate/{_fmt_date(date)}"
        return await self._get(url)

    async def player_game_stats_by_date(
        self, date: dt.date | dt.datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Player Game Stats - by Date (Stats feed).
        Endpoint: /v3/nba/stats/json/PlayerGameStatsByDate/{date}
        """
        url = f"{STATS_BASE}/PlayerGameStatsByDate/{_fmt_date(date)}"
        return await self._get(url)

    # ------------------ Injuries (projections feed) ------------------

    async def injuries(self) -> Optional[List[Dict[str, Any]]]:
        """
        Injuries (current).
        Endpoint: /v3/nba/projections/json/InjuredPlayers
        """
        url = f"{PROJ_BASE}/InjuredPlayers"
        return await self._get(url)

    # ------------------ Betting / odds ------------------

    async def game_odds_by_date(
        self, date: dt.date | dt.datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Game Odds - by Date (betting feed, plan-dependent).
        Typical endpoint: /v3/nba/odds/json/GameOddsByDate/{date}
        """
        url = f"{ODDS_BASE}/GameOddsByDate/{_fmt_date(date)}"
        return await self._get(url)

    # ------------------ Utilities ------------------

    async def iterate_dates(
        self, start: dt.date, end: dt.date
    ) -> AsyncIterator[dt.date]:
        """Inclusive date range generator."""
        cur = start
        while cur <= end:
            yield cur
            cur += dt.timedelta(days=1)
