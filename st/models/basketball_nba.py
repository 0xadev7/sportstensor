import asyncio
import datetime as dt
from typing import Dict, Any, Optional

import pandas as pd
import bittensor as bt

from st.sport_prediction_model import SportPredictionModel
from common.data import ProbabilityChoice

from st.models.impl.basketball_nba.nba_sportsdataio_client import SportsDataIOClient
from st.models.impl.basketball_nba.nba_features import (
    EloState,
    RollingTeamState,
    summarize_injuries,
    build_match_feature_row,
)
from st.models.impl.basketball_nba.nba_model import load_model, predict_proba

def _american_to_implied_prob(ml: int) -> Optional[float]:
    """
    Convert American moneyline to implied probability (with vig).
    Returns None for 0.
    """
    if ml is None:
        return None
    try:
        ml = int(ml)
    except Exception:
        return None
    if ml == 0:
        return None
    if ml < 0:
        return 100.0 / (abs(ml) + 100.0)
    else:
        return (ml) / (ml + 100.0)


def _strip_vig_two(p_home: float, p_away: float) -> Optional[float]:
    """
    Strip vig from two implied probabilities; return normalized home prob.
    """
    if p_home is None or p_away is None:
        return None
    s = p_home + p_away
    if s <= 0:
        return None
    return p_home / s


def _blend_model_market(
    p_model_home: float,
    p_market_home: Optional[float],
    hours_to_tip: float,
) -> float:
    """
    Blend model and market probabilities as a function of hours_to_tip.

    - Far from game (>=48h): alpha ~ 0 (pure model).
    - At tip (0h): alpha ~ 0.8 (80% market, 20% model) if odds exist.
    """
    if p_market_home is None:
        return p_model_home

    horizon = 48.0
    t = max(0.0, min(horizon, hours_to_tip))
    alpha = 0.8 * (1.0 - t / horizon)  # 0..0.8

    return float((1.0 - alpha) * p_model_home + alpha * p_market_home)


class NBABasketballPredictionModel(SportPredictionModel):
    async def make_prediction(self):
        bt.logging.info(
            "Predicting NBA basketball game with SportsDataIO + ML model..."
        )
        try:
            pipe = load_model()
            if pipe is None:
                bt.logging.error(
                    "NBA model not found at ./models/nba_winprob.joblib. "
                    "Run nba_update.py first."
                )
                return

            match_dt = (
                self.prediction.matchDate
                if isinstance(self.prediction.matchDate, dt.datetime)
                else dt.datetime.fromisoformat(str(self.prediction.matchDate))
            )

            # "Now" from the miner's perspective: predictionDate if set, else UTC now.
            prediction_dt = (
                self.prediction.predictionDate
                if isinstance(self.prediction.predictionDate, dt.datetime)
                else dt.datetime.utcnow()
            )

            # Do not look past tipoff.
            ref_dt = prediction_dt
            if ref_dt > match_dt:
                ref_dt = match_dt

            home_input = self.prediction.homeTeamName
            away_input = self.prediction.awayTeamName

            async with SportsDataIOClient() as cli:
                # Canonicalize team names using SportsDataIO Teams metadata.
                teams = await cli.teams() or []
                name_to_key: Dict[str, str] = {}

                for t in teams:
                    key = str(t.get("Key") or t.get("TeamID") or "").strip()
                    city = str(t.get("City") or "").strip()
                    name = str(t.get("Name") or "").strip()
                    full = f"{city} {name}".strip()

                    aliases = {key, city, name, full}
                    for alias in aliases:
                        if alias:
                            name_to_key[alias.lower()] = key

                def canon(x: str) -> str:
                    return name_to_key.get(x.lower(), x)

                home = canon(home_input)
                away = canon(away_input)

                # Build rolling/Elo state up to ref_dt (exclusive).
                season_start_year = ref_dt.year - 1 if ref_dt.month < 7 else ref_dt.year
                season_start = dt.datetime(season_start_year, 8, 1)
                rolling = RollingTeamState()
                elo = EloState()

                cur = season_start.date()
                end = ref_dt.date()

                async for day in cli.iterate_dates(cur, end):
                    games = await cli.games_by_date(day) or []
                    for g in games:
                        dt_str = g.get("DateTime") or g.get("Day")
                        if not dt_str:
                            continue
                        try:
                            gd = pd.to_datetime(dt_str)
                        except Exception:
                            continue

                        # Only use games strictly before ref_dt.
                        if gd >= ref_dt:
                            continue

                        h_raw = g.get("HomeTeam") or g.get("HomeTeamName") or ""
                        a_raw = g.get("AwayTeam") or g.get("AwayTeamName") or ""
                        h = canon(h_raw)
                        a = canon(a_raw)

                        hs = int(g.get("HomeTeamScore") or 0)
                        as_ = int(g.get("AwayTeamScore") or 0)
                        status = str(g.get("Status") or "").strip()

                        if status.startswith("Final") and (hs or as_):
                            rolling.push_game(
                                {
                                    "date": gd,
                                    "home": h,
                                    "away": a,
                                    "home_pts": hs,
                                    "away_pts": as_,
                                }
                            )
                            elo.update(h, a, hs, as_)

                # Injuries as of now.
                injuries = await cli.injuries() or []
                inj_summary_raw = summarize_injuries(injuries)
                inj_summary = {canon(k): v for k, v in inj_summary_raw.items()}

                # Base model features at ref_dt.
                feat_row = build_match_feature_row(
                    home=home,
                    away=away,
                    ref_date=pd.to_datetime(ref_dt),
                    elo=elo,
                    rolling=rolling,
                    inj_summary=inj_summary,
                )

                p_model_home = predict_proba(pipe, feat_row)

                # Optional: market odds blending, if odds feed is available.
                p_market_home: Optional[float] = None
                try:
                    odds_list = await cli.game_odds_by_date(match_dt.date()) or []
                    for o in odds_list:
                        h_raw = o.get("HomeTeam") or o.get("HomeTeamName") or ""
                        a_raw = o.get("AwayTeam") or o.get("AwayTeamName") or ""
                        h = canon(h_raw)
                        a = canon(a_raw)
                        if h != home or a != away:
                            continue

                        home_ml = o.get("HomeMoneyLine")
                        away_ml = o.get("AwayMoneyLine")

                        p_h = _american_to_implied_prob(home_ml)
                        p_a = _american_to_implied_prob(away_ml)
                        p_market_home = _strip_vig_two(p_h, p_a)
                        break
                except Exception as e:
                    bt.logging.warning(f"Odds lookup failed: {e}")

                hours_to_tip = max(0.0, (match_dt - ref_dt).total_seconds() / 3600.0)
                p_home = _blend_model_market(
                    p_model_home=p_model_home,
                    p_market_home=p_market_home,
                    hours_to_tip=hours_to_tip,
                )

            # Map final probability into your Prediction object.
            if p_home >= 0.5:
                self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
                self.prediction.probability = float(p_home)
            else:
                self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
                self.prediction.probability = float(1.0 - p_home)

            bt.logging.info(
                f"NBA prediction: {self.prediction.probabilityChoice} "
                f"@ {self.prediction.probability:.3f}"
            )

        except Exception as e:
            bt.logging.error(f"NBABasketballPredictionModel failed: {e}")
