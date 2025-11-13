import asyncio
import datetime as dt
import pandas as pd
import bittensor as bt
from st.sport_prediction_model import SportPredictionModel
from common.data import ProbabilityChoice

# Import our pipeline utilities
from st.models.impl.basketball_nba.nba_sportsdataio_client import SportsDataIOClient
from st.models.impl.basketball_nba.nba_features import (
    EloState,
    RollingTeamState,
    summarize_injuries,
    build_match_feature_row,
)
from st.models.impl.basketball_nba.nba_model import load_model, predict_proba


class NBABasketballPredictionModel(SportPredictionModel):
    async def make_prediction(self):
        bt.logging.info(
            "Predicting NBA basketball game with SportsDataIO + ML model..."
        )
        try:
            pipe = load_model()
            if pipe is None:
                bt.logging.error(
                    "NBA model not found at models/nba_winprob.joblib. Run nba_update.py first."
                )
                return

            match_dt = (
                self.prediction.matchDate
                if isinstance(self.prediction.matchDate, dt.datetime)
                else dt.datetime.fromisoformat(str(self.prediction.matchDate))
            )
            home_input = self.prediction.homeTeamName
            away_input = self.prediction.awayTeamName
            ref_date = match_dt

            async with SportsDataIOClient() as cli:
                # Build mapping between various name variants and canonical team key (abbreviation)
                teams = await cli.teams() or []
                name_to_key = {}
                for t in teams:
                    key = str(t.get("Key") or t.get("TeamID") or "").strip()
                    city = str(t.get("City") or "").strip()
                    name = str(t.get("Name") or "").strip()
                    full = f"{city} {name}".strip()
                    for alias in {key, city, name, full}:
                        if alias:
                            name_to_key[alias.lower()] = key

                def canon(x: str) -> str:
                    return name_to_key.get(x.lower(), x)

                # Determine canonical keys for input teams
                home = canon(home_input)
                away = canon(away_input)

                season_start = dt.datetime(
                    ref_date.year - 1 if ref_date.month < 7 else ref_date.year, 8, 1
                )
                rolling = RollingTeamState()
                elo = EloState()

                # iterate through dates to update history
                cur = season_start.date()
                end = (ref_date - dt.timedelta(days=1)).date()
                async for day in cli.iterate_dates(cur, end):
                    games = await cli.games_by_date(day) or []
                    for g in games:
                        dt_str = g.get("DateTime") or g.get("Day")
                        if not dt_str:
                            continue
                        try:
                            gd = dt.datetime.fromisoformat(dt_str.replace("Z", ""))
                        except Exception:
                            try:
                                gd = pd.to_datetime(dt_str).to_pydatetime()
                            except Exception:
                                continue
                        h = canon(g.get("HomeTeam") or g.get("HomeTeamName") or "")
                        a = canon(g.get("AwayTeam") or g.get("AwayTeamName") or "")
                        hs = int(g.get("HomeTeamScore") or 0)
                        as_ = int(g.get("AwayTeamScore") or 0)
                        if g.get("Status", "").startswith("Final") and (hs or as_):
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

                injuries = await cli.injuries() or []
                # summarize by canonical team key
                inj_summary_raw = summarize_injuries(injuries)
                inj_summary = {canon(k): v for k, v in inj_summary_raw.items()}

            feat_row = build_match_feature_row(
                home, away, pd.to_datetime(ref_date), elo, rolling, inj_summary
            )

            # Predict home win probability
            p_home = predict_proba(pipe, feat_row)

            if p_home >= 0.5:
                self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
                self.prediction.probability = float(p_home)
            else:
                self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
                self.prediction.probability = float(1.0 - p_home)

            bt.logging.info(
                f"NBA prediction: {self.prediction.probabilityChoice} @ {self.prediction.probability:.3f}"
            )
        except Exception as e:
            bt.logging.error(f"NBABasketballPredictionModel failed: {e}")
