"""
Model training + inference utilities for NBA game outcome probability.

- Uses scikit-learn HistGradientBoostingClassifier with isotonic calibration.
- Trains on features built with nba_features.RollingTeamState + EloState.
- Persists model with joblib and metadata JSON.

NOTE: Training is "static" (no odds, no time-to-tip), but inference can later
blend with market odds dynamically, closer to tipoff.
"""

from __future__ import annotations

import os
import json
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import joblib

# Feature names kept in sync with nba_features.build_match_feature_row
NUMERIC_FEATURES = [
    "elo_diff",
    "home_win_pct_l10",
    "away_win_pct_l10",
    "pdiff_l10",
    "home_days_since",
    "away_days_since",
    "home_b2b",
    "away_b2b",
    "home_gp_l10",
    "away_gp_l10",
    "inj_home_ct",
    "inj_away_ct",
]


@dataclass
class ModelMetadata:
    trained_at: str
    seasons: List[int]
    n_samples: int
    features: List[str]
    model_class: str = "HistGradientBoostingClassifier+Isotonic"
    calibration: str = "isotonic"
    notes: str = "Pre-game features; odds blending handled at inference (if used)."


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(with_mean=True, with_std=True),
                NUMERIC_FEATURES,
            )
        ],
        remainder="drop",
    )

    base = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.05,
        max_iter=500,
        min_samples_leaf=20,
        l2_regularization=0.0,
    )

    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe


def train_model(
    df: pd.DataFrame, seasons: List[int]
) -> Tuple[Pipeline, ModelMetadata, Dict[str, float]]:
    """
    Train the calibrated classifier and return (pipeline, metadata, metrics).

    Expects df columns:
    - all NUMERIC_FEATURES
    - date
    - home_win (0/1)
    """
    df = df.dropna(subset=NUMERIC_FEATURES + ["home_win"]).copy()
    X = df[NUMERIC_FEATURES]
    y = df["home_win"].astype(int)

    pipe = build_pipeline()
    pipe.fit(X, y)

    metrics: Dict[str, float] = {}

    if len(df) > 200:
        df_sorted = df.sort_values("date")
        split = int(len(df_sorted) * 0.9)
        train, test = df_sorted.iloc[:split], df_sorted.iloc[split:]
        if len(test) > 50:
            p = pipe.predict_proba(test[NUMERIC_FEATURES])[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(test["home_win"], p >= 0.5)),
                "brier": float(brier_score_loss(test["home_win"], p)),
                "log_loss": float(
                    log_loss(
                        test["home_win"],
                        np.clip(p, 1e-6, 1 - 1e-6),
                    )
                ),
            }

    meta = ModelMetadata(
        trained_at=dt.datetime.utcnow().isoformat(),
        seasons=seasons,
        n_samples=int(len(df)),
        features=list(NUMERIC_FEATURES),
    )

    return pipe, meta, metrics


def save_model(
    pipe: Pipeline,
    meta: ModelMetadata,
    model_dir: str = "./models",
    model_name: str = "nba_winprob.joblib",
) -> str:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    joblib.dump({"pipeline": pipe, "meta": asdict(meta)}, model_path)
    with open(
        os.path.join(model_dir, model_name.replace(".joblib", ".meta.json")),
        "w",
    ) as f:
        json.dump(asdict(meta), f, indent=2)
    return model_path


def load_model(
    model_dir: str = "./models",
    model_name: str = "nba_winprob.joblib",
):
    path = os.path.join(model_dir, model_name)
    if not os.path.exists(path):
        return None
    obj = joblib.load(path)
    return obj["pipeline"]


def predict_proba(pipe, feat_row: Dict[str, float]) -> float:
    """
    Predict the home win probability given a single feature row dict.
    """
    X = pd.DataFrame([feat_row], columns=NUMERIC_FEATURES)
    proba = pipe.predict_proba(X)[:, 1][0]
    return float(proba)
