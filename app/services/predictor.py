import json
from pathlib import Path

import joblib
import pandas as pd

from app.schemas.extraction import ExtractedFeatures
from app.schemas.response import PredictionSummary
from ml.preprocess import SELECTED_FEATURES


ARTIFACT_DIR = Path("ml/artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
STATS_PATH = ARTIFACT_DIR / "training_stats.json"


class PredictorService:
    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)

        with open(STATS_PATH, "r", encoding="utf-8") as f:
            self.training_stats = json.load(f)

    def features_to_row(self, features: ExtractedFeatures) -> pd.DataFrame:
        feature_dict = features.model_dump()
        row = {col: feature_dict.get(col) for col in SELECTED_FEATURES}
        return pd.DataFrame([row])

    def predict(self, features: ExtractedFeatures) -> PredictionSummary:
        X = self.features_to_row(features)
        predicted_price = float(self.model.predict(X)[0])

        stats = self.training_stats
        q1 = float(stats["q1_price"])
        q3 = float(stats["q3_price"])

        if predicted_price < q1:
            relative_position = "below_typical_range"
        elif predicted_price > q3:
            relative_position = "above_typical_range"
        else:
            relative_position = "within_typical_range"

        return PredictionSummary(
            predicted_price=predicted_price,
            formatted_price=f"${predicted_price:,.0f}",
            median_price=float(stats["median_price"]),
            mean_price=float(stats["mean_price"]),
            q1_price=q1,
            q3_price=q3,
            relative_position=relative_position,
        )

    def get_model_name(self) -> str:
        return str(self.training_stats.get("best_model", "unknown_model"))