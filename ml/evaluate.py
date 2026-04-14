import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from .preprocess import SELECTED_FEATURES, TARGET


ARTIFACT_DIR = Path("ml/artifacts")


def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def main():
    test_df = pd.read_csv("data/processed/test.csv")

    X_test = test_df[SELECTED_FEATURES]
    y_test = test_df[TARGET]

    model = joblib.load(ARTIFACT_DIR / "best_model.joblib")

    y_pred = model.predict(X_test)

    metrics = {
        "test_rmse": float(rmse(y_test, y_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_r2": float(r2_score(y_test, y_pred)),
        "test_count": int(len(test_df))
    }

    print(metrics)

    with open(ARTIFACT_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()