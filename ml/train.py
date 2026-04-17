import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from .preprocess import make_preprocessor, SELECTED_FEATURES, TARGET


ARTIFACT_DIR = Path("ml/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def evaluate_model(name, pipeline, X_train, y_train, X_val, y_val):
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)

    return {
        "model_name": name,
        "pipeline": pipeline,
        "train_rmse": rmse(y_train, train_pred),
        "val_rmse": rmse(y_val, val_pred),
        "val_mae": mean_absolute_error(y_val, val_pred),
        "val_r2": r2_score(y_val, val_pred),
    }


def main():
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    X_train = train_df[SELECTED_FEATURES]
    y_train = train_df[TARGET]

    X_val = val_df[SELECTED_FEATURES]
    y_val = val_df[TARGET]

    preprocessor = make_preprocessor()

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01, max_iter=10000),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        result = evaluate_model(name, pipeline, X_train, y_train, X_val, y_val)
        results.append(result)

    results_df = pd.DataFrame([
        {
            "model_name": r["model_name"],
            "train_rmse": r["train_rmse"],
            "val_rmse": r["val_rmse"],
            "val_mae": r["val_mae"],
            "val_r2": r["val_r2"],
        }
        for r in results
    ]).sort_values("val_rmse")

    print("\nModel Comparison:")
    print(results_df)

    results_df.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    best_result = min(results, key=lambda x: x["val_rmse"])
    best_pipeline = best_result["pipeline"]

    joblib.dump(best_pipeline, ARTIFACT_DIR / "best_model.joblib")

    stats = {
        "target": TARGET,
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "median_price": float(train_df[TARGET].median()),
        "mean_price": float(train_df[TARGET].mean()),
        "min_price": float(train_df[TARGET].min()),
        "max_price": float(train_df[TARGET].max()),
        "q1_price": float(train_df[TARGET].quantile(0.25)),
        "q3_price": float(train_df[TARGET].quantile(0.75)),
        "best_model": best_result["model_name"],
        "validation_rmse": float(best_result["val_rmse"]),
        "validation_mae": float(best_result["val_mae"]),
        "validation_r2": float(best_result["val_r2"]),
    }

    with open(ARTIFACT_DIR / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nBest model saved: {best_result['model_name']}")
    print(f"Best validation RMSE: {best_result['val_rmse']:.4f}")


if __name__ == "__main__":
    main()