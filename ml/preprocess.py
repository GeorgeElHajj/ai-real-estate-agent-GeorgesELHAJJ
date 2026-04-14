"""
Feature Preprocessing Pipeline
================================

Handles:
1. Feature Scaling: StandardScaler for numeric features
2. Feature Encoding: OneHotEncoder for categorical features  
3. Log Transformation: For skewed numeric features
4. Imputation: As safety net (missing values already handled in notebook)

Feature Groups:
- LOG_FEATURES: Right-skewed features requiring log1p transformation + scaling
- PLAIN_NUMERIC_FEATURES: Normal distribution numeric features + scaling
- CATEGORICAL_FEATURES: Categorical features + one-hot encoding
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import numpy as np

# ────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS (must match step2-ml-pipeline-modeling.ipynb)
# ────────────────────────────────────────────────────────────────

SELECTED_FEATURES = [
    'OverallQual',
    'GrLivArea',
    'Neighborhood',
    'TotalBsmtSF',
    'GarageCars',
    'FullBath',
    'LotArea',
    'BedroomAbvGr',
    'HouseStyle',
    'HouseAge'
]

TARGET = "SalePrice"

NUMERIC_FEATURES = [
    'OverallQual',
    'GrLivArea',
    'TotalBsmtSF',
    'GarageCars',
    'FullBath',
    'LotArea',
    'BedroomAbvGr',
    'HouseAge'
]

CATEGORICAL_FEATURES = [
    'Neighborhood',
    'HouseStyle'
]

# Features with high right skewness - benefit from log transformation
LOG_FEATURES = [
    'GrLivArea',
    'TotalBsmtSF',
    'LotArea'
]

PLAIN_NUMERIC_FEATURES = [
    col for col in NUMERIC_FEATURES if col not in LOG_FEATURES
]


def make_preprocessor():
    """
    Creates a ColumnTransformer preprocessing pipeline.
    
    Processing order:
    1. LOG_FEATURES → Impute (median) → Log1p transform → StandardScale
    2. PLAIN_NUMERIC_FEATURES → Impute (median) → StandardScale  
    3. CATEGORICAL_FEATURES → Impute (mode) → OneHotEncode
    
    Returns:
        ColumnTransformer: Fitted preprocessing pipeline
    """
    
    # ─── Log-transformed numeric features (right-skewed) ───────────
    log_numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler())
    ])

    # ─── Standard numeric features (normal-ish distribution) ────────
    plain_numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # ─── Categorical features (one-hot encoding) ──────────────────
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # ─── Combine all pipelines ────────────────────────────────────
    preprocessor = ColumnTransformer([
        ("log_num", log_numeric_pipeline, LOG_FEATURES),
        ("num", plain_numeric_pipeline, PLAIN_NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ], verbose_feature_names_out=False)

    return preprocessor