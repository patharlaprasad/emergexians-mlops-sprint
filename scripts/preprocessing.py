import pandas as pd
import yaml

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TARGET_COL = params.get("target_col", "Target")

def load_data(path: str) -> pd.DataFrame:
    """Load CSV data from given path"""
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame):
    """Split dataframe into features X and target y"""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def get_feature_types(df: pd.DataFrame):
    """Identify numeric and categorical features"""
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove target from features if present
    if TARGET_COL in numeric_features:
        numeric_features.remove(TARGET_COL)
    if TARGET_COL in categorical_features:
        categorical_features.remove(TARGET_COL)

    return numeric_features, categorical_features
