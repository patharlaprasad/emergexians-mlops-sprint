import os
import subprocess
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# ========================
# Suppress XGBoost Warnings
# ========================
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["XGB_DISABLE_DEPRECATION_WARNINGS"] = "1"

# ========================
# Load Dataset
# ========================
df = pd.read_csv("data/heart_disease.csv")
print(f"Dataset shape: {df.shape}")
print("Target distribution:\n", df["Target"].value_counts())

X = df.drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========================
# MLflow Tracking
# ========================
mlflow.set_experiment("heart_disease_xgboost")

with mlflow.start_run():
    # ---- Add MLflow tags for reproducibility ----
    try:
        code_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
    except Exception:
        code_sha = "unknown"

    try:
        dataset_ver = subprocess.check_output(
            ["git", "log", "-n", "1", "--pretty=format:%h", "--", "data.dvc"]
        ).decode("utf-8").strip()
    except Exception:
        dataset_ver = "unknown"

    mlflow.set_tag("code_sha", code_sha)
    mlflow.set_tag("dataset_ver", dataset_ver)
    mlflow.set_tag("stage", "train")

    mlflow.log_param("code_sha", code_sha)
    mlflow.log_param("dataset_ver", dataset_ver)
    mlflow.log_param("stage", "train")

    # ========================
    # Load Hyperparameters
    # ========================
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    xgb_params = params["xgboost"]

    # ========================
    # Build Pipeline
    # ========================
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=10)),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    use_label_encoder=False,
                    random_state=42,
                    **xgb_params,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    best_model = pipeline

    # ========================
    # Evaluation
    # ========================
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    # ========================
    # Save Metrics (for DVC)
    # ========================
    os.makedirs("artifacts", exist_ok=True)
    metrics = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
    }
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # ========================
    # Save Model
    # ========================
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/xgboost_model.pkl")

    # Save pipeline parts
    joblib.dump(best_model.named_steps["scaler"], "artifacts/scaler.pkl")
    joblib.dump(best_model.named_steps["selector"], "artifacts/selector.pkl")

    # Feature importance
    model = best_model.named_steps["model"]
    importances = model.feature_importances_

    selected_features = best_model.named_steps["selector"].get_support(indices=True)
    feature_names = X.columns[selected_features]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance (XGBoost)")
    plt.savefig("artifacts/feature_importance.png")
    plt.close()

    # ========================
    # Log to MLflow
    # ========================
    mlflow.log_params(xgb_params)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_auc", test_auc)

    mlflow.log_artifacts("artifacts")
    mlflow.log_artifact("models/xgboost_model.pkl")

print("\n✅ Training completed! Metrics saved to 'artifacts/metrics.json' and model saved to 'models/xgboost_model.pkl'")
