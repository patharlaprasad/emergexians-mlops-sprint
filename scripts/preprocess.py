import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt

# Paths
DATA_PATH = "data/heart_disease.csv"
MODEL_DIR = "models"
METRICS_DIR = "metrics"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
print(df.duplicated().sum())
print(df.shape)
print(df["Target"].value_counts())

print(f"📂 Raw dataset shape: {df.shape}")

X = df.drop("Target", axis=1)
y = df["Target"]

print(f"Feature shape: {X.shape} | Target shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"🔀 Train size: {X_train.shape} Test size: {X_test.shape}")

# Feature names
numeric_features = list(X.select_dtypes(include=np.number).columns)
categorical_features = list(X.select_dtypes(exclude=np.number).columns)
print(f"Numeric Features: {numeric_features}")
print(f"Categorical Features: {categorical_features}")

# Models + pipelines
models = {
    "log_reg": Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=8)),   # pick top 8 features
        ("classifier", LogisticRegression(max_iter=500))
    ]),
    "random_forest": Pipeline([
        ("classifier", RandomForestClassifier(random_state=42))
    ]),
    "xgboost": Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=8)),
        ("classifier", XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ))
    ])
}

# Hyperparameters
params = {
    "log_reg": {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"]
    },
    # Random Forest Grid Search params (update these)
    "random_forest": {
        "classifier": [RandomForestClassifier(random_state=42, class_weight="balanced")],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5, 7, 10],  # limit depth
        "classifier__min_samples_split": [5, 10, 20],
        "classifier__min_samples_leaf": [5, 10, 20],
        "classifier__max_features": ["sqrt", "log2"]
    },
    "xgboost": {
        "classifier__n_estimators": [100, 200, 500],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0]
    }
}

best_models = {}
metrics = {}

# Train + Evaluate
for name, pipeline in models.items():
    print(f"\n🔎 Training {name} ...")
    grid = GridSearchCV(
        pipeline, params[name], cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Best Params for {name}: {grid.best_params_}")
    print(f"Train Accuracy: {accuracy_score(y_train, best_model.predict(X_train)):.2f}")
    print(f"Test Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cv_acc = cross_val_score(best_model, X, y, cv=5, scoring="accuracy").mean()
    cv_f1 = cross_val_score(best_model, X, y, cv=5, scoring="f1").mean()
    print(f"CV Accuracy: {cv_acc:.2f}, CV F1: {cv_f1:.2f}")

    best_models[name] = best_model
    metrics[name] = {
        "test_accuracy": acc,
        "test_f1": f1,
        "cv_accuracy": cv_acc,
        "cv_f1": cv_f1,
        "best_params": grid.best_params_
    }

# Pick best model
best_model_name = max(metrics, key=lambda m: metrics[m]["test_accuracy"])
best_model = best_models[best_model_name]

joblib.dump(best_model, os.path.join(MODEL_DIR, f"{best_model_name}_model.pkl"))
print(f"\n✅ Best Model: {best_model_name} (Test Acc: {metrics[best_model_name]['test_accuracy']:.2f}) saved to {MODEL_DIR}/{best_model_name}_model.pkl")

# Save metrics
with open(os.path.join(METRICS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
print(f"\n📊 Metrics saved to {METRICS_DIR}/metrics.json")

# Feature importance (for tree models only)
if best_model_name in ["random_forest", "xgboost"]:
    clf = best_model.named_steps["classifier"]
    importances = clf.feature_importances_
    feature_names = X.columns if "select" not in best_model.named_steps else \
        X.columns[best_model.named_steps["select"].get_support()]

    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.title(f"{best_model_name} Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, f"{best_model_name}_feature_importance.png"))
    print(f"📌 Saved feature importance plot to {METRICS_DIR}/{best_model_name}_feature_importance.png")
