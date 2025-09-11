import json
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import joblib

# Load dataset
data = pd.read_csv("data/heart_disease.csv")
X = data.drop("Target", axis=1)
y = data["Target"]

# Load trained model
model = joblib.load("models/xgboost_model.pkl")

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
cv_results = {
    "macro_f1": scores.mean(),
    "cv_scores": scores.tolist()
}

# Save metrics
with open("artifacts/cv_metrics.json", "w") as f:
    json.dump(cv_results, f, indent=4)

print("✅ CV Evaluation done. Saved to artifacts/cv_metrics.json")
