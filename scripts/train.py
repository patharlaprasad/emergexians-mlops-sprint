import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------
# 1. Load parameters
# ---------------------------
params = yaml.safe_load(open("params.yaml"))

test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]
C = params["train"]["C"]   # regularization strength for Logistic Regression

# ---------------------------
# 2. Load dataset (NEW HEART DISEASE DATASET)
# ---------------------------
df = pd.read_csv("data/heart_disease.csv")   # <--- this is the dataset we added
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# ---------------------------
# 3. Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# ---------------------------
# 4. Train model
# ---------------------------
model = LogisticRegression(C=C, max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------
# 5. Evaluate model
# ---------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

# ---------------------------
# 6. Log to MLflow
# ---------------------------
mlflow.set_experiment("Heart Disease Prediction")

with mlflow.start_run():
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("C", C)
    mlflow.log_metric("accuracy", acc)

    # log model
    mlflow.sklearn.log_model(model, "model")

    # save model locally
    joblib.dump(model, "models/heart_disease_model.pkl")
