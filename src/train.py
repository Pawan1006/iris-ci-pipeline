import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# ✅ Ensure writable local directory
os.makedirs("mlruns", exist_ok=True)

# ✅ Point MLflow tracking to local directory (works in CI)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Iris_Pipeline_MLflow")

def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y

def train_and_log():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for n_estimators in [50, 100, 150]:
        for max_depth in [3, 5, 8]:
            with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}"):
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                # ✅ Log params & metrics
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_metric("accuracy", acc)

                # ✅ Log model safely inside repo directory
                mlflow.sklearn.log_model(model, artifact_path="model")

                print(f"✅ n_estimators={n_estimators}, max_depth={max_depth}, acc={acc:.4f}")

if __name__ == "__main__":
    train_and_log()
