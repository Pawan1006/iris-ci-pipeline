import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def load_data():
    iris = load_iris(as_frame=True)
    return iris.data, iris.target

def evaluate_best_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    client = MlflowClient()
    experiment = client.get_experiment_by_name("Iris_Pipeline_MLflow")

    # get best run based on accuracy
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    print(f"Best run ID: {best_run_id}")

    # load best model
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Evaluation Accuracy:", acc)

if __name__ == "__main__":
    evaluate_best_model()