from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

app = FastAPI(title="Iris Prediction API", version="1.0")
mlflow.set_tracking_uri("file:///home/jupyter/iris-ci-pipeline/mlruns")
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load the best model from MLflow once at startup
def load_best_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Iris_Pipeline_MLflow")

    if experiment is None:
        raise Exception("Experiment 'Iris_Pipeline_MLflow' not found. Run train.py first to log models!")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if not runs:
        raise Exception("No runs found in MLflow experiment. Please train a model first!")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    print(f"Best run ID: {best_run_id}")

    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model


model = load_best_model()

@app.get("/")
def root():
    return {"message": "Iris Model API is running!"}

@app.post("/predict")
def predict(features: IrisFeatures):
    data = pd.DataFrame([features.dict()])
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}
