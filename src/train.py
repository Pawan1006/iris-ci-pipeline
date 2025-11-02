import os, shutil
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Clean up
if os.path.exists("mlruns"):
    shutil.rmtree("mlruns")
os.makedirs("mlruns", exist_ok=True)

# Set tracking path
os.environ["MLFLOW_TRACKING_URI"] = f"file://{os.path.abspath('./mlruns')}"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Create experiment safely
if "Iris_Pipeline_MLflow" not in [exp.name for exp in mlflow.search_experiments()]:
    mlflow.create_experiment(
        "Iris_Pipeline_MLflow",
        artifact_location=os.environ["MLFLOW_TRACKING_URI"]
    )
mlflow.set_experiment("Iris_Pipeline_MLflow")

def load_data():
    iris = load_iris(as_frame=True)
    return iris.data, iris.target

def train_and_log():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for n in [50, 100, 150]:
        for d in [3, 5, 8]:
            with mlflow.start_run(run_name=f"RF_n{n}_d{d}"):
                model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                mlflow.log_params({"n_estimators": n, "max_depth": d})
                mlflow.log_metric("accuracy", acc)
                print(">>> Artifact URI:", mlflow.get_artifact_uri())
                mlflow.sklearn.log_model(model, artifact_path="model")

if __name__ == "__main__":
    train_and_log()
