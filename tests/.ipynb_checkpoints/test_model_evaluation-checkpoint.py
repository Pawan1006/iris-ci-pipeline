import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy_threshold():
    model = joblib.load("model.joblib")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc >= 0.7, f"Accuracy too low: {acc}"