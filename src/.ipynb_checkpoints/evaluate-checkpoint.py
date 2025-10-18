import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def evaluate_model():
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    model = joblib.load("model.joblib")
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.2f}")

    return acc

if __name__ == "__main__":
    evaluate_model()