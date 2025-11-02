# Week 5: MLflow Integration in ML Pipeline

## 🎯 Objective
The objective of this assignment is to integrate **MLflow** into the existing Iris Machine Learning pipeline to enable experiment tracking, hyperparameter tuning, and model versioning.  
This includes:
- Introducing hyperparameter tuning as part of the training loop.
- Logging parameters, metrics, and models using MLflow.
- Comparing experiments visually through the MLflow dashboard.
- Removing model tracking dependency from DVC.
- Fetching and evaluating the best/latest model from the MLflow Model Registry.

---

## 🧩 Folder Structure and Utility of Each File

```
week_5/
├── README.md                 → Project description and usage instructions.
│
├── data/                     → Contains test data for model evaluation.
│   ├── X_test.csv
│   └── y_test.csv
│
├── requirements.txt          → Python package dependencies for this project.
│
├── src/
│   ├── train.py              → Trains model, performs hyperparameter tuning, 
│   │                            logs parameters, metrics, and models to MLflow.
│   ├── evaluate.py           → Fetches latest/best model from MLflow Model Registry 
│   │                            and evaluates it on test data.
│
├── params.yaml               → (Optional) Reference file listing hyperparameters used.
│
└── mlruns/                   → Auto-created by MLflow to store experiment data, 
                               including runs, metrics, parameters, and model artifacts.
```

---

## ⚙️ Environment Setup

### 1. Create and activate a virtual environment (optional)
```bash
python3 -m venv venv
source venv/bin/activate     # On Linux/Mac
venv\Scripts\activate        # On Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Pipeline

### 1. Run training
This step performs hyperparameter tuning and logs all runs to MLflow.
```bash
python src/train.py
```

### 2. Run evaluation
This step fetches the latest/best model from the MLflow Model Registry and evaluates it.
```bash
python src/evaluate.py
```

### 3. Launch MLflow UI
To visualize experiments and compare metrics:
```bash
mlflow ui --host 0.0.0.0 --port 5001 --cors-allowed-origins="*" --allowed-hosts="*"
```

Then open in your browser:
```
http://<YOUR_INSTANCE_EXTERNAL_IP>:5001
```

---

## 📊 Results and Observations
- MLflow successfully tracks parameters, metrics, and models.
- Multiple experiment runs can be compared visually in the MLflow UI.
- The evaluation pipeline loads the latest/best model from MLflow Registry.
- Best model achieved **accuracy = 1.0** on test data.
- DVC model tracking removed; only MLflow is used for model versioning.

---

## 🧱 requirements.txt (for reference)
Ensure the following key packages are included:
```
mlflow
scikit-learn
pandas
numpy
joblib
```

---

## 🏁 Conclusion
This assignment demonstrates how to:
- Integrate MLflow into an ML pipeline.
- Track and compare experiments efficiently.
- Manage models using the MLflow Model Registry.
- Simplify model evaluation without relying on DVC for model storage.

The setup is modular, extendable, and ready for integration with CI/CD workflows in future tasks.
