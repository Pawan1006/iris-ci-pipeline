import pandas as pd
from sklearn.datasets import load_iris

def test_iris_data_shape():
    iris = load_iris(as_frame=True)
    assert iris.data.shape[1] == 4, "Iris dataset must have 4 features"