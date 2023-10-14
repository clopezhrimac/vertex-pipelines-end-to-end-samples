import pytest
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from os.path import abspath, dirname

model_path = abspath(dirname(dirname(__file__)))
sys.path.insert(0, f"{model_path}/src")

# Example data
train_data = "id,name,age\n1,Alice,25\n2,Bob,\n3,Charlie,35\n"
valid_data = "id,name,age\n4,David,40\n5,,45\n6,Frank,50\n"
test_data = "id,name,age\n7,Grace,55\n8,Henry,\n9,Ivy,65\n"


class MockModel:
    def __init__(self, importances):
        self.feature_importances_ = importances


class MockFeatureEngineeringStep:
    def __init__(self, names):
        self.feature_names = names

    def get_feature_names_out(self):
        return [
            "__".join(["transformer_name", feature]) for feature in self.feature_names
        ]


@pytest.fixture
def csv_data_paths(tmpdir):
    train_path = tmpdir.join("train.csv")
    valid_path = tmpdir.join("valid.csv")
    test_path = tmpdir.join("test.csv")

    train_path.write(train_data)
    valid_path.write(valid_data)
    test_path.write(test_data)

    return str(train_path), str(valid_path), str(test_path)


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    num_samples = 1000
    num_features = 10
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(2, size=num_samples)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_valid, y_train, y_valid


@pytest.fixture
def sample_test_data():
    y_pred_proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    y_test = pd.Series([1, 1, 0, 1])
    X_test = pd.DataFrame(np.random.rand(4, 10))

    return X_test, y_test, y_pred_proba


@pytest.fixture
def dataframe_test():
    data = {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "label": ["A", "B", "C"]}
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def feature_importances():
    return np.array([5, 6, 7, 8])


@pytest.fixture
def feature_names():
    return ["feature1", "feature2", "feature3", "feature4"]


@pytest.fixture
def pipeline_mock(feature_importances, feature_names):
    pipeline = Pipeline(
        [
            ("feature_engineering", MockFeatureEngineeringStep(feature_names)),
            ("train_model", MockModel(feature_importances)),
        ]
    )

    return pipeline


@pytest.fixture
def metrics():
    metrics = {
        "problemType": "classification",
        "auPrc": 0.15,
        "auRoc": 0.75,
        "logLoss": 0.15,
    }
    return metrics


@pytest.fixture
def sample_transformer_data():
    return pd.DataFrame({"text_column": ["apple", "banana", "cherry", "date", np.nan]})
