import pytest
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from os.path import abspath, dirname

model_path = abspath(dirname(dirname(__file__)))
sys.path.insert(0, f"{model_path}/src")


@pytest.fixture
def column_indices():
    return {
        "numerical": [0, 1, 2],
        "age": [3],
        "nse": [4],
        "rcc": [5],
        "cono_agrup": [6],
        "lima_prov": [7],
        "products": [9],
    }


@pytest.fixture
def categories_order():
    return {
        "nse": [["cat1", "cat2", "cat3"]],
        "rcc": [["cat4", "cat5"]],
        "lima_prov": [["cat6", "cat7", "cat8"]],
    }


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
