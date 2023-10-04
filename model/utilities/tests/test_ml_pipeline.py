from utilities.ml_pipeline import build_ml_pipeline, fit_ml_pipeline, calculate_feature_importance, evaluate_model
from unittest.mock import MagicMock

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator

import lightgbm as lgb
import numpy as np
import pytest


def test_build_ml_pipeline_output_type():
    """
    Test if the output type of build_ml_pipeline is correct.
    """
    preprocessor = ColumnTransformer([])
    model = lgb.LGBMClassifier()
    pipeline = build_ml_pipeline(preprocessor, model)

    assert isinstance(pipeline, Pipeline)


def test_build_ml_pipeline_generate_two_steps():
    """
    Test if the generated pipeline has two steps: 'feature_engineering' and 'train_model'.
    """
    preprocessor = ColumnTransformer([])
    model = lgb.LGBMClassifier()

    pipeline = build_ml_pipeline(preprocessor, model)
    steps = pipeline.steps

    assert len(steps) == 2
    assert isinstance(steps[0][1], ColumnTransformer)
    assert isinstance(steps[1][1], BaseEstimator)
    assert steps[0][0] == "feature_engineering"
    assert steps[1][0] == "train_model"


def test_fit_ml_pipeline_expected_method_calls(synthetic_data):
    """
    Asserts that the 'fit', 'fit_transform', 'transform', and 'get_feature_names_out' methods are called as expected.
    """

    # Use synthetic data
    X_train, X_valid, y_train, y_valid = synthetic_data

    # Create magic mock object
    pipeline = MagicMock()

    fit_ml_pipeline(pipeline, X_train, y_train, X_valid, y_valid, eval_metric="auc")

    # Asserts expected calls
    pipeline.fit.assert_called_once()
    pipeline["feature_engineering"].fit_transform.assert_called_once_with(X_train)
    pipeline["feature_engineering"].transform.assert_called_once_with(X_valid)
    pipeline["feature_engineering"].get_feature_names_out.assert_called_once()


def test_calculate_feature_importance_output_type(pipeline_mock, feature_importances):
    """
    Test the output type of 'calculate_feature_importance' function.
    """
    result = calculate_feature_importance(pipeline_mock)

    assert isinstance(result, dict)


def test_calculate_feature_importance_correct_keys(pipeline_mock):
    """
    Test the correct keys in the output of 'calculate_feature_importance'
    """
    expected_keys = ["feature1", "feature2", "feature3", "feature4"]
    result = calculate_feature_importance(pipeline_mock)

    assert set(result.keys()) == set(expected_keys)


def test_calculate_feature_importance_sorted_importance(pipeline_mock):
    """
    Test the sorted importance values in the output of 'calculate_feature_importance'
    """
    result = calculate_feature_importance(pipeline_mock)
    sorted_importance = sorted(result.values(), reverse=True)

    assert list(result.values()) == sorted_importance


@pytest.mark.parametrize(
    "feature_importances, expected_importances",
    [
        (np.array([5, 6, 7, 8]), {"feature4": 8, "feature3": 7, "feature2": 6, "feature1": 5}),
        (np.array([0.5, 0.6, 0.7, 0.8]), {"feature4": 0.8, "feature3": 0.7, "feature2": 0.6, "feature1": 0.5}),
    ],
)
def test_calculate_feature_importance_correct_values(pipeline_mock, feature_importances, expected_importances):
    """
    Test the correctness of importance values in the output of 'calculate_feature_importance'
    """
    result = calculate_feature_importance(pipeline_mock)

    assert result == expected_importances


def test_evaluate_model_calls_predict_proba(sample_test_data):
    """
    Assert evaluate model calls predict_proba
    """
    X_test, y_test, y_pred_proba = sample_test_data

    pipeline = MagicMock()
    pipeline.predict_proba = MagicMock(return_value=y_pred_proba)

    evaluate_model(pipeline, X_test, y_test)

    pipeline.predict_proba.assert_called_once_with(X_test)


def test_evaluate_model_calls_output_structure(sample_test_data):
    """
    Assert evaluate model output structure is correct
    """
    X_test, y_test, y_pred_proba = sample_test_data

    pipeline = MagicMock()
    pipeline.predict_proba = MagicMock(return_value=y_pred_proba)

    result = evaluate_model(pipeline, X_test, y_test)

    expected_keys = {"problemType", "auPrc", "auRoc", "logLoss"}
    assert isinstance(result, dict)
    assert set(result.keys()) == expected_keys
    assert result["problemType"] == "classification"
