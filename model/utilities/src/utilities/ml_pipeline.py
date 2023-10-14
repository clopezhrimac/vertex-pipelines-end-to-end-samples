import lightgbm as lgb
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def build_ml_pipeline(
    preprocessor: ColumnTransformer, model: BaseEstimator
) -> Pipeline:
    """Build a machine learning pipeline for training.

    Args:
        preprocessor (ColumnTransformer): The preprocessor for feature transformation.
        model (BaseEstimator): The machine learning model.

    Returns:
        Pipeline: A scikit-learn Pipeline object representing the machine learning pipeline.

    Example:
        preprocessor = build_preprocessing_pipeline(column_indices, categories_order)
        model = LGBMClassifier()
        pipeline = build_ml_pipeline(preprocessor, model)
    """
    pipeline = Pipeline(
        steps=[("feature_engineering", preprocessor), ("train_model", model)]
    )
    return pipeline


# noinspection PyPep8Naming
def fit_ml_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    eval_metric: str,
):
    """Fit a machine learning pipeline to the training data.

    Args:
        pipeline (Pipeline): The machine learning pipeline.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.
        X_valid (pd.DataFrame): The validation features.
        y_valid (pd.Series): The validation labels.
        eval_metric (str): Metrics to evaluate model performance

    Returns:
        None

    Example:
        preprocessor = build_preprocessing_pipeline(column_indices, categories_order)
        model = LGBMClassifier()
        pipeline = build_ml_pipeline(preprocessor, model)
        fit_ml_pipeline(pipeline, X_train, y_train, X_valid, y_valid, "auc")
    """

    # Transform train and valid datasets
    X_train_transformed = pipeline["feature_engineering"].fit_transform(X_train)
    X_valid_transformed = pipeline["feature_engineering"].transform(X_valid)

    # Get feature names from pipeline
    transformed_col_names = pipeline["feature_engineering"].get_feature_names_out()
    feature_names = [col_name.split("__")[1] for col_name in transformed_col_names]

    pipeline.fit(
        X_train,
        y_train,
        train_model__eval_set=[
            (X_train_transformed, y_train),
            (X_valid_transformed, y_valid),
        ],
        train_model__eval_names=["train", "valid"],
        train_model__eval_metric=eval_metric,
        train_model__feature_name=feature_names,
        train_model__callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )


def calculate_feature_importance(pipeline: Pipeline) -> dict:
    """Calculate the feature importance using a LightGBM model in a pipeline.

    Args:
        pipeline (Pipeline): The pipeline with the fitted model.

    Returns:
        dict: The variable name and its importance
    """

    # Get features names and importance
    transformed_col_names = pipeline["feature_engineering"].get_feature_names_out()
    feature_names = [col_name.split("__")[1] for col_name in transformed_col_names]
    importance = pipeline["train_model"].feature_importances_.tolist()

    # Sort features by its importance
    feature_importance = dict(zip(feature_names, importance))
    sorted_importance = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )
    sorted_importance = {key: value for key, value in sorted_importance}

    return sorted_importance


# noinspection PyPep8Naming
def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a machine learning model using the provided test data.

    Args:
        pipeline (Pipeline): The fitted machine learning pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test labels.

    Returns:
        dict: A dictionary containing evaluation metrics.

    Example:
        preprocessor = build_preprocessing_pipeline(column_indices, categories_order)
        model = LGBMClassifier()
        pipeline = build_ml_pipeline(preprocessor, model)
        fit_ml_pipeline(pipeline, X_train, y_train, X_valid, y_valid, "auc")
        evaluation_results = evaluate_model(pipeline, X_test, y_test)
    """
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    evaluation_metrics = {
        "problemType": "classification",
        "auPrc": average_precision_score(y_test, y_pred_proba),
        "auRoc": roc_auc_score(y_test, y_pred_proba),
        "logLoss": log_loss(y_test, y_pred_proba),
    }

    return evaluation_metrics
