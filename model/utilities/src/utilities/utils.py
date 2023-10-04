import pandas as pd
import joblib
import logging
import json

from pathlib import Path

from sklearn.pipeline import Pipeline


def read_datasets(train_path: str, valid_path: str, test_path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Read datasets from CSV files.

    Args:
        train_path (str): Path to the training dataset CSV file.
        valid_path (str): Path to the validation dataset CSV file.
        test_path (str): Path to the test dataset CSV file.

    Returns:
        tuple: A tuple containing the pandas DataFrames for the training,
               validation, and test datasets.
    """
    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_valid, df_test


def split_xy(df: pd.DataFrame, label: str) -> (pd.DataFrame, pd.Series):
    """Split a DataFrame into features (X) and labels (y).

    Args:
        df (pd.DataFrame): The input DataFrame.
        label (str): The name of the column containing the labels.

    Returns:
        tuple: A tuple containing two objects:
            - X (pd.DataFrame): The DataFrame containing the features.
            - y (pd.Series): The Series containing the labels.

    Raises:
        KeyError: If the specified label column does not exist in the DataFrame.
    """
    return df.drop(columns=[label]), df[label]


def indices_in_list(elements: list, base_list: list) -> list:
    """Get the indices of specific elements in a base list.

    Args:
        elements (list): The list of elements to search for.
        base_list (list): The base list in which to search for the elements.

    Returns:
        list: A list of indices corresponding to the positions of the elements
        in the base list.

    Example:
        elements = [2, 5, 8]
        base_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        indices_in_list(elements, base_list)
        [1, 4, 7]
    """
    return [idx for idx, elem in enumerate(base_list) if elem in elements]


def save_model_artifact(model_artifact: Pipeline, model_path: str):
    """Save the trained model to a file.

    Args:
        model_artifact (Pipeline): The trained model object.
        model_path (str): The path to save the model.

    Returns:
        None
    """
    if model_path.startswith("gs://"):
        model_path = Path("/gcs/" + model_path[5:])
    else:
        model_path = Path(model_path)

    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, str(model_path / "model.joblib"))


def save_metrics(metrics: dict, metrics_path: str):
    """Save the evaluation metrics to a JSON file.

    Args:
        metrics (Dict[str, float]): A dictionary containing the evaluation metrics.
        metrics_path (str): The path to save the metrics.

    Returns:
        None
    """
    with open(metrics_path, "w") as fp:
        json.dump(metrics, fp)


def save_training_dataset_metadata(model_path: str, train_data_path: str, label: str):
    """Save training dataset information for model monitoring.

    This function persists URIs of training file(s) for model monitoring in batch predictions.

    Args:
        model_path (str): Path to the model directory.
        train_data_path (str): Path to the training data file.
        label (str): Name of the target field.

    Returns:
        None
    """

    training_dataset_info = "training_dataset_info.json"

    if model_path.startswith("gs://"):
        model_path = Path("/gcs/" + model_path[5:])
    else:
        model_path = Path(model_path)

    path = Path(model_path) / training_dataset_info
    training_dataset_for_monitoring = {
        "gcsSource": {"uris": [train_data_path]},
        "dataFormat": "csv",
        "targetField": label,
    }

    logging.info(f"Training dataset info: {training_dataset_for_monitoring}")

    with open(path, "w") as fp:
        logging.info(f"Save training dataset info for model monitoring: {path}")
        json.dump(training_dataset_for_monitoring, fp)
