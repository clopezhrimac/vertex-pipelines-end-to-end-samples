import pandas as pd
import json
import pytest

from utilities.utils import read_datasets, split_xy, save_model_artifact, save_metrics
from utilities.utils import save_training_dataset_metadata
from unittest.mock import patch


def test_read_datasets_calls_read_csv_thrice(csv_data_paths):
    """
    Test if read_datasets calls pandas.read_csv three times with correct file paths.
    """
    train_path, valid_path, test_path = csv_data_paths

    with patch("pandas.read_csv") as mock_read_csv:
        _, _, _ = read_datasets(train_path, valid_path, test_path)
        assert mock_read_csv.call_count == 3


def test_read_datasets_returns_three_dataframes(csv_data_paths):
    """
    Assert read_datasets return three pandas dataframes
    """

    train_path, valid_path, test_path = csv_data_paths
    df_train, df_valid, df_test = read_datasets(train_path, valid_path, test_path)

    assert isinstance(df_train, pd.DataFrame)
    assert isinstance(df_valid, pd.DataFrame)
    assert isinstance(df_test, pd.DataFrame)


def test_read_datasets_returns_dataframes_with_correct_shape(csv_data_paths):
    """
    Assert read_datasets return dataframe with correct shapes
    """
    train_path, valid_path, test_path = csv_data_paths
    df_train, df_valid, df_test = read_datasets(train_path, valid_path, test_path)

    assert df_train.shape == (3, 3)
    assert df_valid.shape == (3, 3)
    assert df_test.shape == (3, 3)


def test_read_dataset_invalid_file_paths():
    """
    Test if the function raises FileNotFoundError for invalid file paths.
    """
    with pytest.raises(FileNotFoundError):
        read_datasets("path/to/invalid_train.csv", "path/to/invalid_valid.csv", "path/to/invalid_test.csv")


def test_split_xy_valid_label(dataframe_test):
    """
    Test shape and data types after split_xy with valid label
    """
    X, y = split_xy(dataframe_test, "label")

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    assert X.shape == (3, 2)
    assert y.shape == (3,)
    assert y.equals(pd.Series(["A", "B", "C"]))

    assert "label" not in X.columns


def test_split_xy_invalid_label(dataframe_test):
    """
    Test shape and data types after split_xy without valid label
    """
    with pytest.raises(KeyError):
        split_xy(dataframe_test, "non_existing_label")


def test_save_model_local(tmpdir, pipeline_mock):
    """
    Test if the function save_model_artifact correctly saves the trained model
    to a local file path.
    """
    model = pipeline_mock
    path = tmpdir.mkdir("local_model")

    save_model_artifact(model, str(path))

    assert path.exists()
    assert (path / "model.joblib").exists()


def test_save_metrics_to_json(tmpdir, metrics):
    """
    Test if the function save_metrics_to_json correctly saves the evaluation metrics
    to a JSON file.
    """
    metrics_file = tmpdir.join("metrics.json")
    metrics_path = str(metrics_file)

    save_metrics(metrics, metrics_path)

    assert metrics_file.exists()
    with open(metrics_path, "r") as fp:
        saved_metrics = json.load(fp)
    assert saved_metrics == metrics


def test_save_metrics_to_json_invalid_path(tmpdir, metrics):
    """
    Test handling invalid file path while saving evaluation metrics.
    """
    metrics_path = "non_existent_directory/metrics.json"

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        save_metrics(metrics, metrics_path)


def test_save_training_dataset_metadata_local(tmpdir):
    """
    Test checks if the function save_training_dataset_metadata correctly saves
    the training dataset information to a JSON file when local paths are provided.
    """
    model_path = tmpdir.mkdir("model")
    train_data_path = tmpdir.join("train_data.csv")
    label = "target"

    save_training_dataset_metadata(str(model_path), str(train_data_path), label)

    assert (model_path / "training_dataset_info.json").exists()
    with open(model_path / "training_dataset_info.json", "r") as fp:
        saved_metadata = json.load(fp)
    assert saved_metadata["gcsSource"]["uris"] == [str(train_data_path)]
    assert saved_metadata["dataFormat"] == "csv"
    assert saved_metadata["targetField"] == label
