import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from typing import List, Tuple, Union, Dict
from utilities.custom_transformers import StringContainsTransformer


def build_numeric_transformers(
    column_indices: Dict[str, List[int]]
) -> List[Tuple[str, Union[Pipeline, TransformerMixin], List[int]]]:
    """Build the numeric transformers for preprocessing.

    Args:
        column_indices (dict): A dictionary containing the column indices.

    Returns:
        List[Tuple[str, TransformerMixin, Union[int, List[int]]]]: List of numeric transformers.
    """
    numeric_transformers = [
        (
            "impute_numerical",
            SimpleImputer(strategy="constant", fill_value=0),
            column_indices["numerical"],
        ),
        ("impute_age", SimpleImputer(strategy="median"), column_indices["age"]),
    ]
    return numeric_transformers


def build_categorical_transformers(
    column_indices: Dict[str, List[int]], categories_order: Dict[str, List[List[str]]]
) -> List[Tuple[str, Union[Pipeline, TransformerMixin], List[int]]]:
    """Build the categorical transformers for preprocessing.

    Args:
        column_indices (dict): A dictionary containing the column indices.
        categories_order (dict): A dictionary containing the category orders.

    Returns:
        List[Tuple[str, Union[Pipeline, TransformerMixin], List[int]]]: List of categorical transformers.
    """
    nse_pipe = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="constant", fill_value="OTRO")),
            (
                "ordinal_encoding",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    categories=categories_order["nse"],
                ),
            ),
        ]
    )

    rcc_pipe = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="constant", fill_value="OTRO")),
            (
                "ordinal_encoding",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    categories=categories_order["rcc"],
                ),
            ),
        ]
    )

    des_cono_agrup_pipe = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="constant", fill_value="OTRO")),
            (
                "ordinal_encoding",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    des_lima_prov_pipe = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="constant", fill_value="OTRO")),
            (
                "one_hot_encoding",
                OneHotEncoder(
                    handle_unknown="error",
                    drop="first",
                    categories=categories_order["lima_prov"],
                ),
            ),
        ]
    )

    categorical_transformers = [
        ("nse_encoding", nse_pipe, column_indices["nse"]),
        ("rcc_encoding", rcc_pipe, column_indices["rcc"]),
        ("cono_agrup_encoding", des_cono_agrup_pipe, column_indices["cono_agrup"]),
        ("lima_prov_encoding", des_lima_prov_pipe, column_indices["lima_prov"]),
        ("flag_desgravamen", StringContainsTransformer(target_string='DESGRA'), column_indices["products"])
    ]
    return categorical_transformers


def build_preprocessing_transformer(
    column_indices: Dict[str, List[int]], categories_order: Dict[str, List[List[str]]]
) -> ColumnTransformer:
    """Build a preprocessing column transformer for feature engineering.

    Args:
        column_indices (dict): A dictionary containing the column indices.
        categories_order (dict): A dictionary containing the category orders.

    Returns:
        ColumnTransformer: A ColumnTransformer object representing the preprocessing pipeline.
    """
    numeric_transformers = build_numeric_transformers(column_indices)
    categorical_transformers = build_categorical_transformers(column_indices, categories_order)

    all_transformers = numeric_transformers + categorical_transformers
    preprocesser = ColumnTransformer(transformers=all_transformers)

    return preprocesser


def create_model(hyperparameters: dict) -> lgb.LGBMClassifier:
    """Create a LightGBM classifier model with the specified hyperparameters.

    Args:
        hyperparameters (dict): A dictionary containing hyperparameters for the LightGBM model.
            The hyperparameters should be compatible with LGBMClassifier constructor arguments.

    Returns:
        LGBMClassifier: A LightGBM classifier model configured with the provided hyperparameters.
    """
    lgb_model = lgb.LGBMClassifier(**hyperparameters)
    return lgb_model
