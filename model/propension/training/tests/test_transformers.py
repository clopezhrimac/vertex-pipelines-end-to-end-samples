from transformers import build_numeric_transformers
from transformers import build_categorical_transformers
from transformers import build_preprocessing_transformer

from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


def test_build_numeric_transformers_types(column_indices):
    """
    Assert datatype of numeric transformer
    """
    transformers = build_numeric_transformers(column_indices)

    # Assert that the transformers list is not empty
    assert len(transformers) > 0

    # Assert that each transformer tuple has the correct format
    for transformer in transformers:
        assert isinstance(transformer, tuple)
        assert len(transformer) == 3
        assert isinstance(transformer[0], str)
        assert isinstance(transformer[1], TransformerMixin)
        assert isinstance(transformer[2], list)


def test_build_categorical_transformers_types(column_indices, categories_order):
    """
    Assert datatype of categorical transformer
    """
    transformers = build_categorical_transformers(column_indices, categories_order)

    # Assert that the transformers list is not empty
    assert len(transformers) > 0

    # Assert that each transformer tuple has the correct format
    for transformer in transformers:
        assert isinstance(transformer, tuple)
        assert len(transformer) == 3
        assert isinstance(transformer[0], str)
        assert isinstance(transformer[1], (Pipeline, TransformerMixin))
        assert isinstance(transformer[2], list)


def test_build_preprocessing_transformer_types(column_indices, categories_order):
    """
    Assert datatype of preprocessing transformer
    """
    transformer = build_preprocessing_transformer(column_indices, categories_order)

    # Assert that the transformer is an instance of ColumnTransformer
    assert isinstance(transformer, ColumnTransformer)
