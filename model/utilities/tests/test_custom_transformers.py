import pandas as pd
import numpy as np
import pytest
from utilities.custom_transformers import StringContainsTransformer


def test_string_contains_transformer_fit_returns_self(sample_transformer_data):
    """
      Verify that the fit method returns the same object.
    """
    target_string = 'a'
    transformer = StringContainsTransformer(target_string=target_string)
    assert transformer.fit(sample_transformer_data) is transformer
    assert transformer.column_names == ["text_column"]


@pytest.mark.parametrize("input_data, target_string, expected_output", [
    (pd.DataFrame({'text_column': ['apple', 'banana']}), 'an', pd.DataFrame({'text_column': [0.0, 1.0]})),
    (pd.DataFrame({'text_column': ['apple', 'banana']}), 'ch', pd.DataFrame({'text_column': [0.0, 0.0]})),
    (pd.DataFrame({'text_column': ['apple', 'banana']}), 'kiwi', pd.DataFrame({'text_column': [0.0, 0.0]})),
])
def test_string_contains_transformer_without_nans(input_data, target_string, expected_output):
    """
    Verify that the transform method produces the expected result without nans.
    """
    transformer = StringContainsTransformer(target_string)
    result = transformer.fit_transform(input_data)
    pd.testing.assert_frame_equal(result, expected_output)


@pytest.mark.parametrize("input_data, target_string, expected_output", [
    (pd.DataFrame({'text_column': ['banana', np.nan]}), 'e', pd.DataFrame({'text_column': [0.0, np.nan]})),
    (pd.DataFrame({'text_column': ['banana', np.nan]}), 'na', pd.DataFrame({'text_column': [1.0, np.nan]})),
    (pd.DataFrame({'text_column': [np.nan, 'kiwis']}), 'kiwi', pd.DataFrame({'text_column': [np.nan, 1.0]}))
])
def test_string_contains_transformer_with_nans(input_data, target_string, expected_output):
    """
    Verify that the transform method produces the expected result without nans.
    """
    transformer = StringContainsTransformer(target_string)
    result = transformer.fit_transform(input_data)
    pd.testing.assert_frame_equal(result, expected_output)


@pytest.mark.parametrize("target_string, expected_feature_names_out", [
    ('apple', ['text_column_1_apple', 'text_column_2_apple']),
    ('banana', ['text_column_1_banana', 'text_column_2_banana']),
    ('cherry', ['text_column_1_cherry', 'text_column_2_cherry']),
])
def test_get_feature_names_out(target_string, expected_feature_names_out):
    """
    Verify if the get_feature_names_out method of the StringContainsTransformer correctly transforms
    the feature names based on the provided target string and compares it to the expected result.
    """
    example_data = pd.DataFrame({
        'text_column_1': ['apple', 'banana', np.nan, 'date'],
        'text_column_2': ['apple', np.nan, 'cherry', 'date']
    })

    transformer = StringContainsTransformer(target_string)
    transformer.fit(example_data)
    feature_names_out = transformer.get_feature_names_out(example_data.columns.tolist())

    assert feature_names_out == expected_feature_names_out
