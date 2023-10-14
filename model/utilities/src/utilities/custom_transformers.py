from sklearn.base import BaseEstimator, TransformerMixin


class StringContainsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_string: str):
        self.target_string = target_string
        self.column_names = None

    def fit(self, X, y=None):
        self.column_names = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.astype(object)
        strings_contains = X.apply(lambda x: x.str.contains(self.target_string))
        return strings_contains.astype(float)

    def get_feature_names_out(self, feature_names):
        features_names_out = [
            f"{name}_{self.target_string}" for name in self.column_names
        ]
        return features_names_out
