from sklearn.base import BaseEstimator, TransformerMixin


class StringContainsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_string):
        self.target_string = target_string
        self.column_names = None

    def fit(self, X, y=None):
        self.column_names = X.columns.tolist()
        return self

    def transform(self, X):
        flag_df = X.apply(lambda x: x.str.contains(self.target_string))
        return flag_df.astype(float)

    def get_feature_names_out(self, feature_names):
        features_names_out = [f"{name}_{self.target_string}" for name in self.column_names]
        return features_names_out
