from dataclasses import dataclass

from pipelines.algorithm_strategy_interface import AlgorithmStrategyInterface


@dataclass
class XGBoostStrategy(AlgorithmStrategyInterface):
    train_script_name = "train_xgb_model.py"
    table_suffix = "_xgb_training"
    default_model_name = "simple_xgboost"
    default_table_suffix = "_xgb_training"
    script_filename = "train_xgb_model.py"
    train_container_uri = (
        "europe-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest",
    )  # noqa: E501
    serve_container_uri = (
        "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
    )  # noqa: E501
    requirements = ["scikit-learn==0.24.0"]

    def hyperparameters(self, label_column_name) -> dict:
        return dict(
            n_estimators=200,
            early_stopping_rounds=10,
            objective="reg:squarederror",
            booster="gbtree",
            learning_rate=0.3,
            min_split_loss=0,
            max_depth=6,
            label=label_column_name,
        )
