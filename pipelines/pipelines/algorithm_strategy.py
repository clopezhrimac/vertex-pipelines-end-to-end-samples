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
        "europe-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest"
    )
    serve_container_uri = (
        "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
    )
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


@dataclass
class TensorflowStrategy(AlgorithmStrategyInterface):
    train_script_name = "train_xgb_model.py"
    table_suffix = "_xgb_training"
    default_model_name = "simple_tensorflow"
    default_table_suffix = "_tf_training"
    script_filename = "train_tf_model.py"
    train_container_uri = "europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-6:latest"
    serve_container_uri = (
        "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest"
    )
    requirements = []

    def hyperparameters(self) -> dict:
        return dict(
            batch_size=100,
            epochs=5,
            loss_fn="MeanSquaredError",
            optimizer="Adam",
            learning_rate=0.01,
            hidden_units=[(64, "relu"), (32, "relu")],
            distribute_strategy="single",
            early_stopping_epochs=5,
        )
