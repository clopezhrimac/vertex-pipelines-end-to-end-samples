# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pathlib

from kfp.v2 import compiler, dsl
from google_cloud_pipeline_components.experimental.custom_job.utils import (
    create_custom_training_job_op_from_component,
)
from pipelines import generate_query
from pipelines.components import (
    lookup_model,
    export_model,
    upload_model,
    extract_bq_to_dataset,
    bq_query_to_table,
    train_xgboost_model,
    predict_xgboost_model,
    calculate_eval_metrics,
    compare_models,
)


SKL_SERVING_CONTAINER_IMAGE_URI = (
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
)


@dsl.pipeline(name="xgboost-train-pipeline")
def xgboost_pipeline(
    project_id: str = os.environ.get("VERTEX_PROJECT_ID"),
    project_location: str = os.environ.get("VERTEX_LOCATION"),
    ingestion_project_id: str = os.environ.get("VERTEX_PROJECT_ID"),
    model_name: str = "xgboost_with_preprocessing",
    dataset_id: str = "preprocessing",
    dataset_location: str = os.environ.get("VERTEX_LOCATION"),
    ingestion_dataset_id: str = "chicago_taxi_trips",
    timestamp: str = "2022-12-01 00:00:00",
):
    """
    XGB training pipeline which:
     1. Extracts a dataset from BQ
     2. Generates statistics
     3. Performs the validation of it against a tfdv schema
     4. Alert if there are any anomalies
     5. Trains the model via Vertex AI CustomTrainJob
     6. Evaluates the model against the current champion model
     7. If better than the current champion model it pushes the model to
     Vertex AI Models

    Args:
        project_id (str): project id of the Google Cloud project
        project_location (str): location of the Google Cloud project
        ingestion_project_id (str): project id containing the source bigquery data
            for ingestion. This can be the same as `project_id` if the source data is
            in the same project where the ML pipeline is executed.
        model_name (str): name of model
        model_label (str): label of model
        dataset_id (str): id of BQ dataset used to store all staging data & predictions
        dataset_location (str): location of dataset
        ingestion_dataset_id (str): dataset id of ingestion data
        timestamp (str): Optional. Empty or a specific timestamp in ISO 8601 format
            (YYYY-MM-DDThh:mm:ss.sss±hh:mm or YYYY-MM-DDThh:mm:ss).
            If any time part is missing, it will be regarded as zero.


    Returns:
        None

    """

    # Create variables to ensure the same arguments are passed
    # into different components of the pipeline
    file_pattern = ""  # e.g. "files-*.csv", used as file pattern on storage
    label_column_name = "total_fare"
    pred_column_name = "predictions"
    time_column = "trip_start_timestamp"
    metrics_names = ["MeanSquaredError"]
    ingestion_table = "taxi_trips"
    table_suffix = "_xgb_training"  # suffix to table names
    ingested_table = "ingested_data" + table_suffix
    preprocessed_table = "preprocessed_data" + table_suffix
    train_table = "train_data" + table_suffix
    valid_table = "valid_data" + table_suffix
    test_table = "test_data" + table_suffix

    # generate sql queries which are used in ingestion and preprocessing
    # operations

    queries_folder = pathlib.Path(__file__).parent / "queries"

    ingest_query = generate_query(
        queries_folder / "ingest.sql",
        source_dataset=f"{ingestion_project_id}.{ingestion_dataset_id}",
        source_table=ingestion_table,
        filter_column=time_column,
        target_column=label_column_name,
        filter_start_value=timestamp,
    )
    split_train_query = generate_query(
        queries_folder / "sample.sql",
        source_dataset=dataset_id,
        source_table=ingested_table,
        num_lots=10,
        lots=tuple(range(8)),
    )
    split_valid_query = generate_query(
        queries_folder / "sample.sql",
        source_dataset=dataset_id,
        source_table=ingested_table,
        num_lots=10,
        lots="(8)",
    )
    split_test_query = generate_query(
        queries_folder / "sample.sql",
        source_dataset=dataset_id,
        source_table=ingested_table,
        num_lots=10,
        lots="(9)",
    )
    data_cleaning_query = generate_query(
        queries_folder / "engineer_features.sql",
        source_dataset=dataset_id,
        source_table=train_table,
    )

    # data ingestion and preprocessing operations

    kwargs = dict(
        bq_client_project_id=project_id,
        destination_project_id=project_id,
        dataset_id=dataset_id,
        dataset_location=dataset_location,
        query_job_config=json.dumps(dict(write_disposition="WRITE_TRUNCATE")),
    )
    ingest = bq_query_to_table(
        query=ingest_query, table_id=ingested_table, **kwargs
    ).set_display_name("Ingest data")

    # exporting data to GCS from BQ

    split_train_data = (
        bq_query_to_table(query=split_train_query, table_id=train_table, **kwargs)
        .after(ingest)
        .set_display_name("Split train data")
    )
    split_valid_data = (
        bq_query_to_table(query=split_valid_query, table_id=valid_table, **kwargs)
        .after(ingest)
        .set_display_name("Split validation data")
    )
    split_test_data = (
        bq_query_to_table(query=split_test_query, table_id=test_table, **kwargs)
        .after(ingest)
        .set_display_name("Split test data")
    )
    data_cleaning = (
        bq_query_to_table(
            query=data_cleaning_query, table_id=preprocessed_table, **kwargs
        )
        .after(split_train_data)
        .set_display_name("Data Cleansing")
    )

    # data extraction to gcs

    train_dataset = (
        extract_bq_to_dataset(
            bq_client_project_id=project_id,
            source_project_id=project_id,
            dataset_id=dataset_id,
            table_name=preprocessed_table,
            dataset_location=dataset_location,
            file_pattern=file_pattern,
        )
        .after(data_cleaning)
        .set_display_name("Extract train data to storage")
    )
    valid_dataset = (
        extract_bq_to_dataset(
            bq_client_project_id=project_id,
            source_project_id=project_id,
            dataset_id=dataset_id,
            table_name=valid_table,
            dataset_location=dataset_location,
            file_pattern=file_pattern,
        )
        .after(split_valid_data)
        .set_display_name("Extract validation data to storage")
    )
    test_dataset = (
        extract_bq_to_dataset(
            bq_client_project_id=project_id,
            source_project_id=project_id,
            dataset_id=dataset_id,
            table_name=test_table,
            dataset_location=dataset_location,
            file_pattern=file_pattern,
        )
        .after(split_test_data)
        .set_display_name("Extract test data to storage")
    )

    # train xgboost model
    """
    scikit-learn version for training job requirement, local predict component,
    train image & serving image need to be in sync
    Training job req - v0.24.1
    Local predict - v0.24.1
    Training image - v0.23.1 (latest available)
    Serving image - v0.24.1 (latest available)
    """
    model_params = dict(
        n_estimators=200,
        early_stopping_rounds=10,
        objective="reg:squarederror",
        booster="gbtree",
        learning_rate=0.3,
        min_split_loss=0,
        max_depth=6,
    )

    train_model = (
        custom_train_job(
            training_data=train_dataset.outputs["dataset"],
            validation_data=valid_dataset.outputs["dataset"],
            file_pattern=file_pattern,
            label_name=label_column_name,
            model_params=json.dumps(model_params),
            # Training wrapper specific parameters
            project=project_id,
            location=project_location,
        )
        .after(train_dataset)
        .set_display_name("Vertex Training for XGB model")
    )

    model = train_model.outputs["model"]
    metrics_artifact = train_model.outputs["metrics_artifact"]

    # predict test dataset using trained model

    challenger_predictions = predict_xgboost_model(
        test_dataset.outputs["dataset"],
        model,
        label_column_name=label_column_name,
        predictions_column_name=pred_column_name,
        file_pattern=file_pattern,
    ).set_display_name("Predict test data")

    # Calculate evaluation metrics of challenger model
    challenger_eval_metrics = calculate_eval_metrics(
        csv_file=challenger_predictions.output,
        metrics_names=json.dumps(metrics_names),
        label_column_name=label_column_name,
        pred_column_name=pred_column_name,
    ).set_display_name("Evaluate test metrics for challenger model")

    champion_model_lookup = lookup_model(
        model_name=model_name,
        project_location=project_location,
        project_id=project_id,
        fail_on_model_not_found=False,
    ).set_display_name("Lookup champion model")

    champion_model_resource_name = champion_model_lookup.outputs["model_resource_name"]

    # If there is no champion model, upload challenger model
    with dsl.Condition(
        name="champion-model-not-exists",
        condition=(champion_model_resource_name == ""),
    ):

        # Upload model
        upload_model(
            display_name=model_name,
            serving_container_image_uri=SKL_SERVING_CONTAINER_IMAGE_URI,
            model=model,
            project_id=project_id,
            project_location=project_location,
            description="",
            labels=json.dumps(
                dict(
                    pipeline_job_uuid="{{$.pipeline_job_uuid}}",
                    pipeline_job_name="{{$.pipeline_job_name}}",
                )
            ),
        ).set_display_name("Upload challenger model")

    with dsl.Condition(
        name="champion-model-exists",
        condition=(champion_model_resource_name != ""),
    ):

        exported_champion_model = export_model(
            model_resource_name=champion_model_resource_name,
        ).set_display_name("Export champion model")

        champion_model = exported_champion_model.outputs["model"]

        champion_predictions = predict_xgboost_model(
            test_dataset.outputs["dataset"],
            champion_model,
            label_column_name=label_column_name,
            predictions_column_name=pred_column_name,
            file_pattern=file_pattern,
        ).set_display_name("Predict test data")

        # Calculate evaluation metrics of champion model
        champion_eval_metrics = calculate_eval_metrics(
            csv_file=champion_predictions.output,
            metrics_names=json.dumps(metrics_names),
            label_column_name=label_column_name,
            pred_column_name=pred_column_name,
        ).set_display_name("Evaluate test metrics for the champion model")

        # Determine if challenger model is better than champion model
        compare_champion_challenger_models = compare_models(
            metrics=champion_eval_metrics.outputs["eval_metrics"],
            other_metrics=challenger_eval_metrics.outputs["eval_metrics"],
            evaluation_metric="mean_squared_error",
            higher_is_better=False,
            absolute_difference=0.0,
        ).set_display_name("Compare champion and challenger models")

        # Upload challenger model if it is better than champion model
        with dsl.Condition(
            name="challenger-better-than-champion",
            condition=(compare_champion_challenger_models.output == "true"),
        ):

            # Upload model
            upload_model(
                display_name=model_name,
                serving_container_image_uri=SKL_SERVING_CONTAINER_IMAGE_URI,
                model=model,
                project_id=project_id,
                project_location=project_location,
                description="",
                labels=json.dumps(
                    dict(
                        pipeline_job_uuid="{{$.pipeline_job_uuid}}",
                        pipeline_job_name="{{$.pipeline_job_name}}",
                    )
                ),
            ).set_display_name("Upload challenger model")


def compile():
    """
    Uses the kfp compiler package to compile the pipeline function into a workflow yaml

    Args:
        None

    Returns:
        None
    """
    compiler.Compiler().compile(
        pipeline_func=xgboost_pipeline,
        package_path="training.json",
        type_check=False,
    )


if __name__ == "__main__":
    custom_train_job = create_custom_training_job_op_from_component(
        component_spec=train_xgboost_model,
        replica_count=1,
        machine_type="n1-standard-4",
    )
    compile()
