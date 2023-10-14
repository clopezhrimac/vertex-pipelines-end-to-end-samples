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
import pathlib

from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output
from pipelines import generate_query
from pipelines.propension.config import TrainingConfig
from bigquery_components import extract_bq_to_dataset
from vertex_components import upload_model

config = TrainingConfig()


@dsl.container_component
def train(
    train_data: Input[Dataset],
    valid_data: Input[Dataset],
    test_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    feature_importance: Output[Metrics],
    hparams: dict,
    label: str,
):
    return dsl.ContainerSpec(
        image=config.train_container_uri,
        command=["python", "-m"],
        args=[
            "src.main",
            "--train-data",
            train_data.path,
            "--valid-data",
            valid_data.path,
            "--test-data",
            test_data.path,
            "--label",
            label,
            "--model",
            model.path,
            "--feature-importance",
            feature_importance.path,
            "--metrics",
            metrics.path,
            "--hparams",
            hparams,
        ],
    )


@dsl.pipeline(name=config.pipeline_name)
def pipeline(
    project_id: str = config.project_id,
    project_location: str = config.project_location,
    model_name: str = config.model_name,
    dataset_id: str = config.dataset_id,
    dataset_location: str = config.dataset_location,
    period: str = config.period,
):
    """
    XGB training pipeline which:
     1. Splits and extracts a dataset from BQ to GCS
     2. Trains a model via Vertex AI CustomContainerTrainingJob
     3. Evaluates the model against the current champion model
     4. If better the model becomes the new default model

    Args:
        project_id (str): project id of the Google Cloud project
        project_location (str): location of the Google Cloud project
        model_name (str): name of model
        dataset_id (str): id of BQ dataset used to store all staging data & predictions
        dataset_location (str): location of dataset
        period (str): Reference period to build training dataset
    """

    # generate sql queries which are used in ingestion and preprocessing
    # operations
    queries_folder = pathlib.Path(__file__).parent / "queries"

    target_engineering_query = generate_query(
        queries_folder / "target_engineering.sql",
        source_dataset=config.ingestion_dataset,
        source_table=config.ingestion_table,
        filter_column=config.time_column,
        target_column=config.target_column,
        filter_start_value=period,
        target_engineering_dataset=config.common_dataset,
        target_engineering_table=config.target_engineering_table,
    )

    enrich_query = generate_query(
        queries_folder / "enrich.sql",
        enriched_dataset=config.common_dataset,
        enriched_table=config.enriched_table,
        target_column=config.target_column,
        target_engineering_dataset=config.common_dataset,
        target_engineering_table=config.target_engineering_table,
        feature_store_dataset=config.feature_store_dataset,
    )

    split_data_query = generate_query(
        queries_folder / "split_data.sql",
        enriched_dataset=config.common_dataset,
        enriched_table=config.enriched_table,
        filter_column=config.time_column,
        split_data_dataset=config.common_dataset,
        train_table=config.train_table,
        train_start_date=config.train_start_date,
        train_final_date=config.train_final_date,
        validation_table=config.valid_table,
        validation_start_date=config.validation_start_date,
        validation_final_date=config.validation_final_date,
        test_table=config.test_table,
        test_start_date=config.test_start_date,
        test_final_date=config.test_final_date,
    )

    # data ingestion and preprocessing operations
    target_engineering = BigqueryQueryJobOp(
        project=project_id, location=dataset_location, query=target_engineering_query
    ).set_display_name("Target Engineering")

    enrich = (
        BigqueryQueryJobOp(
            project=project_id, location=dataset_location, query=enrich_query
        )
        .after(target_engineering)
        .set_display_name("Enrich")
    )

    split_data = (
        BigqueryQueryJobOp(
            project=project_id, location=dataset_location, query=split_data_query
        )
        .after(enrich)
        .set_display_name("Split data")
    )

    # data extraction to gcs
    train_dataset = (
        extract_bq_to_dataset(
            bq_client_project_id=project_id,
            source_project_id=project_id,
            dataset_id=dataset_id,
            table_name=config.train_table,
            dataset_location=dataset_location,
        )
        .after(split_data)
        .set_display_name("Extract train data to storage")
        .set_caching_options(False)
    ).outputs["dataset"]

    valid_dataset = (
        extract_bq_to_dataset(
            bq_client_project_id=project_id,
            source_project_id=project_id,
            dataset_id=dataset_id,
            table_name=config.valid_table,
            dataset_location=dataset_location,
        )
        .after(split_data)
        .set_display_name("Extract validation data to storage")
        .set_caching_options(False)
    ).outputs["dataset"]

    test_dataset = (
        extract_bq_to_dataset(
            bq_client_project_id=project_id,
            source_project_id=project_id,
            dataset_id=dataset_id,
            table_name=config.test_table,
            dataset_location=dataset_location,
        )
        .after(split_data)
        .set_display_name("Extract test data to storage")
        .set_caching_options(False)
    ).outputs["dataset"]

    train_model = train(
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        hparams=config.hparams,
        label=config.target_column,
    ).set_display_name("Train model")

    _ = upload_model(
        project_id=project_id,
        project_location=project_location,
        model=train_model.outputs["model"],
        model_evaluation=train_model.outputs["metrics"],
        test_dataset=test_dataset,
        eval_metric=config.primary_metric,
        eval_lower_is_better=True,
        serving_container_image=config.serving_container_uri,
        model_name=model_name,
        pipeline_job_id="{{$.pipeline_job_name}}",
    ).set_display_name("Upload model")
