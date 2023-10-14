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

import time
import pathlib

from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp

from kfp import dsl

from pipelines import generate_query
from pipelines.propension.config import PredictionConfig
from vertex_components import lookup_model

config = PredictionConfig()


@dsl.pipeline(name=config.pipeline_name)
def pipeline(
    project_id: str = config.project_id,
    project_location: str = config.project_location,
    model_name: str = config.model_name,
    dataset_id: str = config.dataset_id,
    dataset_location: str = config.dataset_location,
    period: str = config.period,
    batch_prediction_machine_type: str = config.machine_type,
    batch_prediction_min_replicas: int = config.min_replicas,
    batch_prediction_max_replicas: int = config.max_replicas,
    batch_prediction_batch_size: int = config.batch_size,
):
    """
    XGB prediction pipeline which:
     1. Looks up the default model version (champion) and
        dataset which was used to the train model.
     2. Runs a BatchPredictionJob with optional training-serving skew detection.

    Args:
        project_id (str): project id of the Google Cloud project
        project_location (str): location of the Google Cloud project
        model_name (str): name of model
        dataset_id (str): id of BQ dataset used to store all staging data & predictions
        dataset_location (str): location of dataset
        period (str): Reference period to build predict dataset
        batch_prediction_machine_type (str): Machine type to be used for Vertex Batch
            Prediction. Example machine_types - n1-standard-4, n1-standard-16 etc.
        batch_prediction_min_replicas (int): Minimum no of machines to distribute the
            Vertex Batch Prediction job for horizontal scalability
        batch_prediction_max_replicas (int): Maximum no of machines to distribute the
            Vertex Batch Prediction job for horizontal scalability.
        batch_prediction_batch_size (int): The number of the records (e.g. instances) of
            the operation given in each batch to a machine replica.

    Returns:
        None

    """

    # generate sql queries which are used in ingestion and preprocessing
    # operations
    queries_folder = pathlib.Path(__file__).parent / "queries"

    population_query = generate_query(
        queries_folder / "population.sql",
        population_dataset=config.common_dataset,
        population_table=config.population_table,
        feature_store_dataset=config.feature_store_dataset,
        source_table=config.persona_table,
        prediction_period=period,
    )

    enrich_query = generate_query(
        queries_folder / "enrich.sql",
        enriched_dataset=config.common_dataset,
        enriched_table=config.enriched_table,
        population_dataset=config.common_dataset,
        population_table=config.population_table,
        feature_store_dataset=config.feature_store_dataset,
    )

    population = (
        BigqueryQueryJobOp(
            project=project_id,
            location=dataset_location,
            query=population_query
        )
        .set_display_name("Get population data")
    )

    enrich = (
        BigqueryQueryJobOp(
            project=project_id,
            location=dataset_location,
            query=enrich_query
        )
        .after(population)
        .set_display_name("Enrich")
    )

    # lookup champion model
    champion_model = (
        lookup_model(
            model_name=model_name,
            project_location=project_location,
            project_id=project_id,
            fail_on_model_not_found=True,
        )
        .set_display_name("Look up champion model")
        .set_caching_options(False)
    )

    # batch predict from BigQuery to BigQuery
    bigquery_source_input_uri = f"bq://{project_id}.{dataset_id}.{config.enriched_table}"
    bigquery_destination_output_uri = f"bq://{project_id}.{dataset_id}"

    batch_prediction = (
        ModelBatchPredictOp(
            project=project_id,
            model=champion_model.outputs["model"],
            job_display_name=f"{config.display_job_name}-{int(time.time())}",
            location=project_location,
            instances_format="bigquery",
            predictions_format="bigquery",
            bigquery_source_input_uri=bigquery_source_input_uri,
            bigquery_destination_output_uri=bigquery_destination_output_uri,
            instance_type="object",
            machine_type=batch_prediction_machine_type,
            starting_replica_count=batch_prediction_min_replicas,
            max_replica_count=batch_prediction_max_replicas,
            manual_batch_tuning_parameters_batch_size=batch_prediction_batch_size
        )
        .set_caching_options(False)
        .after(enrich)
        .set_display_name("Batch prediction job")
    )
