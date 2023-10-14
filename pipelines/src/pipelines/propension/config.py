from os import environ as env

from dataclasses import dataclass


@dataclass(init=False, repr=True, frozen=True)
class Config:
    """Configuration for all pipelines."""

    project_id = env.get("VERTEX_PROJECT_ID")
    project_id_ingestion = env.get("VERTEX_PROJECT_ID")
    project_location = env.get("VERTEX_LOCATION")
    dataset_location = "US"
    dataset_id = "tmp"
    staging_bucket = env.get("VERTEX_PIPELINE_ROOT")
    pipeline_files_gcs_path = env.get("PIPELINE_FILES_GCS_PATH")
    resource_suffix = env.get("RESOURCE_SUFFIX", "")
    solution_name = env.get("SOLUTION_NAME")
    model_name = "per_propension_vn_salud_cls"
    model_type = "propension"
    display_job_name = f"{solution_name}-{model_type}"
    period = "2023-05-01"
    table_suffix = "_sv_salud_prop"
    fs_project_id = "rs-prd-dlk-sbx-fsia-8104"
    fs_dataset_id = "prod_featurestore"
    common_dataset = f"{project_id}.{dataset_id}"
    feature_store_dataset = f"{fs_project_id}.{fs_dataset_id}"


@dataclass(init=False, repr=True, frozen=True)
class TrainingConfig(Config):
    """Configuration for training pipeline."""

    pipeline_name = Config.display_job_name + "-train"
    mdm_project_id = "rs-shr-al-analyticsz-prj-ebc1"
    anl_digital_dataset_id = "anl_digital"
    ingestion_dataset = f"{mdm_project_id}.{anl_digital_dataset_id}"
    ingestion_table = "leads"

    target_column = "per_propension_vn_salud_cls"
    time_column = "periodo"

    target_engineering_table = "target_engineering" + Config.table_suffix
    enriched_table = "enriched_train" + Config.table_suffix
    train_table = "train_data" + Config.table_suffix
    valid_table = "valid_data" + Config.table_suffix
    test_table = "test_data" + Config.table_suffix

    train_start_date = "2022-05-01"
    train_final_date = "2023-02-01"
    validation_start_date = "2023-03-01"
    validation_final_date = "2023-03-01"
    test_start_date = "2023-04-01"
    test_final_date = "2023-04-01"

    primary_metric = "auRoc"
    hparams = dict(
        n_estimators=1500, objective="binary", learning_rate=0.01, max_depth=3
    )

    train_container_uri = (
        f"{env.get('CONTAINER_URI_PREFIX')}/custom-training-images/{Config.solution_name}-"
        f"{Config.model_type}:latest"
    )
    serving_container_uri = (
        f"{env.get('CONTAINER_URI_PREFIX')}/custom-serving-images/{Config.solution_name}-"
        f"{Config.model_type}:latest"
    )
    serving_container_predict_route = "/predict"
    serving_container_health_route = "/health"


@dataclass(init=False, repr=True, frozen=True)
class PredictionConfig(Config):
    """Configuration for prediction pipeline."""

    pipeline_name = Config.display_job_name + "-predict"
    persona_table = "persona__static"
    population_table = "population" + Config.table_suffix
    enriched_table = "enriched_predict" + Config.table_suffix
    machine_type = "n1-standard-4"
    min_replicas = 30
    max_replicas = 35
    batch_size = 800
    monitoring_alert_email_addresses = ["carlos.lopez@rimac.com.pe"]
    monitoring_skew_config = {"defaultSkewThreshold": {"value": 0.001}}
    instance_config = {}
