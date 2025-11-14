"""Query modules for database operations."""

# Export all query functions for convenient imports
from deployment.app.db.queries.configs import (
    auto_activate_best_config_if_enabled,
    create_or_get_config,
    delete_configs_by_ids,
    get_active_config,
    get_best_config_by_metric,
    get_configs,
    get_effective_config,
    get_top_configs,
    set_config_active,
)
from deployment.app.db.queries.features import (
    adjust_dataset_boundaries,
    delete_features_by_table,
    get_feature_dataframe,
    get_features_by_date_range,
    get_report_features,
    insert_features_batch,
    insert_report_features,
)
from deployment.app.db.queries.jobs import (
    create_job,
    get_job,
    get_job_params,
    get_job_prediction_month,
    list_jobs,
    try_acquire_job_submission_lock,
    update_job_status,
)
from deployment.app.db.queries.models import (
    auto_activate_best_model_if_enabled,
    create_model_record,
    delete_model_record_and_file,
    delete_models_by_ids,
    get_active_model,
    get_active_model_primary_metric,
    get_all_models,
    get_best_model_by_metric,
    get_recent_models,
    set_model_active,
)
from deployment.app.db.queries.multiindex import (
    get_multiindex_mapping_batch,
    get_multiindex_mapping_by_ids,
    get_or_create_multiindex_ids_batch,
)
from deployment.app.db.queries.predictions import (
    get_latest_prediction_month,
    get_next_prediction_month,
)
from deployment.app.db.queries.processing_runs import (
    create_processing_run,
    update_processing_run,
)
from deployment.app.db.queries.results import (
    create_data_upload_result,
    create_prediction_result,
    create_training_result,
    create_tuning_result,
    get_data_upload_result,
    get_prediction_result,
    get_prediction_results_by_month,
    get_predictions,
    get_report_result,
    get_training_results,
    get_tuning_results,
    insert_predictions,
)
from deployment.app.db.queries.retry_events import (
    fetch_recent_retry_events,
    insert_retry_event,
)

__all__ = [
    # Configs
    "create_or_get_config",
    "get_active_config",
    "set_config_active",
    "get_best_config_by_metric",
    "get_configs",
    "get_top_configs",
    "delete_configs_by_ids",
    "auto_activate_best_config_if_enabled",
    "get_effective_config",
    # Features
    "get_feature_dataframe",
    "get_features_by_date_range",
    "insert_features_batch",
    "delete_features_by_table",
    "get_report_features",
    "insert_report_features",
    "adjust_dataset_boundaries",
    # Jobs
    "create_job",
    "update_job_status",
    "get_job",
    "list_jobs",
    "try_acquire_job_submission_lock",
    "get_job_params",
    "get_job_prediction_month",
    # Models
    "create_model_record",
    "get_active_model",
    "get_active_model_primary_metric",
    "set_model_active",
    "get_best_model_by_metric",
    "get_recent_models",
    "delete_model_record_and_file",
    "get_all_models",
    "delete_models_by_ids",
    "auto_activate_best_model_if_enabled",
    # Multiindex
    "get_or_create_multiindex_ids_batch",
    "get_multiindex_mapping_by_ids",
    "get_multiindex_mapping_batch",
    # Predictions
    "get_next_prediction_month",
    "get_latest_prediction_month",
    # Processing runs
    "create_processing_run",
    "update_processing_run",
    # Results
    "create_training_result",
    "create_tuning_result",
    "create_prediction_result",
    "get_training_results",
    "get_tuning_results",
    "get_prediction_result",
    "get_prediction_results_by_month",
    "get_predictions",
    "insert_predictions",
    "get_data_upload_result",
    "create_data_upload_result",
    "get_report_result",
    # Retry events
    "insert_retry_event",
    "fetch_recent_retry_events",
]
