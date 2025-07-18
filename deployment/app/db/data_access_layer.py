import sqlite3
from typing import Any, List, Dict, Optional
from datetime import datetime, date

# Import all necessary functions from the database module
from deployment.app.db.database import (
    DatabaseError,
    create_data_upload_result,
    create_job,
    create_model_record,
    create_or_get_config,
    create_prediction_result,
    create_processing_run,
    create_training_result,
    create_tuning_result,
    delete_configs_by_ids,
    delete_model_record_and_file,
    delete_models_by_ids,
    adjust_dataset_boundaries,
    fetch_recent_retry_events,
    get_active_config,
    get_active_model,
    get_all_models,
    get_best_config_by_metric,
    get_best_model_by_metric,
    get_configs,
    get_data_upload_result,
    get_effective_config,
    get_job,
    get_latest_prediction_month,
    get_prediction_result,
    get_prediction_results_by_month,
    get_predictions_for_jobs,
    get_report_result,
    get_recent_models,
    get_top_configs,
    get_tuning_results,
    get_training_results,
    list_jobs,
    set_config_active,
    set_model_active,
    update_job_status,
    update_processing_run,
    insert_retry_event,
    get_or_create_multiindex_id,
    execute_query,
    auto_activate_best_config_if_enabled,
    auto_activate_best_model_if_enabled
)

# Import batch utility functions
from deployment.app.utils.batch_utils import (
    execute_query_with_batching,
    execute_many_with_batching,
)

# Define roles/permissions (example - these would be defined more formally elsewhere)
class UserRoles:
    ADMIN = "admin"
    USER = "user"
    SYSTEM = "system"

# A placeholder for the current user's context (e.g., roles)
# In a real FastAPI app, this would come from a dependency injection system
class UserContext:
    def __init__(self, roles: List[str]):
        self.roles = roles

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_any_role(self, roles: List[str]) -> bool:
        return any(role in self.roles for role in roles)


class DataAccessLayer:
    """Centralized data access layer to enforce security and business rules."""

    def __init__(self, user_context: UserContext = None):
        self.user_context = user_context or UserContext(roles=[UserRoles.SYSTEM])

    def _authorize(self, required_roles: List[str]):
        """Helper to check if the current user has the required roles."""
        if not self.user_context.has_any_role(required_roles):
            raise DatabaseError("Authorization failed: Insufficient permissions.")

    # Example: Wrapped function with authorization
    def create_job(self, job_type: str, parameters: dict[str, Any] = None, connection: sqlite3.Connection = None, status: str = "pending") -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_job(job_type, parameters, connection, status)

    def update_job_status(self, job_id: str, status: str, progress: float = None, result_id: str = None, error_message: str = None, status_message: str = None, connection: sqlite3.Connection = None) -> None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return update_job_status(job_id, status, progress, result_id, error_message, status_message, connection)

    def get_job(self, job_id: str, connection: sqlite3.Connection = None) -> dict:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_job(job_id, connection)

    def list_jobs(self, job_type: str = None, status: str = None, limit: int = 100, connection: sqlite3.Connection = None) -> List[Dict]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return list_jobs(job_type, status, limit, connection)

    def create_data_upload_result(self, job_id: str, records_processed: int, features_generated: List[str], processing_run_id: int, connection: sqlite3.Connection = None) -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_data_upload_result(job_id, records_processed, features_generated, processing_run_id, connection)

    def create_or_get_config(self, config_dict: dict[str, Any], is_active: bool = False, source: str | None = None, connection: sqlite3.Connection = None) -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_or_get_config(config_dict, is_active, source, connection)

    def get_active_config(self, connection: sqlite3.Connection = None) -> Dict[str, Any] | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_active_config(connection)

    def set_config_active(self, config_id: str, deactivate_others: bool = True, connection: sqlite3.Connection = None) -> bool:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return set_config_active(config_id, deactivate_others, connection)

    def get_best_config_by_metric(self, metric_name: str, higher_is_better: bool = True, connection: sqlite3.Connection = None) -> Dict[str, Any] | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_best_config_by_metric(metric_name, higher_is_better, connection)

    def create_model_record(self, model_id: str, job_id: str, model_path: str, created_at: datetime, metadata: dict[str, Any] | None = None, is_active: bool = False, connection: sqlite3.Connection = None) -> None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_model_record(model_id, job_id, model_path, created_at, metadata, is_active, connection)

    def get_active_model(self, connection: sqlite3.Connection = None) -> Dict[str, Any] | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_active_model(connection)

    def set_model_active(self, model_id: str, deactivate_others: bool = True, connection: sqlite3.Connection = None) -> bool:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return set_model_active(model_id, deactivate_others, connection)

    def get_best_model_by_metric(self, metric_name: str, higher_is_better: bool = True, connection: sqlite3.Connection = None) -> Dict[str, Any] | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_best_model_by_metric(metric_name, higher_is_better, connection)

    def get_recent_models(self, limit: int = 5, connection: sqlite3.Connection = None) -> List[Dict[str, Any]]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_recent_models(limit, connection)

    def delete_model_record_and_file(self, model_id: str, connection: sqlite3.Connection = None) -> bool:
        self._authorize([UserRoles.ADMIN, UserRoles.SYSTEM])
        return delete_model_record_and_file(model_id, connection)

    def create_training_result(self, job_id: str, model_id: str, config_id: str, metrics: dict[str, Any], config: dict[str, Any], duration: int | None, connection: sqlite3.Connection = None) -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_training_result(job_id, model_id, config_id, metrics, config, duration, connection)

    def create_prediction_result(self, job_id: str, model_id: str, output_path: str, summary_metrics: dict[str, Any] | None, prediction_month: date | None = None, connection: sqlite3.Connection = None) -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_prediction_result(job_id, model_id, output_path, summary_metrics, prediction_month, connection)

    def get_data_upload_result(self, result_id: str, connection: sqlite3.Connection = None) -> Dict:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_data_upload_result(result_id, connection)

    def get_training_results(self, result_id: str | None = None, limit: int = 100, connection: sqlite3.Connection = None) -> dict | list[dict]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_training_results(result_id, limit, connection)

    def get_prediction_result(self, result_id: str, connection: sqlite3.Connection = None) -> Dict:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_prediction_result(result_id, connection)

    def get_prediction_results_by_month(
        self,
        prediction_month: date,
        model_id: str | None = None,
        connection: sqlite3.Connection = None,
    ) -> list[dict]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_prediction_results_by_month(prediction_month, model_id, connection)

    def get_predictions_for_jobs(self, job_ids: list[str], model_id: str | None = None, connection: sqlite3.Connection = None) -> list[dict]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_predictions_for_jobs(job_ids, model_id, connection)

    def get_report_result(self, result_id: str, connection: sqlite3.Connection = None) -> Dict:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_report_result(result_id, connection)

    def create_processing_run(self, start_time: datetime, status: str, cutoff_date: str, source_files: str, end_time: datetime = None, connection: sqlite3.Connection = None) -> int:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_processing_run(start_time, status, cutoff_date, source_files, end_time, connection)

    def update_processing_run(self, run_id: int, status: str, end_time: datetime = None, connection: sqlite3.Connection = None) -> None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return update_processing_run(run_id, status, end_time, connection)

    def get_or_create_multiindex_id(self, barcode: str, artist: str, album: str, cover_type: str, price_category: str, release_type: str, recording_decade: str, release_decade: str, style: str, record_year: int, connection: sqlite3.Connection = None) -> int:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_or_create_multiindex_id(barcode, artist, album, cover_type, price_category, release_type, recording_decade, release_decade, style, record_year, connection)

    def get_configs(self, limit: int = 5, metric_name: str | None = None, higher_is_better: bool = True, include_active: bool = True, connection: sqlite3.Connection = None) -> List[Dict[str, Any]]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_configs(limit=limit, connection=connection)

    def delete_configs_by_ids(self, config_ids: List[str], connection: sqlite3.Connection = None) -> Dict[str, Any]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return delete_configs_by_ids(config_ids, connection)

    def get_all_models(self, limit: int = 100, include_active_status: bool = True, connection: sqlite3.Connection = None) -> List[Dict[str, Any]]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_all_models(limit, include_active_status, connection)

    def delete_models_by_ids(self, model_ids: List[str], connection: sqlite3.Connection = None) -> Dict[str, Any]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return delete_models_by_ids(model_ids, connection)

    def get_effective_config(self, settings, logger=None, connection=None):
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_effective_config(settings, logger, connection)

    def create_tuning_result(self, job_id: str, config_id: str, metrics: dict[str, Any] | None, duration: int | None, connection: sqlite3.Connection = None) -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_tuning_result(job_id, config_id, metrics, duration, connection)

    def get_tuning_results(self, result_id: str | None = None, metric_name: str | None = None, higher_is_better: bool | None = None, limit: int = 100, connection: sqlite3.Connection = None) -> dict | list[dict]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_tuning_results(result_id, metric_name, higher_is_better, limit, connection)

    def get_top_configs(self, limit: int = 5, metric_name: str | None = None, higher_is_better: bool = True, include_active: bool = True, connection: sqlite3.Connection = None) -> List[Dict[str, Any]]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_top_configs(limit, metric_name, higher_is_better, include_active, connection)

    def insert_retry_event(self, event: dict[str, Any], connection: sqlite3.Connection = None) -> None:
        self._authorize([UserRoles.ADMIN, UserRoles.SYSTEM])
        return insert_retry_event(event, connection)

    def fetch_recent_retry_events(self, limit: int = 1000, connection: sqlite3.Connection = None) -> List[Dict[str, Any]]:
        self._authorize([UserRoles.ADMIN, UserRoles.SYSTEM])
        return fetch_recent_retry_events(limit, connection)

    def execute_raw_query(self, query: str, params: tuple = (), fetchall: bool = False, connection: sqlite3.Connection = None) -> List[Dict] | Dict | None:
        self._authorize([UserRoles.ADMIN, UserRoles.SYSTEM])
        return execute_query(query, params, fetchall, connection)

    def auto_activate_best_config_if_enabled(self, connection: sqlite3.Connection = None) -> bool:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return auto_activate_best_config_if_enabled(connection)

    def auto_activate_best_model_if_enabled(self, connection: sqlite3.Connection = None) -> bool:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return auto_activate_best_model_if_enabled(connection)

    def adjust_dataset_boundaries(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        connection: sqlite3.Connection = None,
    ) -> date | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return adjust_dataset_boundaries(start_date, end_date, connection)
    
    def get_latest_prediction_month(self, connection: sqlite3.Connection = None) -> date:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_latest_prediction_month(connection)

    # Batch utility methods
    def execute_query_with_batching(
        self,
        query_template: str,
        ids: List[Any],
        batch_size: Optional[int] = None,
        connection: Optional[sqlite3.Connection] = None,
        fetchall: bool = True,
        placeholder_name: str = "placeholders"
    ) -> List[Any]:
        """Execute query with IN clause using batching to avoid SQLite variable limit."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return execute_query_with_batching(
            query_template, ids, batch_size, connection, fetchall, placeholder_name
        )

    def execute_many_with_batching(
        self,
        query: str,
        params_list: List[tuple],
        batch_size: Optional[int] = None,
        connection: Optional[sqlite3.Connection] = None
    ) -> None:
        """Execute multiple queries through batching to avoid SQLite variable limit."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return execute_many_with_batching(query, params_list, batch_size, connection)