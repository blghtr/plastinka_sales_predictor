import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any
from functools import wraps

import pandas as pd

# Import all necessary functions from the database module
from deployment.app.db.database import (
    DatabaseError,
    adjust_dataset_boundaries,
    auto_activate_best_config_if_enabled,
    auto_activate_best_model_if_enabled,
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
    dict_factory,
    execute_many_with_batching,
    execute_query,
    execute_query_with_batching,
    get_active_config,
    get_active_model,
    get_active_model_primary_metric,
    get_all_models,
    get_best_config_by_metric,
    get_best_model_by_metric,
    get_configs,
    get_data_upload_result,
    get_db_connection,
    get_effective_config,
    get_feature_dataframe,
    get_features_by_date_range,
    get_job,
    get_job_params,
    get_job_prediction_month,
    get_latest_prediction_month,
    get_next_prediction_month,
    get_prediction_result,
    get_prediction_results_by_month,
    get_predictions,
    get_recent_models,
    get_report_features,
    get_report_result,
    get_top_configs,
    get_training_results,
    get_tuning_results,
    insert_predictions,
    list_jobs,
    try_acquire_job_submission_lock,
    set_config_active,
    set_model_active,
    update_job_status,
    update_processing_run,
    delete_features_by_table,
    insert_features_batch,
)
from deployment.app.db.schema import init_db

# Define roles/permissions
class UserRoles:
    ADMIN = "admin"
    USER = "user"
    SYSTEM = "system"

# A placeholder for the current user's context (e.g., roles)
# In a real FastAPI app, this would come from a dependency injection system
class UserContext:
    def __init__(self, roles: list[str]):
        self.roles = roles

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_any_role(self, roles: list[str]) -> bool:
        return any(role in self.roles for role in roles)


def transaction_required(func):
    """
    Decorator that automatically manages transactions for data modification methods.
    If no transaction is active, it creates one. If a transaction is already active,
    it just executes the method within the existing transaction.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._in_transaction:
            # No active transaction, create one
            with self.transaction():
                return func(self, *args, **kwargs)
        else:
            # Transaction already active, just execute
            return func(self, *args, **kwargs)
    return wrapper


class DataAccessLayer:
    """Centralized data access layer to enforce security and business rules."""

    def __init__(self, user_context: UserContext = None, db_path: str = None, connection: sqlite3.Connection = None):
        self.user_context = user_context or UserContext(roles=[UserRoles.SYSTEM])
        self._connection = None
        self._owns_connection = False
        self._in_transaction = False  # Track if we're inside a transaction

        if connection:
            self._connection = connection
            self._owns_connection = False  # DAL does not own externally provided connection
        elif db_path:
            self._connection = get_db_connection(db_path)
            self._owns_connection = True
            init_db(connection=self._connection)  # init_db now handles commit
        else:
            # Default to in-memory if no path or connection is provided
            self._connection = get_db_connection()
            self._owns_connection = True
            init_db(connection=self._connection)  # init_db now handles commit

        self._connection.row_factory = dict_factory

    def close(self):
        """Closes the database connection if this DAL instance owns it."""
        if self._owns_connection and self._connection:
            self._connection.close()
            self._connection = None

    @property
    def connection(self) -> sqlite3.Connection | None:
        """Expose the underlying sqlite connection (read-only)."""
        return self._connection

    def commit(self) -> None:
        """Commit current transaction on the managed connection (no-op if none)."""
        if self._connection:
            self._connection.commit()

    @contextmanager
    def transaction(self):
        """
        Provides a context manager for transactions using the DAL's managed connection.
        The connection remains open after the transaction, unless self._owns_connection is true and self.close() is called.
        """
        if not self._connection:
            raise DatabaseError("No database connection available for transaction.")

        was_in_transaction = self._in_transaction
        self._in_transaction = True
        
        try:
            yield self._connection
            if not was_in_transaction:  # Only commit if we started the transaction
                self._connection.commit()
        except Exception as e:
            if not was_in_transaction:  # Only rollback if we started the transaction
                self._connection.rollback()
            raise DatabaseError(f"Transaction failed: {str(e)}") from e
        finally:
            self._in_transaction = was_in_transaction

    def _authorize(self, required_roles: list[str]):
        """Helper to check if the current user has the required roles."""
        if not self.user_context.has_any_role(required_roles):
            raise DatabaseError("Authorization failed: Insufficient permissions.")

    # Regular operations - accessible by USER, ADMIN and SYSTEM roles
    @transaction_required
    def create_job(self, job_type: str, parameters: dict[str, Any] = None, status: str = "pending") -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_job(job_type, parameters, self._connection, status)

    @transaction_required
    def try_acquire_job_submission_lock(self, job_type: str, parameters: dict[str, Any] | None) -> tuple[bool, int]:
        """Expose submission lock acquisition through DAL.

        Returns (acquired, retry_after_seconds).
        """
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return try_acquire_job_submission_lock(job_type, parameters, self._connection)

    @transaction_required
    def update_job_status(self, job_id: str, status: str, progress: float = None, result_id: str = None, error_message: str = None, status_message: str = None) -> None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return update_job_status(job_id, status, progress, result_id, error_message, status_message, self._connection)

    def get_job(self, job_id: str) -> dict:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_job(job_id, self._connection)

    def list_jobs(self, job_type: str = None, status: str = None, limit: int = 100) -> list[dict]:
        """List jobs with optional filters"""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return list_jobs(
            job_type=job_type,
            status=status,
            limit=limit,
            connection=self._connection,
        )

    def get_job_params(self, job_id: str, param_name: str = None) -> dict[str, Any]:
        """Get job parameters from the database."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_job_params(job_id, connection=self._connection, param_name=param_name)

    def get_job_prediction_month(self, job_id: str) -> date:
        """Get prediction month from job parameters."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_job_prediction_month(job_id, connection=self._connection)

    @transaction_required
    def create_data_upload_result(self, job_id: str, records_processed: int, features_generated: list[str], processing_run_id: int) -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_data_upload_result(job_id, records_processed, features_generated, processing_run_id, self._connection)

    @transaction_required
    def create_or_get_config(self, config_dict: dict[str, Any], is_active: bool = False, source: str | None = None) -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_or_get_config(config_dict, is_active, source, self._connection)

    def get_active_config(self) -> dict[str, Any] | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_active_config(self._connection)

    @transaction_required
    def set_config_active(self, config_id: str, deactivate_others: bool = True) -> bool:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return set_config_active(config_id, deactivate_others, self._connection)

    def get_best_config_by_metric(self, metric_name: str, higher_is_better: bool = True, metric_source: str = "train") -> dict[str, Any] | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_best_config_by_metric(metric_name, higher_is_better, metric_source, self._connection)

    @transaction_required
    def create_model_record(self, model_id: str, job_id: str, model_path: str, created_at: datetime, metadata: dict[str, Any] | None = None, is_active: bool = False) -> None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_model_record(model_id, job_id, model_path, created_at, metadata, is_active, self._connection)

    def get_active_model(self) -> dict[str, Any] | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_active_model(self._connection)

    def get_active_model_primary_metric(self) -> float | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_active_model_primary_metric(self._connection)

    @transaction_required
    def set_model_active(self, model_id: str, deactivate_others: bool = True) -> bool:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return set_model_active(model_id, deactivate_others, self._connection)

    def get_best_model_by_metric(self, metric_name: str, higher_is_better: bool = True) -> dict[str, Any] | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_best_model_by_metric(metric_name, higher_is_better, self._connection)

    def get_recent_models(self, limit: int = 5) -> list[dict[str, Any]]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_recent_models(limit, self._connection)

    # Admin-only operations - require ADMIN role
    @transaction_required
    def delete_model_record_and_file(self, model_id: str) -> bool:
        self._authorize([UserRoles.ADMIN, UserRoles.SYSTEM])
        return delete_model_record_and_file(model_id, self._connection)

    @transaction_required
    def create_training_result(self, job_id: str, model_id: str, config_id: str, metrics: dict[str, Any], duration: int | None) -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_training_result(job_id, model_id, config_id, metrics, duration, self._connection)

    @transaction_required
    def create_prediction_result(self, job_id: str, model_id: str, output_path: str, summary_metrics: dict[str, Any] | None, prediction_month: date | None = None) -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_prediction_result(
            job_id=job_id,
            model_id=model_id,
            output_path=output_path,
            summary_metrics=summary_metrics,
            prediction_month=prediction_month,
            connection=self._connection,
        )

    def get_data_upload_result(self, result_id: str) -> dict:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_data_upload_result(result_id, self._connection)

    def get_training_results(self, result_id: str | None = None, limit: int = 100) -> dict | list[dict]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_training_results(result_id, limit, self._connection)

    def get_prediction_result(self, result_id: str) -> dict:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_prediction_result(result_id, self._connection)

    def get_prediction_results_by_month(
        self,
        prediction_month: date,
        model_id: str | None = None,
    ) -> list[dict]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        # Convert date to string format expected by the database function
        prediction_month_str = prediction_month.isoformat() if prediction_month else None
        return get_prediction_results_by_month(prediction_month_str, model_id, self._connection)

    def get_predictions(self, job_ids: list[str], model_id: str | None = None, prediction_month: date | None = None) -> list[dict]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_predictions(job_ids, model_id, prediction_month, self._connection)

    def get_report_result(self, result_id: str) -> dict:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_report_result(result_id, self._connection)

    @transaction_required
    def create_processing_run(self, start_time: datetime, status: str, source_files: str, end_time: datetime = None) -> int:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_processing_run(start_time, status, source_files, end_time, self._connection)

    @transaction_required
    def update_processing_run(self, run_id: int, status: str, end_time: datetime = None) -> None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return update_processing_run(run_id, status, end_time, self._connection)

    def get_configs(self, limit: int = 5) -> list[dict[str, Any]]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_configs(limit, self._connection)

    # Admin-only operations - require ADMIN role
    @transaction_required
    def delete_configs_by_ids(self, config_ids: list[str]) -> dict[str, Any]:
        self._authorize([UserRoles.ADMIN, UserRoles.SYSTEM])
        return delete_configs_by_ids(config_ids, self._connection)

    def get_all_models(self, limit: int = 100, include_active_status: bool = True) -> list[dict[str, Any]]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_all_models(limit, include_active_status, self._connection)

    # Admin-only operations - require ADMIN role
    @transaction_required
    def delete_models_by_ids(self, model_ids: list[str]) -> dict[str, Any]:
        self._authorize([UserRoles.ADMIN, UserRoles.SYSTEM])
        return delete_models_by_ids(model_ids, self._connection)

    def get_effective_config(self, settings, logger=None):
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_effective_config(settings, self._connection, logger)

    @transaction_required
    def create_tuning_result(self, job_id: str, config_id: str, metrics: dict[str, Any] | None, duration: int | None) -> str:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return create_tuning_result(job_id, config_id, metrics, duration, self._connection)

    def get_tuning_results(self, result_id: str | None = None, metric_name: str | None = None, higher_is_better: bool | None = None, limit: int = 100) -> dict | list[dict]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_tuning_results(self._connection, result_id, metric_name, higher_is_better, limit)

    def get_top_configs(self, limit: int = 5, metric_name: str | None = None, higher_is_better: bool = True, include_active: bool = True, source: str | None = None) -> list[dict[str, Any]]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_top_configs(self._connection, limit, metric_name, higher_is_better, include_active, source)

    @transaction_required
    def insert_retry_event(self, event: dict[str, Any]) -> None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        from deployment.app.db.database import insert_retry_event
        return insert_retry_event(event, self._connection)

    # Admin-only operations - require ADMIN role
    def fetch_recent_retry_events(self, limit: int = 1000) -> list[dict[str, Any]]:
        self._authorize([UserRoles.ADMIN, UserRoles.SYSTEM])
        from deployment.app.db.database import fetch_recent_retry_events
        return fetch_recent_retry_events(limit, self._connection)

    # Admin-only operations - require ADMIN role
    @transaction_required
    def execute_raw_query(self, query: str, params: tuple = (), fetchall: bool = False) -> list[dict] | dict | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return execute_query(query, connection=self._connection, params=params, fetchall=fetchall)

    def auto_activate_best_config_if_enabled(self) -> bool:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return auto_activate_best_config_if_enabled(self._connection)

    def auto_activate_best_model_if_enabled(self) -> bool:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return auto_activate_best_model_if_enabled(self._connection)

    def adjust_dataset_boundaries(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> date | None:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return adjust_dataset_boundaries(start_date, end_date, self._connection)

    def get_latest_prediction_month(self) -> date:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_latest_prediction_month(self._connection)

    def get_feature_dataframe(
        self,
        table_name: str,
        columns: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_feature_dataframe(table_name, columns, self._connection, start_date, end_date)

    def get_report_features(
        self,
        multiidx_ids: list[int] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        feature_subset: list[str] | None = None,
    ) -> list[dict]:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_report_features(
            multiidx_ids=multiidx_ids,
            start_date=start_date,
            end_date=end_date,
            feature_subset=feature_subset,
            connection=self._connection,
        )

    # Batch utility methods
    def execute_query_with_batching(
        self,
        query_template: str,
        ids: list[Any],
        batch_size: int | None = None,
        fetchall: bool = True,
        placeholder_name: str = "placeholders"
    ) -> list[Any]:
        """Execute query with IN clause using batching to avoid SQLite variable limit."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return execute_query_with_batching(
            query_template, ids, batch_size, self._connection, fetchall, placeholder_name
        )

    def execute_many_with_batching(
        self,
        query: str,
        params_list: list[tuple],
        batch_size: int | None = None
    ) -> None:
        """Execute multiple queries through batching to avoid SQLite variable limit."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return execute_many_with_batching(query, params_list, batch_size, self._connection)

    def get_next_prediction_month(self) -> date:
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_next_prediction_month(self._connection)

    @transaction_required
    def insert_predictions(self, result_id: str, model_id: str, prediction_month: date, df: pd.DataFrame):
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return insert_predictions(result_id, model_id, prediction_month, df, self._connection)

    @transaction_required
    def delete_features_by_table(self, table: str) -> None:
        """Delete all records from a feature table."""
        self._authorize([UserRoles.ADMIN, UserRoles.SYSTEM])
        return delete_features_by_table(table, self._connection)

    @transaction_required
    def insert_features_batch(self, table: str, params_list: list[tuple]) -> None:
        """Insert a batch of feature records into the specified table."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return insert_features_batch(table, params_list, self._connection)

    def get_features_by_date_range(self, table: str, start_date: str | None = None, end_date: str | None = None) -> list[dict]:
        """Get features from a table within a date range."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        return get_features_by_date_range(table, start_date, end_date, self._connection)

    def get_multiindex_mapping_batch(self, tuples_to_process: list[tuple]) -> dict[tuple, int]:
        """Get multiindex mapping for a batch of tuples."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        from deployment.app.db.database import get_multiindex_mapping_batch
        return get_multiindex_mapping_batch(tuples_to_process, self._connection)

    @transaction_required
    def get_or_create_multiindex_ids_batch(self, tuples_to_process: list[tuple]) -> tuple[int, ...]:
        """Get or create multiindex IDs for a batch of tuples."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        from deployment.app.db.database import get_or_create_multiindex_ids_batch
        return get_or_create_multiindex_ids_batch(tuples_to_process, self._connection)

    def get_multiindex_mapping_by_ids(self, multiindex_ids: list[int]) -> list[tuple[int, ...]]:
        """Get multiindex mapping data by IDs."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        from deployment.app.db.database import get_multiindex_mapping_by_ids
        return get_multiindex_mapping_by_ids(multiindex_ids, self._connection)

    @transaction_required
    def insert_report_features(self, features_to_insert: list[tuple]) -> None:
        """Insert a batch of report features into the report_features table."""
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        from deployment.app.db.database import insert_report_features
        return insert_report_features(features_to_insert, self._connection)