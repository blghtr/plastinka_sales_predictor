"""Refactored DataAccessLayer with QueryMethod descriptors and async PostgreSQL support."""

from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any

import asyncpg
from asyncpg import Pool

from deployment.app.db.connection import get_db_pool, transaction
from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.query_method import QueryMethod
from deployment.app.db import queries
from deployment.app.db.utils import split_ids_for_batching

# Define roles/permissions
class UserRoles:
    ADMIN = "admin"
    USER = "user"
    SYSTEM = "system"


# A placeholder for the current user's context (e.g., roles)
class UserContext:
    def __init__(self, roles: list[str]):
        self.roles = roles

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_any_role(self, roles: list[str]) -> bool:
        return any(role in self.roles for role in roles)


class DataAccessLayer:
    """Centralized data access layer with automatic authorization and transaction management."""

    def __init__(self, user_context: UserContext = None, pool: Pool = None):
        """
        Initialize DataAccessLayer.
        
        Args:
            user_context: User context with roles. Defaults to SYSTEM role.
            pool: PostgreSQL connection pool. If None, uses global pool.
        """
        self.user_context = user_context or UserContext(roles=[UserRoles.SYSTEM])
        self._pool = pool or get_db_pool()

    def _authorize(self, required_roles: list[str]):
        """Helper to check if the current user has the required roles."""
        if not self.user_context.has_any_role(required_roles):
            raise DatabaseError("Authorization failed: Insufficient permissions.")

    # Jobs - using QueryMethod descriptors
    create_job = QueryMethod(
        queries.create_job,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    try_acquire_job_submission_lock = QueryMethod(
        queries.try_acquire_job_submission_lock,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    update_job_status = QueryMethod(
        queries.update_job_status,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_job = QueryMethod(
        queries.get_job,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    list_jobs = QueryMethod(
        queries.list_jobs,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_job_params = QueryMethod(
        queries.get_job_params,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_job_prediction_month = QueryMethod(
        queries.get_job_prediction_month,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )

    # Configs
    create_or_get_config = QueryMethod(
        queries.create_or_get_config,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_active_config = QueryMethod(
        queries.get_active_config,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    set_config_active = QueryMethod(
        queries.set_config_active,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_best_config_by_metric = QueryMethod(
        queries.get_best_config_by_metric,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_configs = QueryMethod(
        queries.get_configs,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_top_configs = QueryMethod(
        queries.get_top_configs,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    delete_configs_by_ids = QueryMethod(
        queries.delete_configs_by_ids,
        required_roles=[UserRoles.ADMIN, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    auto_activate_best_config_if_enabled = QueryMethod(
        queries.auto_activate_best_config_if_enabled,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_effective_config = QueryMethod(
        queries.get_effective_config,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )

    # Models
    create_model_record = QueryMethod(
        queries.create_model_record,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_active_model = QueryMethod(
        queries.get_active_model,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_active_model_primary_metric = QueryMethod(
        queries.get_active_model_primary_metric,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    set_model_active = QueryMethod(
        queries.set_model_active,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_best_model_by_metric = QueryMethod(
        queries.get_best_model_by_metric,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_recent_models = QueryMethod(
        queries.get_recent_models,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    delete_model_record_and_file = QueryMethod(
        queries.delete_model_record_and_file,
        required_roles=[UserRoles.ADMIN, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_all_models = QueryMethod(
        queries.get_all_models,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    delete_models_by_ids = QueryMethod(
        queries.delete_models_by_ids,
        required_roles=[UserRoles.ADMIN, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    auto_activate_best_model_if_enabled = QueryMethod(
        queries.auto_activate_best_model_if_enabled,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )

    # Results
    create_training_result = QueryMethod(
        queries.create_training_result,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    create_tuning_result = QueryMethod(
        queries.create_tuning_result,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    create_prediction_result = QueryMethod(
        queries.create_prediction_result,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_training_results = QueryMethod(
        queries.get_training_results,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_tuning_results = QueryMethod(
        queries.get_tuning_results,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_prediction_result = QueryMethod(
        queries.get_prediction_result,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_prediction_results_by_month = QueryMethod(
        queries.get_prediction_results_by_month,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_predictions = QueryMethod(
        queries.get_predictions,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    insert_predictions = QueryMethod(
        queries.insert_predictions,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_data_upload_result = QueryMethod(
        queries.get_data_upload_result,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    create_data_upload_result = QueryMethod(
        queries.create_data_upload_result,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_report_result = QueryMethod(
        queries.get_report_result,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )

    # Processing runs
    create_processing_run = QueryMethod(
        queries.create_processing_run,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    update_processing_run = QueryMethod(
        queries.update_processing_run,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )

    # Features
    get_feature_dataframe = QueryMethod(
        queries.get_feature_dataframe,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_features_by_date_range = QueryMethod(
        queries.get_features_by_date_range,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    insert_features_batch = QueryMethod(
        queries.insert_features_batch,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    delete_features_by_table = QueryMethod(
        queries.delete_features_by_table,
        required_roles=[UserRoles.ADMIN, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_report_features = QueryMethod(
        queries.get_report_features,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    insert_report_features = QueryMethod(
        queries.insert_report_features,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    adjust_dataset_boundaries = QueryMethod(
        queries.adjust_dataset_boundaries,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )

    # Multiindex
    get_or_create_multiindex_ids_batch = QueryMethod(
        queries.get_or_create_multiindex_ids_batch,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    get_multiindex_mapping_by_ids = QueryMethod(
        queries.get_multiindex_mapping_by_ids,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_multiindex_mapping_batch = QueryMethod(
        queries.get_multiindex_mapping_batch,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )

    # Predictions
    get_next_prediction_month = QueryMethod(
        queries.get_next_prediction_month,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )
    
    get_latest_prediction_month = QueryMethod(
        queries.get_latest_prediction_month,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=False
    )

    # Retry events
    insert_retry_event = QueryMethod(
        queries.insert_retry_event,
        required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
        requires_transaction=True
    )
    
    fetch_recent_retry_events = QueryMethod(
        queries.fetch_recent_retry_events,
        required_roles=[UserRoles.ADMIN, UserRoles.SYSTEM],
        requires_transaction=False
    )

    # Raw query execution methods for admin/debugging purposes
    async def execute_raw_query(
        self,
        query: str,
        params: tuple = (),
        fetchall: bool = False
    ) -> list[dict] | dict | None:
        """
        Execute a raw SQL query. Use with caution - no automatic parameterization.
        For admin/debugging purposes only.
        """
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        from deployment.app.db.queries.core import execute_query
        
        async with self._pool.acquire() as conn:
            return await execute_query(query, connection=conn, params=params, fetchall=fetchall)

    async def execute_query_with_batching(
        self,
        query_template: str,
        ids: list[Any],
        batch_size: int | None = None,
        fetchall: bool = True,
        placeholder_name: str = "placeholders"
    ) -> list[Any]:
        """
        Execute query with IN clause using batching for large lists.
        """
        self._authorize([UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM])
        from deployment.app.db.queries.core import execute_query
        
        all_results = []
        for batch in split_ids_for_batching(ids, batch_size=batch_size):
            placeholders = ", ".join(f"${i+1}" for i in range(len(batch)))
            query = query_template.replace(f"{{{placeholder_name}}}", placeholders)
            
            async with self._pool.acquire() as conn:
                batch_results = await execute_query(
                    query, connection=conn, params=tuple(batch), fetchall=fetchall
                )
                if batch_results:
                    if isinstance(batch_results, list):
                        all_results.extend(batch_results)
                    else:
                        all_results.append(batch_results)
        
        return all_results

