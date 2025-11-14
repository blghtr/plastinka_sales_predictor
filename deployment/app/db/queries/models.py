"""Database queries for model management."""

import json
import logging
import os
from datetime import datetime
from typing import Any

import asyncpg

from deployment.app.config import get_settings
from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.core import execute_query
from deployment.app.db.types import ALLOWED_METRICS
from deployment.app.db.utils import _is_path_safe, json_default_serializer, split_ids_for_batching

logger = logging.getLogger("plastinka.database")


async def create_model_record(
    model_id: str,
    job_id: str,
    model_path: str,
    created_at: datetime,
    metadata: dict[str, Any] | None = None,
    is_active: bool = False,
    connection: asyncpg.Connection = None,
) -> None:
    """
    Creates a record for a trained model artifact.
    If `is_active` is True, this model will be marked active, deactivating others.

    Args:
        model_id: Unique identifier for the model
        job_id: ID of the job that produced the model
        model_path: Path to the model file
        created_at: Creation timestamp
        metadata: Optional metadata for the model
        is_active: Whether to explicitly set this model as active
        connection: Required database connection to use
    """
    await execute_query(
        query="""
        INSERT INTO models (
            model_id, job_id, model_path, created_at, metadata, is_active
        ) VALUES ($1, $2, $3, $4, $5, $6)
        """,
        connection=connection,
        params=(
            model_id,
            job_id,
            model_path,
            created_at,
            json.dumps(metadata, default=json_default_serializer) if metadata else None,
            is_active,  # PostgreSQL boolean
        ),
    )

    if is_active:
        await execute_query(
            query="""
            UPDATE models
            SET is_active = FALSE
            WHERE model_id != $1
            """,
            connection=connection,
            params=(model_id,),
        )

    logger.info(f"Created model record: {model_id}, active: {is_active}")


async def get_active_model(connection: asyncpg.Connection) -> dict[str, Any] | None:
    """
    Returns the currently active model or None if none is active.

    Args:
        connection: Required database connection to use

    Returns:
        Dictionary with model information if an active model exists,
        otherwise None.
    """
    result = await execute_query(
        query="""
        SELECT model_id, model_path, metadata
        FROM models
        WHERE is_active = TRUE
        LIMIT 1
        """,
        connection=connection
    )
    if not result:
        return None

    model_id = result["model_id"]
    model_path = result["model_path"]
    metadata_str = result.get("metadata")

    try:
        # PostgreSQL JSONB returns as dict, not string
        if isinstance(metadata_str, dict):
            metadata = metadata_str
        elif isinstance(metadata_str, str):
            metadata = json.loads(metadata_str) if metadata_str else {}
        else:
            metadata = {}
    except json.JSONDecodeError:
        logger.warning(f"Could not decode metadata JSON for model {model_id}")
        metadata = {}
    return {"model_id": model_id, "model_path": model_path, "metadata": metadata}


async def get_active_model_primary_metric(connection: asyncpg.Connection) -> float | None:
    """
    Retrieves the primary metric of the currently active model.

    Args:
        connection: Required database connection.

    Returns:
        The value of the primary metric as a float, or None if not found.
    """
    settings = get_settings()
    primary_metric_name = settings.default_metric

    if primary_metric_name not in ALLOWED_METRICS:
        logger.error(f"Primary metric '{primary_metric_name}' is not in ALLOWED_METRICS.")
        return None

    query = f"""
        SELECT
            (tr.metrics->>'{primary_metric_name}')::float as metric_value
        FROM models m
        JOIN training_results tr ON m.model_id = tr.model_id
        WHERE m.is_active = TRUE
        ORDER BY m.created_at DESC
        LIMIT 1
    """

    try:
        result = await execute_query(query, connection=connection)
        if result and result.get("metric_value") is not None:
            return float(result["metric_value"])
        return None
    except (DatabaseError, ValueError, TypeError) as e:
        logger.error(f"Could not retrieve or cast active model's primary metric: {e}", exc_info=True)
        return None


async def set_model_active(
    model_id: str,
    deactivate_others: bool = True,
    connection: asyncpg.Connection = None,
) -> bool:
    """
    Sets a model as active and optionally deactivates others.

    Args:
        model_id: The model ID to activate
        deactivate_others: Whether to deactivate all other models
        connection: Required database connection to use

    Returns:
        True if successful, False otherwise
    """
    # First check if model exists
    result = await execute_query(
        query="SELECT 1 FROM models WHERE model_id = $1",
        connection=connection,
        params=(model_id,)
    )

    if not result:
        logger.error(f"Model {model_id} not found")
        return False

    if deactivate_others:
        await execute_query(
            query="UPDATE models SET is_active = FALSE",
            connection=connection
        )

    await execute_query(
        query="UPDATE models SET is_active = TRUE WHERE model_id = $1",
        connection=connection,
        params=(model_id,)
    )

    logger.info(f"Set model {model_id} as active")
    return True


async def get_best_model_by_metric(
    metric_name: str,
    higher_is_better: bool = True,
    connection: asyncpg.Connection = None,
) -> dict[str, Any] | None:
    """
    Returns the model with the best metric value based on training_results.

    Args:
        metric_name: The name of the metric to use for evaluation
        higher_is_better: True if higher values of the metric are better, False otherwise
        connection: Required database connection to use

    Returns:
        Dictionary with model information if a best model exists, otherwise None.
    """
    if metric_name not in ALLOWED_METRICS:
        logger.error(
            f"Invalid metric_name '{metric_name}' provided to get_best_model_by_metric."
        )
        raise ValueError(
            f"Invalid metric_name: {metric_name}. Allowed metrics are: {ALLOWED_METRICS}"
        )

    order_direction = "DESC" if higher_is_better else "ASC"
    json_path = f"'{metric_name}'"

    query = f"""
        SELECT
            m.model_id,
            m.model_path,
            m.metadata,
            tr.metrics,
            (tr.metrics->>{json_path})::float as metric_value
        FROM training_results tr
        JOIN models m ON tr.model_id = m.model_id
        WHERE tr.model_id IS NOT NULL 
          AND tr.metrics IS NOT NULL 
          AND jsonb_typeof(tr.metrics) = 'object' 
          AND (tr.metrics->>{json_path}) IS NOT NULL
        ORDER BY metric_value {order_direction}
        LIMIT 1
    """

    result = await execute_query(query=query, connection=connection)

    if result:
        # PostgreSQL JSONB returns as dict, not string
        if result.get("metadata"):
            if isinstance(result["metadata"], dict):
                result["metadata"] = result["metadata"]
            elif isinstance(result["metadata"], str):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode metadata JSON for model {result['model_id']}"
                    )
                    result["metadata"] = {}
            else:
                result["metadata"] = {}
        else:
            result["metadata"] = {}

        if result.get("metrics"):
            if isinstance(result["metrics"], dict):
                result["metrics"] = result["metrics"]
            elif isinstance(result["metrics"], str):
                try:
                    result["metrics"] = json.loads(result["metrics"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode metrics JSON for model {result['model_id']} in training_results"
                    )
                    result["metrics"] = {}
            else:
                result["metrics"] = {}
        else:
            result["metrics"] = {}

        result.pop("metric_value", None)
        return result

    return None


async def get_recent_models(
    limit: int = 5,
    connection: asyncpg.Connection = None
) -> list[dict[str, Any]]:
    """
    Get the most recent models from the database.
    """
    query = """
        SELECT model_id, job_id, model_path, created_at, metadata, is_active
        FROM models
        ORDER BY created_at DESC
        LIMIT $1
    """
    results = await execute_query(
        query=query,
        connection=connection,
        params=(limit,),
        fetchall=True
    ) or []
    
    # Parse JSONB metadata
    for result in results:
        if "metadata" in result and result["metadata"]:
            if isinstance(result["metadata"], dict):
                # Already parsed by asyncpg
                pass
            elif isinstance(result["metadata"], str):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    result["metadata"] = {}
    
    return results


async def delete_model_record_and_file(
    model_id: str,
    connection: asyncpg.Connection = None
) -> bool:
    """
    Deletes a model record from the database and its associated file from the filesystem.
    Returns True if successful, False otherwise.
    """
    # Get the model path before deleting the record
    result = await execute_query(
        query="SELECT model_path FROM models WHERE model_id = $1",
        connection=connection,
        params=(model_id,)
    )
    if not result:
        logger.warning(f"Model with ID {model_id} not found for deletion.")
        return False

    model_path = result["model_path"]
    models_base_dir = get_settings().models_dir

    # 1. Delete the model file safely (with path traversal protection)
    if model_path and os.path.exists(model_path):
        if not _is_path_safe(models_base_dir, model_path):
            logger.error(
                f"Path traversal attempt detected for model {model_id}. Path '{model_path}' is outside of designated models directory '{models_base_dir}'. File will not be deleted."
            )
        else:
            try:
                os.remove(model_path)
                logger.info(f"Deleted model file: {model_path}")
            except OSError as e:
                logger.error(
                    f"Error removing model file {model_path}: {e}", exc_info=True
                )

    # 2. Delete dependent records from training_results
    await execute_query(
        query="DELETE FROM training_results WHERE model_id = $1",
        connection=connection,
        params=(model_id,)
    )

    # 3. Delete the model record
    await execute_query(
        query="DELETE FROM models WHERE model_id = $1",
        connection=connection,
        params=(model_id,)
    )
    return True


async def get_all_models(
    limit: int = 100,
    include_active_status: bool = True,
    connection: asyncpg.Connection = None,
) -> list[dict[str, Any]]:
    """
    Retrieves a list of all models with their details.

    Args:
        limit: Maximum number of models to return
        include_active_status: Whether to include the active status in the results
        connection: Required database connection to use

    Returns:
        List of models with their details
    """
    results = await execute_query(
        query="""
        SELECT model_id, job_id, model_path, created_at, metadata, is_active
        FROM models
        ORDER BY created_at DESC
        LIMIT $1
        """,
        connection=connection,
        params=(limit,),
        fetchall=True
    ) or []

    # Parse JSONB metadata
    for result in results:
        if "metadata" in result and result["metadata"]:
            if isinstance(result["metadata"], dict):
                # Already parsed by asyncpg
                pass
            elif isinstance(result["metadata"], str):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    result["metadata"] = {}

    return results


async def delete_models_by_ids(
    model_ids: list[str],
    connection: asyncpg.Connection
) -> dict[str, Any]:
    """
    Deletes multiple model records by their IDs, skipping active models.

    Args:
        model_ids: List of model IDs to delete.
        connection: Required database connection.

    Returns:
        A dictionary with a deletion summary.
    """
    if not model_ids:
        return {"deleted_count": 0, "skipped_count": 0, "skipped_models": []}

    summary = {"deleted_count": 0, "skipped_count": 0, "skipped_models": []}

    # Find which models are active
    # PostgreSQL can handle large IN clauses, but we'll still batch for performance
    
    active_model_ids = set()
    for batch in split_ids_for_batching(model_ids, batch_size=1000):
        placeholders = ", ".join(f"${i+1}" for i in range(len(batch)))
        query = f"SELECT model_id FROM models WHERE is_active = TRUE AND model_id IN ({placeholders})"
        active_models = await execute_query(
            query, connection=connection, params=tuple(batch), fetchall=True
        ) or []
        active_model_ids.update(m["model_id"] for m in active_models)

    models_to_delete = []
    for model_id in model_ids:
        if model_id in active_model_ids:
            summary["skipped_count"] += 1
            summary["skipped_models"].append(model_id)
            logger.warning(f"Skipping deletion of active model: {model_id}")
        else:
            models_to_delete.append(model_id)

    if models_to_delete:
        # Delete dependent records from training_results first
        for batch in split_ids_for_batching(models_to_delete, batch_size=1000):
            placeholders = ", ".join(f"${i+1}" for i in range(len(batch)))
            query = f"DELETE FROM training_results WHERE model_id IN ({placeholders})"
            await execute_query(query, connection=connection, params=tuple(batch))

        # Delete the model records
        for batch in split_ids_for_batching(models_to_delete, batch_size=1000):
            placeholders = ", ".join(f"${i+1}" for i in range(len(batch)))
            query = f"DELETE FROM models WHERE model_id IN ({placeholders})"
            await execute_query(query, connection=connection, params=tuple(batch))
        
        summary["deleted_count"] = len(models_to_delete)

        # Delete associated model files
        for model_id in models_to_delete:
            await delete_model_record_and_file(model_id, connection=connection)

    return summary


async def auto_activate_best_model_if_enabled(connection: asyncpg.Connection = None) -> bool:
    """
    Automatically activates the best model by metric if auto_select_best_model is enabled.

    Args:
        connection: Required database connection to use

    Returns:
        bool: True if activation was performed, False if disabled or no model found

    Raises:
        DatabaseError: If database operations fail
    """
    settings = get_settings()

    if not settings.auto_select_best_model:
        return False

    try:
        best_model = await get_best_model_by_metric(
            metric_name=settings.default_metric,
            higher_is_better=settings.default_metric_higher_is_better,
            connection=connection,
        )

        if best_model and best_model.get("model_id"):
            model_id = best_model["model_id"]
            logger.info(f"Auto-activating best model by {settings.default_metric}: {model_id}")

            success = await set_model_active(model_id, connection=connection)
            if success:
                logger.info(f"Successfully auto-activated model: {model_id}")
                return True
            else:
                logger.warning(f"Failed to auto-activate model: {model_id}")
                return False
        else:
            logger.warning(
                f"Auto-activation enabled, but no best model found by metric '{settings.default_metric}'"
            )
            return False

    except Exception as e:
        logger.error(f"Error during auto-activation of best model: {e}", exc_info=True)
        # Don't re-raise - auto-activation should not break the main flow
        return False

