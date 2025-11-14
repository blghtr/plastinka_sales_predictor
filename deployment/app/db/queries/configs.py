"""Database queries for configuration management."""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

import asyncpg
from pydantic import ValidationError

from deployment.app.models.api_models import TrainingConfig
from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.core import execute_query
from deployment.app.db.types import ALLOWED_METRICS
from deployment.app.db.utils import json_default_serializer, split_ids_for_batching

logger = logging.getLogger("plastinka.database")


async def create_or_get_config(
    config_dict: dict[str, Any],
    is_active: bool = False,
    source: str | None = None,
    connection: asyncpg.Connection = None,
) -> str:
    """
    Creates a config record if it doesn't exist, based on a hash of the config.
    If `is_active` is True, this config will be marked active, deactivating others.
    Returns the config_id.

    Args:
        config_dict: Dictionary of config
        is_active: Whether to explicitly set this config as active
        source: Optional source for the config
        connection: Required database connection to use

    Returns:
        Config ID (hash)
    """
    try:
        _ = TrainingConfig(**config_dict)
    except (ValidationError, ValueError) as e:
        logger.error(f"Configuration is invalid: {e}", exc_info=True)
        raise ValueError(f"Invalid configuration provided: {e}") from e

    logger.debug(f"Storing config: {config_dict}")

    config_json = json.dumps(
        config_dict, sort_keys=True, default=json_default_serializer
    )
    config_id = hashlib.md5(config_json.encode()).hexdigest()

    result = await execute_query(
        query="SELECT config_id FROM configs WHERE config_id = $1",
        connection=connection,
        params=(config_id,)
    )

    if not result:
        now = datetime.now()

        await execute_query(
            query="""
            INSERT INTO configs (config_id, config, is_active, created_at, source)
            VALUES ($1, $2, $3, $4, $5)
            """,
            connection=connection,
            params=(config_id, config_json, is_active, now, source),
        )

        logger.info(
            f"Created config: {config_id}, active: {is_active}, source: {source}"
        )
    else:
        if source is not None:
            await execute_query(
                query="UPDATE configs SET source = $1 WHERE config_id = $2",
                connection=connection,
                params=(source, config_id)
            )

    if is_active:
        await _activate_config(connection, config_id)
        logger.info(f"Set config {config_id} as active")

    return config_id


async def _activate_config(connection: asyncpg.Connection, config_id: str) -> None:
    """Helper function to activate a config and deactivate others."""
    await execute_query(
        query="UPDATE configs SET is_active = FALSE WHERE config_id != $1",
        connection=connection,
        params=(config_id,)
    )
    await execute_query(
        query="UPDATE configs SET is_active = TRUE WHERE config_id = $1",
        connection=connection,
        params=(config_id,)
    )


async def get_active_config(connection: asyncpg.Connection) -> dict[str, Any] | None:
    """
    Returns the currently active config or None if none is active.
    
    Args:
        connection: Required database connection to use
        
    Returns:
        Dictionary with config_id and config fields if an active config exists,
        otherwise None.
    """
    result = await execute_query(
        query="""
        SELECT config_id, config
        FROM configs
        WHERE is_active = TRUE
        LIMIT 1
        """,
        connection=connection
    )
    if not result:
        return None

    config_id = result["config_id"]
    config_json = result["config"]
    try:
        # PostgreSQL JSONB returns as dict, not string
        if isinstance(config_json, dict):
            config = config_json
        elif isinstance(config_json, str):
            config = json.loads(config_json) if config_json else {}
        else:
            config = {}
    except (json.JSONDecodeError, TypeError):
        logger.error(f"Error parsing config JSON: {config_json}")
        config = {}
    return {"config_id": config_id, "config": config}


async def set_config_active(
    config_id: str,
    deactivate_others: bool = True,
    connection: asyncpg.Connection = None,
) -> bool:
    """
    Sets a config as active and optionally deactivates others.

    Args:
        config_id: The config ID to activate
        deactivate_others: Whether to deactivate all other configs
        connection: Required database connection to use

    Returns:
        True if successful, False otherwise
    """
    # First check if config exists
    result = await execute_query(
        query="SELECT 1 FROM configs WHERE config_id = $1",
        connection=connection,
        params=(config_id,)
    )

    if not result:
        logger.error(f"Config {config_id} not found")
        return False

    if deactivate_others:
        await execute_query(
            query="UPDATE configs SET is_active = FALSE",
            connection=connection
        )

    await execute_query(
        query="UPDATE configs SET is_active = TRUE WHERE config_id = $1",
        connection=connection,
        params=(config_id,)
    )

    logger.info(f"Set config {config_id} as active")
    return True


async def get_best_config_by_metric(
    metric_name: str,
    higher_is_better: bool = True,
    metric_source: str = "train",
    connection: asyncpg.Connection = None,
) -> dict[str, Any] | None:
    """
    Returns the config with the best metric value by searching across both
    training_results and tuning_results.

    Args:
        metric_name: The name of the metric to use for evaluation.
        higher_is_better: True if higher values of the metric are better.
        metric_source: The source of the metric, either 'train' or 'tune'.
        connection: Required database connection.

    Returns:
        A dictionary with config_id, config, and metrics, or None if no config is found.
    """
    # Use the generalized get_top_configs function to find the single best config
    # Call get_top_configs directly (defined in same module)
    top_configs = await get_top_configs(
        connection=connection,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        limit=1,
    )

    if not top_configs:
        return None

    # get_top_configs returns a list, so we take the first element
    best_config_result = top_configs[0]

    # The result from get_top_configs already includes `config_id`, `config`, and `metrics`
    # in the desired format, so we can return it directly.
    return best_config_result


async def get_configs(
    limit: int = 5,
    connection: asyncpg.Connection = None
) -> list[dict[str, Any]]:
    """
    Retrieves a list of configs ordered by creation date.

    Args:
        limit: Maximum number of configs to return
        connection: Required database connection to use

    Returns:
        List of configs with their details
    """
    results = await execute_query(
        query="""
        SELECT config_id, config, created_at, is_active
        FROM configs
        ORDER BY created_at DESC
        LIMIT $1
        """,
        connection=connection,
        params=(limit,),
        fetchall=True
    ) or []

    # Parse JSONB config
    for result in results:
        if "config" in result and result["config"]:
            if isinstance(result["config"], dict):
                # Already parsed by asyncpg
                pass
            elif isinstance(result["config"], str):
                try:
                    result["config"] = json.loads(result["config"])
                except json.JSONDecodeError:
                    result["config"] = {}

    return results


async def get_top_configs(
    connection: asyncpg.Connection,
    limit: int = 5,
    metric_name: str | None = None,
    higher_is_better: bool = True,
    include_active: bool = True,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Return best historical configs for seeding tuning.

    Selection rules:
    1. Optionally include the currently active config first.
    2. Then order configs by metric from both *training_results* and *tuning_results* tables.
       Rows where metric is NULL are ignored.
    3. Falls back to most recently created configs if no metrics.
    """
    from datetime import datetime
    
    metric_name = metric_name or "val_MIC"
    if metric_name not in ALLOWED_METRICS:
        logger.error("get_top_configs: Invalid metric name provided. Raising error.")
        raise ValueError("Invalid metric_name")
    order = "DESC" if higher_is_better else "ASC"

    json_path = f"'{metric_name}'"
    higher_better_str = "TRUE" if higher_is_better else "FALSE"

    sql = f"""
        WITH config_metrics AS (
            SELECT
                c.config_id,
                c.config,
                c.created_at,
                c.is_active,
                c.source,
                (SELECT (tr.metrics->>{json_path})::float
                 FROM training_results tr
                 WHERE tr.config_id = c.config_id
                 AND tr.metrics IS NOT NULL AND jsonb_typeof(tr.metrics) = 'object' AND (tr.metrics->>{json_path}) IS NOT NULL
                 ORDER BY (tr.metrics->>{json_path})::float {order} LIMIT 1) as training_metric,
                (SELECT (tu.metrics->>{json_path})::float
                 FROM tuning_results tu
                 WHERE tu.config_id = c.config_id
                 AND tu.metrics IS NOT NULL AND jsonb_typeof(tu.metrics) = 'object' AND (tu.metrics->>{json_path}) IS NOT NULL
                 ORDER BY (tu.metrics->>{json_path})::float {order} LIMIT 1) as tuning_metric
            FROM configs c
        ),
        final_metrics AS (
            SELECT
                config_id,
                config,
                created_at,
                is_active,
                source,
                CASE
                    WHEN training_metric IS NOT NULL AND tuning_metric IS NOT NULL THEN
                        CASE WHEN {higher_better_str} THEN
                            GREATEST(training_metric, tuning_metric)
                        ELSE
                            LEAST(training_metric, tuning_metric)
                        END
                    WHEN training_metric IS NOT NULL THEN training_metric
                    WHEN tuning_metric IS NOT NULL THEN tuning_metric
                    ELSE NULL
                END as best_metric,
                (SELECT metrics FROM training_results WHERE config_id = config_metrics.config_id ORDER BY (metrics->>{json_path})::float {order} LIMIT 1) as training_metrics_json,
                (SELECT metrics FROM tuning_results WHERE config_id = config_metrics.config_id ORDER BY (metrics->>{json_path})::float {order} LIMIT 1) as tuning_metrics_json
            FROM config_metrics
        )
        SELECT config_id, config, created_at, is_active, source, best_metric, training_metrics_json, tuning_metrics_json
        FROM final_metrics
    """

    where_conditions = []

    if not include_active:
        where_conditions.append("is_active = FALSE")

    if source:
        where_conditions.append(f"source = $2")
        # Note: We'll need to adjust params if source is provided

    if where_conditions:
        sql += " WHERE " + " AND ".join(where_conditions)

    sql += f" ORDER BY is_active DESC, best_metric {order} NULLS LAST, created_at DESC LIMIT $1"
    
    params = (limit,)
    if source:
        # Replace $2 with actual source value in SQL
        sql = sql.replace("source = $2", f"source = '{source}'")

    rows = await execute_query(sql, connection, params, fetchall=True) or []

    top_cfgs = []
    for r in rows:
        try:
            new_row = {
                "config_id": r["config_id"],
                "config": json.loads(r["config"]) if isinstance(r["config"], str) else r["config"],
                "created_at": r["created_at"],
                "is_active": bool(r["is_active"]),
                "source": r["source"],
                "metrics": {},
            }

            training_metrics = r.get('training_metrics_json')
            tuning_metrics = r.get('tuning_metrics_json')

            if training_metrics:
                if isinstance(training_metrics, dict):
                    new_row["metrics"] = training_metrics
                elif isinstance(training_metrics, str):
                    new_row["metrics"] = json.loads(training_metrics)
                else:
                    new_row["metrics"] = {}
            elif tuning_metrics:
                if isinstance(tuning_metrics, dict):
                    new_row["metrics"] = tuning_metrics
                elif isinstance(tuning_metrics, str):
                    new_row["metrics"] = json.loads(tuning_metrics)
                else:
                    new_row["metrics"] = {}
            else:
                new_row["metrics"] = {}

            top_cfgs.append(new_row)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse config or metrics JSON for {r.get('config_id')}: {e}")
            continue
    return top_cfgs


async def delete_configs_by_ids(
    config_ids: list[str],
    connection: asyncpg.Connection = None
) -> dict[str, Any]:
    """
    Deletes multiple config records by their IDs, skipping active configs.

    Args:
        config_ids: List of config IDs to delete.
        connection: Required database connection.

    Returns:
        A dictionary with a deletion summary.
    """
    if not config_ids:
        return {"deleted_count": 0, "skipped_count": 0, "skipped_configs": []}

    summary = {"deleted_count": 0, "skipped_count": 0, "skipped_configs": []}

    # Use batching for large lists

    # Find which configs are active
    active_configs = set()
    for batch in split_ids_for_batching(config_ids, batch_size=1000):
        placeholders = ", ".join(f"${i+1}" for i in range(len(batch)))
        query = f"SELECT config_id FROM configs WHERE config_id IN ({placeholders}) AND is_active = TRUE"
        active_results = await execute_query(
            query, connection=connection, params=tuple(batch), fetchall=True
        ) or []
        active_configs.update(row["config_id"] for row in active_results)

    summary["skipped_configs"] = list(active_configs)
    summary["skipped_count"] = len(active_configs)

    configs_to_delete = [cid for cid in config_ids if cid not in active_configs]

    if not configs_to_delete:
        return summary

    # Find jobs using these configs
    used_configs_in_jobs = set()
    for batch in split_ids_for_batching(configs_to_delete, batch_size=1000):
        placeholders = ", ".join(f"${i+1}" for i in range(len(batch)))
        query = f"SELECT config_id FROM jobs WHERE config_id IN ({placeholders})"
        jobs_results = await execute_query(
            query, connection=connection, params=tuple(batch), fetchall=True
        ) or []
        used_configs_in_jobs.update(row["config_id"] for row in jobs_results)

    # Also check training_results
    used_configs_in_results = set()
    for batch in split_ids_for_batching(configs_to_delete, batch_size=1000):
        placeholders = ", ".join(f"${i+1}" for i in range(len(batch)))
        query = f"SELECT config_id FROM training_results WHERE config_id IN ({placeholders})"
        training_results = await execute_query(
            query, connection=connection, params=tuple(batch), fetchall=True
        ) or []
        used_configs_in_results.update(row["config_id"] for row in training_results)

    used_configs = used_configs_in_jobs.union(used_configs_in_results)

    # Add used configs to the skipped list
    newly_skipped = used_configs.difference(set(summary["skipped_configs"]))
    if newly_skipped:
        summary["skipped_configs"].extend(list(newly_skipped))
        summary["skipped_count"] += len(newly_skipped)

    # Recalculate configs to delete
    final_configs_to_delete = [
        cid for cid in configs_to_delete if cid not in used_configs
    ]

    if not final_configs_to_delete:
        return summary

    # Delete the unused, non-active configs using batching
    deleted_count = 0
    for batch in split_ids_for_batching(final_configs_to_delete, batch_size=1000):
        placeholders = ", ".join(f"${i+1}" for i in range(len(batch)))
        await execute_query(
            query=f"DELETE FROM configs WHERE config_id IN ({placeholders})",
            connection=connection,
            params=tuple(batch),
        )
        deleted_count += len(batch)

    summary["deleted_count"] = deleted_count

    return summary


async def auto_activate_best_config_if_enabled(connection: asyncpg.Connection = None) -> bool:
    """
    Automatically activates the best config by metric if auto_select_best_configs is enabled.

    Args:
        connection: Required database connection to use

    Returns:
        bool: True if activation was performed, False if disabled or no config found

    Raises:
        DatabaseError: If database operations fail
    """
    from deployment.app.config import get_settings
    
    settings = get_settings()

    if not settings.auto_select_best_configs:
        return False

    try:
        best_config = await get_best_config_by_metric(
            metric_name=settings.default_metric,
            higher_is_better=settings.default_metric_higher_is_better,
            connection=connection,
        )

        if best_config and best_config.get("config_id"):
            config_id = best_config["config_id"]
            logger.info(f"Auto-activating best config by {settings.default_metric}: {config_id}")

            success = await set_config_active(config_id, connection=connection)
            if success:
                logger.info(f"Successfully auto-activated config: {config_id}")
                return True
            else:
                logger.warning(f"Failed to auto-activate config: {config_id}")
                return False
        else:
            logger.warning(
                f"Auto-activation enabled, but no best config found by metric '{settings.default_metric}'"
            )
            return False

    except Exception as e:
        logger.error(f"Error during auto-activation of best config: {e}", exc_info=True)
        # Don't re-raise - auto-activation should not break the main flow
        return False


async def get_effective_config(settings, logger=None, *, connection: asyncpg.Connection):
    """
    Determines the effective configuration by first trying to load the active config
    from the database. If no active config is found or an error occurs, it falls back
    to the provided settings.

    Args:
        settings: The current application settings object.
        logger: Optional logger instance.
        connection: Required database connection (keyword-only).

    Returns:
        A dictionary representing the effective configuration.
    """
    active_config_data = await get_active_config(connection=connection)
    if active_config_data:
        if logger:
            logger.info(f"Found active config: {active_config_data['config_id']}")
        return active_config_data
    # Fallback: best by metric
    metric_name = getattr(settings, "default_metric", None)
    higher_is_better = getattr(settings, "default_metric_higher_is_better", True)
    best_config_data = await get_best_config_by_metric(
        metric_name, higher_is_better, connection=connection
    )
    if best_config_data:
        if logger:
            logger.info(
                f"Using best config by {metric_name}: {best_config_data['config_id']}"
            )
        return best_config_data
    error_msg = "No active config and no best config by metric available"
    if logger:
        logger.error(error_msg)
    raise ValueError(error_msg)

