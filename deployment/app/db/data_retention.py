"""
Data retention management for optimizing database storage.

This module provides functions for implementing data retention policies,
including cleanup of historical predictions and management of model files.
"""

import logging
from datetime import datetime, timedelta

from ..config import get_settings
from .connection import get_db_pool
from .data_access_layer import (  # Import DataAccessLayer
    DataAccessLayer,
    UserContext,
    UserRoles,
)
from .types import ALLOWED_METRICS

logger = logging.getLogger(__name__)


async def cleanup_old_predictions(days_to_keep: int | None = None, dal: DataAccessLayer = None) -> int:
    """
    Remove prediction records older than the specified retention period.

    Args:
        days_to_keep: Number of days to keep predictions for.
                      If None, uses the value from settings.
        dal: Optional DataAccessLayer instance. If None, a new one is created.

    Returns:
        Number of records removed
    """
    if days_to_keep is None:
        days_to_keep = get_settings().data_retention.prediction_retention_days

    if dal is None:
        pool = get_db_pool()
        dal = DataAccessLayer(user_context=UserContext(roles=[UserRoles.SYSTEM]), pool=pool)

    retention_date = datetime.now() - timedelta(days=days_to_keep)
    cutoff_date = retention_date.date()  # Use date object for asyncpg

    try:
        # Count records to be deleted
        count_result = await dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_predictions WHERE prediction_month < $1",
            (cutoff_date,),
            fetchall=False,
        )
        count = count_result["count"] if count_result else 0
        logger.info(f"Found {count} predictions older than {cutoff_date}")

        if count > 0:
            await dal.execute_raw_query(
                "DELETE FROM fact_predictions WHERE prediction_month < $1",
                (cutoff_date,),
            )
            logger.info(f"Deleted {count} predictions older than {cutoff_date}")
        return count

    except Exception as e:
        logger.error(f"Error cleaning up old predictions: {str(e)}")
        return 0


async def cleanup_old_historical_data(
        sales_days_to_keep: int | None = None,
        stock_days_to_keep: int | None = None,
        dal: DataAccessLayer = None,
    ) -> dict[str, int]:
    """
    Remove historical sales and stock data older than the specified retention period.

    Args:
        sales_days_to_keep: Number of days to keep sales data.
                           If None, uses the value from settings.
        stock_days_to_keep: Number of days to keep stock data.
                           If None, uses the value from settings.
        dal: Optional DataAccessLayer instance. If None, a new one is created.

    Returns:
        Dictionary with counts of removed records by type
    """
    if sales_days_to_keep is None:
        sales_days_to_keep = get_settings().data_retention.sales_retention_days
    if stock_days_to_keep is None:
        stock_days_to_keep = get_settings().data_retention.stock_retention_days

    if dal is None:
        pool = get_db_pool()
        dal = DataAccessLayer(user_context=UserContext(roles=[UserRoles.SYSTEM]), pool=pool)

    # Calculate cutoff dates (use date objects for asyncpg)
    sales_cutoff = (datetime.now() - timedelta(days=sales_days_to_keep)).date()
    stock_cutoff = (datetime.now() - timedelta(days=stock_days_to_keep)).date()

    result = {"sales": 0, "stock": 0, "stock_movement": 0, "prices": 0}

    try:
        # Clean up sales data
        sales_count_result = await dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_sales WHERE data_date < $1",
            (sales_cutoff,),
            fetchall=False
        )
        sales_count = sales_count_result["count"] if sales_count_result else 0

        if sales_count > 0:
            await dal.execute_raw_query(
                "DELETE FROM fact_sales WHERE data_date < $1", (sales_cutoff,)
            )
            result["sales"] = sales_count
            logger.info(
                f"Deleted {sales_count} sales records older than {sales_cutoff}"
            )

        # Clean up stock movement data
        changes_count_result = await dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_stock_movement WHERE data_date < $1",
            (stock_cutoff,),
            fetchall=False
        )
        changes_count = changes_count_result["count"] if changes_count_result else 0

        if changes_count > 0:
            await dal.execute_raw_query(
                "DELETE FROM fact_stock_movement WHERE data_date < $1",
                (stock_cutoff,),
            )
            result["stock_movement"] = changes_count
            logger.info(
                f"Deleted {changes_count} stock movement records older than {stock_cutoff}"
            )

        # Clean up price data
        # Code removed due to fact_prices table made time-agnostic

        return result

    except Exception as e:
        logger.error(f"Error cleaning up historical data: {str(e)}")
        return result


async def cleanup_old_models(
        models_to_keep: int | None = None,
        inactive_days_to_keep: int | None = None,
        dal: DataAccessLayer = None,
    ) -> list[str]:
    """
    Clean up old model records and files based on retention policy.

    For each active parameter set, keeps the top N models ranked by validation metric.
    For inactive models, removes those older than the specified period.

    Args:
        models_to_keep: Number of models to keep per parameter set.
                        If None, uses the value from settings.
        inactive_days_to_keep: Number of days to keep inactive models.
                               If None, uses the value from settings.
        dal: Optional DataAccessLayer instance. If None, a new one is created.

    Returns:
        List of model IDs that were deleted
    """
    if models_to_keep is None:
        models_to_keep = get_settings().data_retention.models_to_keep
    if inactive_days_to_keep is None:
        inactive_days_to_keep = (
            get_settings().data_retention.inactive_model_retention_days
        )

    if dal is None:
        pool = get_db_pool()
        dal = DataAccessLayer(user_context=UserContext(roles=[UserRoles.SYSTEM]), pool=pool)

    deleted_model_ids = []

    try:
        # Get all configs to check which ones are still referenced
        configs = await dal.get_configs()
        active_config_ids = [c["config_id"] for c in configs if c["is_active"]]

        # 2. For each active config set, keep only top N models by metric
        settings = get_settings()
        default_metric = settings.default_metric
        higher_is_better = settings.default_metric_higher_is_better

        order_direction = "DESC" if higher_is_better else "ASC"

        if default_metric not in ALLOWED_METRICS:
            raise ValueError(
                f"Invalid default_metric: {default_metric}. Allowed metrics are: {ALLOWED_METRICS}"
            )

        for config_id in active_config_ids:
            # Use strict whitelist validation: default_metric is already validated against ALLOWED_METRICS above
            # This prevents SQL injection by ensuring only pre-validated metric names are used
            # PostgreSQL JSONB operator ->> requires literal string, so we escape single quotes defensively
            safe_metric = default_metric.replace("'", "''")
            query = f"""
                SELECT m.model_id, m.model_path
                FROM models m
                JOIN training_results tr ON m.model_id = tr.model_id
                WHERE tr.config_id = $1
                ORDER BY (tr.metrics->>'{safe_metric}')::numeric {order_direction}
            """
            models = await dal.execute_raw_query(query, (config_id,), fetchall=True)

            if len(models) > models_to_keep:
                models_to_delete = models[models_to_keep:]

                for model in models_to_delete:
                    model_id = model["model_id"]

                    prediction_count_result = await dal.execute_raw_query(
                        "SELECT COUNT(*) as count FROM fact_predictions WHERE model_id = $1",
                        (model_id,),
                        fetchall=False
                    )
                    prediction_count = prediction_count_result["count"] if prediction_count_result else 0

                    if prediction_count == 0:
                        await dal.delete_model_record_and_file(model_id)
                        deleted_model_ids.append(model_id)
                        logger.info(
                            f"Deleted model {model_id} (excess model for config set {config_id})"
                        )

        # 3. Clean up inactive models older than retention period
        retention_date = datetime.now() - timedelta(days=inactive_days_to_keep)
        cutoff_datetime = retention_date  # Use datetime object for asyncpg

        inactive_models = await dal.execute_raw_query(
            """
            SELECT model_id, model_path
            FROM models
            WHERE is_active = FALSE AND created_at < $1
        """,
            (cutoff_datetime,),
            fetchall=True,
        )

        for model in inactive_models:
            model_id = model["model_id"]
            prediction_count_result = await dal.execute_raw_query(
                "SELECT COUNT(*) as count FROM fact_predictions WHERE model_id = $1",
                (model_id,),
                fetchall=False
            )
            prediction_count = prediction_count_result["count"] if prediction_count_result else 0

            if prediction_count == 0:
                await dal.delete_model_record_and_file(model_id)
                deleted_model_ids.append(model_id)
                logger.info(
                    f"Deleted inactive model {model_id} (older than {inactive_days_to_keep} days)"
                )

        return deleted_model_ids

    except Exception as e:
        logger.error(f"Error cleaning up old models: {str(e)}")
        return []


async def run_cleanup_job(dal: DataAccessLayer = None) -> None:
    """
    Runs all cleanup routines: predictions, models, and historical data.
    
    NOTE: Each cleanup runs independently - if one fails, others still execute.
    This is intentional for resilience, but consider if atomic cleanup is needed.
    """
    if dal is None:
        pool = get_db_pool()
        dal = DataAccessLayer(user_context=UserContext(roles=[UserRoles.SYSTEM]), pool=pool)
    try:
        deleted_predictions = await cleanup_old_predictions(dal=dal)
        logger.info(f"Deleted {deleted_predictions} old predictions.")
    except Exception as e:
        logger.error(f"Error in prediction cleanup: {e}")
    try:
        deleted_models = await cleanup_old_models(dal=dal)
        logger.info(f"Deleted models: {deleted_models}")
    except Exception as e:
        logger.error(f"Error in model cleanup: {e}")
    try:
        deleted_historical = await cleanup_old_historical_data(dal=dal)
        logger.info(f"Deleted historical data: {deleted_historical}")
    except Exception as e:
        logger.error(f"Error in historical data cleanup: {e}")
