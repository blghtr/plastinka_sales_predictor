"""
Data retention management for optimizing database storage.

This module provides functions for implementing data retention policies,
including cleanup of historical predictions and management of model files.
"""

import logging
from datetime import datetime, timedelta

from ..config import get_settings
from .data_access_layer import (  # Import DataAccessLayer
    DataAccessLayer,
    UserContext,
    UserRoles,
)
from .database import ALLOWED_METRICS

logger = logging.getLogger(__name__)


def cleanup_old_predictions(days_to_keep: int | None = None, dal: DataAccessLayer = None) -> int:
    """
    Remove prediction records older than the specified retention period.

    Args:
        days_to_keep: Number of days to keep predictions for.
                      If None, uses the value from settings.
        conn: Optional database connection. If None, a new connection is created.

    Returns:
        Number of records removed
    """
    if days_to_keep is None:
        days_to_keep = get_settings().data_retention.prediction_retention_days

    if dal is None:
        dal = DataAccessLayer(user_context=UserContext(roles=[UserRoles.SYSTEM]))

    retention_date = datetime.now() - timedelta(days=days_to_keep)
    cutoff_date_str = retention_date.strftime("%Y-%m-%d")

    try:
        # Count records to be deleted
        count_result = dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_predictions WHERE prediction_month < ?",
            (cutoff_date_str,),
            fetchall=False,
        )
        count = count_result["count"] if count_result else 0
        logger.info(f"Found {count} predictions older than {cutoff_date_str}")

        if count > 0:
            dal.execute_raw_query(
                "DELETE FROM fact_predictions WHERE prediction_month < ?",
                (cutoff_date_str,),
            )
            logger.info(f"Deleted {count} predictions older than {cutoff_date_str}")
        return count

    except Exception as e:
        logger.error(f"Error cleaning up old predictions: {str(e)}")
        return 0


def cleanup_old_historical_data(
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
        conn: Optional database connection. If None, a new connection is created.

    Returns:
        Dictionary with counts of removed records by type
    """
    if sales_days_to_keep is None:
        sales_days_to_keep = get_settings().data_retention.sales_retention_days
    if stock_days_to_keep is None:
        stock_days_to_keep = get_settings().data_retention.stock_retention_days

    if dal is None:
        dal = DataAccessLayer(user_context=UserContext(roles=[UserRoles.SYSTEM]))

    # Calculate cutoff dates
    sales_cutoff = datetime.now() - timedelta(days=sales_days_to_keep)
    stock_cutoff = datetime.now() - timedelta(days=stock_days_to_keep)

    sales_cutoff_str = sales_cutoff.strftime("%Y-%m-%d")
    stock_cutoff_str = stock_cutoff.strftime("%Y-%m-%d")

    result = {"sales": 0, "stock": 0, "stock_movement": 0, "prices": 0}

    try:
        # Clean up sales data
        sales_count_result = dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_sales WHERE data_date < ?",
            (sales_cutoff_str,),
            fetchall=False
        )
        sales_count = sales_count_result["count"] if sales_count_result else 0

        if sales_count > 0:
            dal.execute_raw_query(
                "DELETE FROM fact_sales WHERE data_date < ?", (sales_cutoff_str,)
            )
            result["sales"] = sales_count
            logger.info(
                f"Deleted {sales_count} sales records older than {sales_cutoff_str}"
            )

        # Clean up stock movement data
        changes_count_result = dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_stock_movement WHERE data_date < ?",
            (stock_cutoff_str,),
            fetchall=False
        )
        changes_count = changes_count_result["count"] if changes_count_result else 0

        if changes_count > 0:
            dal.execute_raw_query(
                "DELETE FROM fact_stock_movement WHERE data_date < ?",
                (stock_cutoff_str,),
            )
            result["stock_movement"] = changes_count
            logger.info(
                f"Deleted {changes_count} stock movement records older than {stock_cutoff_str}"
            )

        # Clean up price data
        # Code removed due to fact_prices table made time-agnostic

        return result

    except Exception as e:
        logger.error(f"Error cleaning up historical data: {str(e)}")
        return result


def cleanup_old_models(
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
        conn: Optional database connection. If None, a new connection is created.

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
        dal = DataAccessLayer(user_context=UserContext(roles=[UserRoles.SYSTEM]))

    deleted_model_ids = []

    try:
        # Get all configs to check which ones are still referenced
        configs = dal.get_configs()
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

        json_path = f"'$.{default_metric}'"

        for config_id in active_config_ids:
            query = f"""
                SELECT m.model_id, m.model_path
                FROM models m
                JOIN training_results tr ON m.model_id = tr.model_id
                WHERE tr.config_id = ?
                ORDER BY json_extract(tr.metrics, {json_path}) {order_direction}
            """
            models = dal.execute_raw_query(query, (config_id,), fetchall=True)

            if len(models) > models_to_keep:
                models_to_delete = models[models_to_keep:]

                for model in models_to_delete:
                    model_id = model["model_id"]

                    prediction_count_result = dal.execute_raw_query(
                        "SELECT COUNT(*) as count FROM fact_predictions WHERE model_id = ?",
                        (model_id,),
                        fetchall=False
                    )
                    prediction_count = prediction_count_result["count"] if prediction_count_result else 0

                    if prediction_count == 0:
                        dal.delete_model_record_and_file(model_id)
                        deleted_model_ids.append(model_id)
                        logger.info(
                            f"Deleted model {model_id} (excess model for config set {config_id})"
                        )

        # 3. Clean up inactive models older than retention period
        retention_date = datetime.now() - timedelta(days=inactive_days_to_keep)
        cutoff_date_str = retention_date.strftime("%Y-%m-%d %H:%M:%S")

        inactive_models = dal.execute_raw_query(
            """
            SELECT model_id, model_path
            FROM models
            WHERE is_active = 0 AND created_at < ?
        """,
            (cutoff_date_str,),
            fetchall=True,
        )

        for model in inactive_models:
            model_id = model["model_id"]
            prediction_count_result = dal.execute_raw_query(
                "SELECT COUNT(*) as count FROM fact_predictions WHERE model_id = ?",
                (model_id,),
                fetchall=False
            )
            prediction_count = prediction_count_result["count"] if prediction_count_result else 0

            if prediction_count == 0:
                dal.delete_model_record_and_file(model_id)
                deleted_model_ids.append(model_id)
                logger.info(
                    f"Deleted inactive model {model_id} (older than {inactive_days_to_keep} days)"
                )

        return deleted_model_ids

    except Exception as e:
        logger.error(f"Error cleaning up old models: {str(e)}")
        return []


def run_cleanup_job(dal: DataAccessLayer = None) -> None:
    """Runs all cleanup routines: predictions, models, and historical data."""
    if dal is None:
        dal = DataAccessLayer(user_context=UserContext(roles=[UserRoles.SYSTEM]))
    try:
        deleted_predictions = cleanup_old_predictions(dal=dal)
        print(f"Deleted {deleted_predictions} old predictions.")
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in prediction cleanup: {e}")
    try:
        deleted_models = cleanup_old_models(dal=dal)
        print(f"Deleted models: {deleted_models}")
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in model cleanup: {e}")
    try:
        deleted_historical = cleanup_old_historical_data(dal=dal)
        print(f"Deleted historical data: {deleted_historical}")
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in historical data cleanup: {e}")
