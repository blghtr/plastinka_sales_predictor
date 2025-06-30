"""
Administrative API endpoints for system management tasks.
"""

import sqlite3
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends

from ..db.data_retention import (
    cleanup_old_historical_data,
    cleanup_old_models,
    cleanup_old_predictions,
    run_cleanup_job,
)
from ..db.database import get_db_connection
from ..services.auth import get_admin_user

router = APIRouter(prefix="/admin", tags=["admin"])


# A dependency that yields a DB connection and ensures it's closed
async def get_db_conn_and_close():
    conn = None
    try:
        conn = get_db_connection()
        yield conn
    finally:
        if conn:
            conn.close()


@router.post("/data-retention/cleanup", response_model=dict[str, Any])
async def trigger_cleanup_job(
    background_tasks: BackgroundTasks,
    admin_user: dict[str, Any] = Depends(get_admin_user),
    cleanup_job_func: Callable[[], None] = Depends(
        lambda: run_cleanup_job
    ),  # Use a lambda to make it depend-able
):
    """
    Trigger a full data retention cleanup job to run in the background.

    This endpoint requires admin authentication.

    Returns:
        Dict with status message
    """
    background_tasks.add_task(cleanup_job_func)
    return {"status": "ok", "message": "Data retention cleanup job started"}


@router.post("/data-retention/clean-predictions", response_model=dict[str, Any])
async def clean_predictions(
    days_to_keep: int = None,
    admin_user: dict[str, Any] = Depends(get_admin_user),
    db_conn: sqlite3.Connection = Depends(get_db_conn_and_close),
    cleanup_func: Callable[[int | None, sqlite3.Connection], int] = Depends(
        lambda: cleanup_old_predictions
    ),
):
    """
    Clean up predictions older than the specified retention period.

    This endpoint requires admin authentication.

    Args:
        days_to_keep: Number of days to keep predictions for.
                      If None, uses the value from settings.

    Returns:
        Dict with cleanup results
    """
    count = cleanup_func(days_to_keep, conn=db_conn)
    return {"status": "ok", "records_removed": count, "days_kept": days_to_keep}


@router.post("/data-retention/clean-historical", response_model=dict[str, Any])
async def clean_historical_data(
    sales_days_to_keep: int = None,
    stock_days_to_keep: int = None,
    admin_user: dict[str, Any] = Depends(get_admin_user),
    db_conn: sqlite3.Connection = Depends(get_db_conn_and_close),
    cleanup_func: Callable[
        [int | None, int | None, sqlite3.Connection], dict[str, int]
    ] = Depends(lambda: cleanup_old_historical_data),
):
    """
    Clean up historical sales, stock, price and stock change data older than the specified periods.

    This endpoint requires admin authentication.

    Args:
        sales_days_to_keep: Number of days to keep sales and price data.
                           If None, uses the value from settings.
        stock_days_to_keep: Number of days to keep stock and stock change data.
                           If None, uses the value from settings.

    Returns:
        Dict with cleanup results
    """
    result = cleanup_func(sales_days_to_keep, stock_days_to_keep, conn=db_conn)
    return {
        "status": "ok",
        "records_removed": {
            "sales": result.get("sales", 0),
            "stock": result.get("stock", 0),
            "stock_changes": result.get("stock_changes", 0),
            "prices": result.get("prices", 0),
            "total": sum(result.values()),
        },
        "sales_days_kept": sales_days_to_keep,
        "stock_days_kept": stock_days_to_keep,
    }


@router.post("/data-retention/clean-models", response_model=dict[str, Any])
async def clean_models(
    models_to_keep: int = None,
    inactive_days_to_keep: int = None,
    admin_user: dict[str, Any] = Depends(get_admin_user),
    db_conn: sqlite3.Connection = Depends(get_db_conn_and_close),
    cleanup_func: Callable[
        [int | None, int | None, sqlite3.Connection], list[str]
    ] = Depends(lambda: cleanup_old_models),
):
    """
    Clean up old models based on retention policy.

    This endpoint requires admin authentication.

    Args:
        models_to_keep: Number of models to keep per parameter set.
                       If None, uses the value from settings.
        inactive_days_to_keep: Number of days to keep inactive models.
                              If None, uses the value from settings.

    Returns:
        Dict with cleanup results
    """
    removed_models = cleanup_func(models_to_keep, inactive_days_to_keep, conn=db_conn)
    return {
        "status": "ok",
        "models_removed": removed_models,
        "models_removed_count": len(removed_models),
        "models_kept": models_to_keep,
        "inactive_days_kept": inactive_days_to_keep,
    }
