"""
Administrative API endpoints for system management tasks.
"""
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status
from typing import Dict, Any, List, Callable
import sqlite3

from ..db.data_retention import run_cleanup_job as default_run_cleanup_job
from ..db.data_retention import cleanup_old_predictions as default_cleanup_old_predictions
from ..db.data_retention import cleanup_old_models as default_cleanup_old_models
from ..db.data_retention import cleanup_old_historical_data as default_cleanup_old_historical_data
from ..db.database import get_db_connection as default_get_db_connection
from ..services.auth import get_admin_user

router = APIRouter(prefix="/admin", tags=["admin"])

async def get_run_cleanup_job_dependency() -> Callable[[], None]:
    return default_run_cleanup_job

async def get_cleanup_old_predictions_dependency() -> Callable[[int | None, sqlite3.Connection], int]:
    return default_cleanup_old_predictions

async def get_cleanup_old_historical_data_dependency() -> Callable[[int | None, int | None, sqlite3.Connection], Dict[str, int]]:
    return default_cleanup_old_historical_data

async def get_cleanup_old_models_dependency() -> Callable[[int | None, int | None, sqlite3.Connection], List[str]]:
    return default_cleanup_old_models

async def get_db_connection_dependency() -> sqlite3.Connection:
    return default_get_db_connection()

@router.post("/data-retention/cleanup", response_model=Dict[str, Any])
async def trigger_cleanup_job(
    background_tasks: BackgroundTasks,
    admin_user: Dict[str, Any] = Depends(get_admin_user),
    run_cleanup_job_injected: Callable[[], None] = Depends(get_run_cleanup_job_dependency)
):
    """
    Trigger a full data retention cleanup job to run in the background.
    
    This endpoint requires admin authentication.
    
    Returns:
        Dict with status message
    """
    background_tasks.add_task(run_cleanup_job_injected)
    return {"status": "ok", "message": "Data retention cleanup job started"}

@router.post("/data-retention/clean-predictions", response_model=Dict[str, Any])
async def clean_predictions(
    days_to_keep: int = None,
    admin_user: Dict[str, Any] = Depends(get_admin_user),
    db_conn: sqlite3.Connection = Depends(get_db_connection_dependency),
    cleanup_predictions_injected: Callable[[int | None, sqlite3.Connection], int] = Depends(get_cleanup_old_predictions_dependency)
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
    try:
        count = cleanup_predictions_injected(days_to_keep, conn=db_conn)
        return {
            "status": "ok", 
            "records_removed": count,
            "days_kept": days_to_keep
        }
    finally:
        db_conn.close()

@router.post("/data-retention/clean-historical", response_model=Dict[str, Any])
async def clean_historical_data(
    sales_days_to_keep: int = None,
    stock_days_to_keep: int = None,
    admin_user: Dict[str, Any] = Depends(get_admin_user),
    db_conn: sqlite3.Connection = Depends(get_db_connection_dependency),
    cleanup_historical_injected: Callable[[int | None, int | None, sqlite3.Connection], Dict[str, int]] = Depends(get_cleanup_old_historical_data_dependency)
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
    try:
        result = cleanup_historical_injected(sales_days_to_keep, stock_days_to_keep, conn=db_conn)
        return {
            "status": "ok",
            "records_removed": {
                "sales": result["sales"],
                "stock": result["stock"],
                "stock_changes": result["stock_changes"],
                "prices": result["prices"],
                "total": sum(result.values())
            },
            "sales_days_kept": sales_days_to_keep,
            "stock_days_kept": stock_days_to_keep
        }
    finally:
        db_conn.close()

@router.post("/data-retention/clean-models", response_model=Dict[str, Any])
async def clean_models(
    models_to_keep: int = None,
    inactive_days_to_keep: int = None,
    admin_user: Dict[str, Any] = Depends(get_admin_user),
    db_conn: sqlite3.Connection = Depends(get_db_connection_dependency),
    cleanup_models_injected: Callable[[int | None, int | None, sqlite3.Connection], List[str]] = Depends(get_cleanup_old_models_dependency)
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
    try:
        removed_models = cleanup_models_injected(models_to_keep, inactive_days_to_keep, conn=db_conn)
        return {
            "status": "ok", 
            "models_removed": removed_models,
            "models_removed_count": len(removed_models),
            "models_kept": models_to_keep,
            "inactive_days_kept": inactive_days_to_keep
        }
    finally:
        db_conn.close() 