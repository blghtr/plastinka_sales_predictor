"""
Administrative API endpoints for system management tasks.
"""
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status
from typing import Dict, Any, List

from ..db.data_retention import run_cleanup_job, cleanup_old_predictions, cleanup_old_models, cleanup_old_historical_data
from ..db.database import get_db_connection
from ..services.auth import get_admin_user

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/data-retention/cleanup", response_model=Dict[str, Any])
async def trigger_cleanup_job(
    background_tasks: BackgroundTasks,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """
    Trigger a full data retention cleanup job to run in the background.
    
    This endpoint requires admin authentication.
    
    Returns:
        Dict with status message
    """
    background_tasks.add_task(run_cleanup_job)
    return {"status": "ok", "message": "Data retention cleanup job started"}

@router.post("/data-retention/clean-predictions", response_model=Dict[str, Any])
async def clean_predictions(
    days_to_keep: int = None,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
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
    # Create a dedicated connection for this operation
    conn = get_db_connection()
    try:
        count = cleanup_old_predictions(days_to_keep, conn=conn)
        return {
            "status": "ok", 
            "records_removed": count,
            "days_kept": days_to_keep
        }
    finally:
        conn.close()

@router.post("/data-retention/clean-historical", response_model=Dict[str, Any])
async def clean_historical_data(
    sales_days_to_keep: int = None,
    stock_days_to_keep: int = None,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
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
    # Create a dedicated connection for this operation
    conn = get_db_connection()
    try:
        result = cleanup_old_historical_data(sales_days_to_keep, stock_days_to_keep, conn=conn)
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
        conn.close()

@router.post("/data-retention/clean-models", response_model=Dict[str, Any])
async def clean_models(
    models_to_keep: int = None,
    inactive_days_to_keep: int = None,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
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
    # Create a dedicated connection for this operation
    conn = get_db_connection()
    try:
        removed_models = cleanup_old_models(models_to_keep, inactive_days_to_keep, conn=conn)
        return {
            "status": "ok", 
            "models_removed": removed_models,
            "models_removed_count": len(removed_models),
            "models_kept": models_to_keep,
            "inactive_days_kept": inactive_days_to_keep
        }
    finally:
        conn.close() 