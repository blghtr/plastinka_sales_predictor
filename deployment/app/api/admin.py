"""
Administrative API endpoints for system management tasks.
"""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Query

from ..db.data_access_layer import DataAccessLayer
from ..db.data_retention import (
    cleanup_old_historical_data,
    cleanup_old_models,
    cleanup_old_predictions,
    run_cleanup_job,
)
from ..dependencies import get_dal_for_admin_user
from ..services.auth import get_admin_token_validated

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post(
    "/data-retention/cleanup",
    response_model=dict[str, Any],
    summary="Trigger a full data retention cleanup job.",
    dependencies=[Depends(get_admin_token_validated)],
)
async def trigger_cleanup_job(
    background_tasks: BackgroundTasks,
    admin_user: dict[str, Any] = Depends(get_admin_token_validated),
    dal: DataAccessLayer = Depends(get_dal_for_admin_user),
):
    """
    Starts a background task to perform all data retention and cleanup operations,
    such as removing old predictions, historical data, and models, based on the
    configured retention policies. Requires admin authentication.
    """
    background_tasks.add_task(run_cleanup_job, dal)
    return {"message": "Data retention cleanup job started"}



@router.post("/data-retention/clean-predictions", response_model=dict[str, Any],
             summary="Clean up old prediction results.",
             dependencies=[Depends(get_admin_token_validated)])
async def clean_predictions(
    days_to_keep: int = Query(None, description="The number of days of prediction data to retain. Uses settings if not provided."),
    admin_user: dict[str, Any] = Depends(get_admin_token_validated),
    dal: DataAccessLayer = Depends(get_dal_for_admin_user),
):
    """
    Deletes prediction records older than the specified number of days.
    If no period is specified, it uses the default from settings.
    Requires admin authentication.
    """
    count = cleanup_old_predictions(days_to_keep, dal=dal)
    return {"status": "ok", "records_removed": count, "days_kept": days_to_keep}


@router.post("/data-retention/clean-historical", response_model=dict[str, Any],
             summary="Clean up old historical data.")
async def clean_historical_data(
    sales_days_to_keep: int = Query(None, description="The number of days of sales and price data to retain."),
    stock_days_to_keep: int = Query(None, description="The number of days of stock data to retain."),
    admin_user: dict[str, Any] = Depends(get_admin_token_validated),
    dal: DataAccessLayer = Depends(get_dal_for_admin_user),
):
    """
    Deletes historical sales, stock, and price data older than the specified periods.
    Uses settings if periods are not provided. Requires admin authentication.
    """
    result = cleanup_old_historical_data(sales_days_to_keep, stock_days_to_keep, dal=dal)
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


@router.post("/data-retention/clean-models", response_model=dict[str, Any],
             summary="Clean up old and inactive models.",
             dependencies=[Depends(get_admin_token_validated)])
async def clean_models(
    models_to_keep: int = Query(None, description="The number of models to keep for each parameter set."),
    inactive_days_to_keep: int = Query(None, description="The number of days to keep models that are not active."),
    admin_user: dict[str, Any] = Depends(get_admin_token_validated),
    dal: DataAccessLayer = Depends(get_dal_for_admin_user),
):
    """
    Deletes old model files and records based on the retention policy, which includes
    keeping a certain number of models per parameter set and removing models that have
    been inactive for a specified period. Requires admin authentication.
    """
    removed_models = cleanup_old_models(models_to_keep, inactive_days_to_keep, dal=dal)
    return {
        "status": "ok",
        "models_removed": removed_models,
        "models_removed_count": len(removed_models),
        "models_kept": models_to_keep,
        "inactive_days_kept": inactive_days_to_keep,
    }
