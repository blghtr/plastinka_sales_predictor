import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.dependencies import get_dal_for_general_user
from deployment.app.models.api_models import (
    TrainingResultResponse,
    TuningResultResponse,
)
from deployment.app.services.auth import get_current_api_key_validated

logger = logging.getLogger("plastinka.api.results")

router = APIRouter(prefix="/api/v1/results", tags=["results"])


@router.get("/training", response_model=List[TrainingResultResponse])
async def get_training_results(
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
    api_key: bool = Depends(get_current_api_key_validated),
):
    """Get a list of recent training results."""
    try:
        results = dal.get_training_results()
        return [TrainingResultResponse(**res) for res in results]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training results: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve training results",
        )


@router.get("/training/{result_id}", response_model=TrainingResultResponse)
async def get_training_result_by_id(
    result_id: str,
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
    api_key: bool = Depends(get_current_api_key_validated),
):
    """Get a single training result by its ID."""
    try:
        result = dal.get_training_results(result_id=result_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Training result not found"
            )
        return TrainingResultResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training result {result_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve training result",
        )


@router.get("/tuning", response_model=List[TuningResultResponse])
async def get_tuning_results(
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
    api_key: bool = Depends(get_current_api_key_validated),
    metric_name: str = Query(None, description="Metric name to use for comparison"),
    higher_is_better: bool = Query(True, description="Whether higher values are better"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of tuning results to return"),
):
    """Get a list of recent tuning results."""
    try:
        results = dal.get_tuning_results(metric_name=metric_name, higher_is_better=higher_is_better, limit=limit, result_id=None)
        return [TuningResultResponse(**res) for res in results]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tuning results: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tuning results",
        )


@router.get("/tuning/{result_id}", response_model=TuningResultResponse)
async def get_tuning_result_by_id(
    result_id: str,
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
    api_key: bool = Depends(get_current_api_key_validated),
):
    """Get a single tuning result by its ID."""
    try:
        result = dal.get_tuning_results(result_id=result_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Tuning result not found"
            )
        return TuningResultResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tuning result {result_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tuning result",
        )
