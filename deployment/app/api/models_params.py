"""
API endpoints for working with models and parameter sets.
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List, Optional
import logging
from pydantic import BaseModel, Field, ConfigDict

from deployment.app.db.database import (
    get_active_parameter_set,
    set_parameter_set_active,
    get_best_parameter_set_by_metric,
    get_active_model,
    set_model_active,
    get_best_model_by_metric,
    get_recent_models,
    get_parameter_sets,
    delete_parameter_sets_by_ids,
    get_all_models,
    delete_models_by_ids
)
from deployment.app.config import settings
from deployment.app.services.auth import get_current_api_key_validated

router = APIRouter(
    prefix="/api/v1/model-params",
    tags=["model-params"],
    responses={
        200: {"description": "Success"},
        404: {"description": "Not Found"},
        500: {"description": "Server Error"}
    }
)

logger = logging.getLogger("plastinka.api.model_params")


# Models for API responses
class ParameterSetResponse(BaseModel):
    """Parameter set information"""
    parameter_set_id: str
    parameters: Dict[str, Any]
    is_active: bool
    
    model_config = ConfigDict(from_attributes=True)


class ModelResponse(BaseModel):
    """Model information"""
    model_id: str
    model_path: str
    is_active: bool
    metadata: Dict[str, Any] = {}
    metrics: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


# После класса ModelResponse добавим новые модели для запросов
class DeleteIdsRequest(BaseModel):
    """Request model for deleting items by IDs"""
    ids: List[str] = Field(..., description="List of IDs to delete")
    
    model_config = ConfigDict(from_attributes=True)


class DeleteResponse(BaseModel):
    """Response model for delete operations"""
    successful: int = Field(..., description="Number of items successfully deleted")
    failed: int = Field(..., description="Number of items that failed to delete")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    
    model_config = ConfigDict(from_attributes=True)


# Parameter Set Endpoints
@router.get("/parameter-sets/active", response_model=ParameterSetResponse)
async def get_active_parameter_set_endpoint(api_key: bool = Depends(get_current_api_key_validated)):
    """Get the currently active parameter set."""
    active_params = get_active_parameter_set()
    if not active_params:
        raise HTTPException(status_code=404, detail="No active parameter set found")
    return active_params


@router.post("/parameter-sets/{parameter_set_id}/set-active")
async def activate_parameter_set(parameter_set_id: str, api_key: bool = Depends(get_current_api_key_validated)):
    """Set a parameter set as active."""
    if set_parameter_set_active(parameter_set_id):
        return {"success": True, "message": f"Parameter set {parameter_set_id} set as active"}
    raise HTTPException(status_code=404, detail=f"Parameter set {parameter_set_id} not found")


@router.get("/parameter-sets/best", response_model=ParameterSetResponse)
async def get_best_parameter_set(
    metric_name: str = Query(None, description="Metric name to use for comparison"),
    higher_is_better: bool = Query(True, description="Whether higher values are better"),
    api_key: bool = Depends(get_current_api_key_validated)
):
    """
    Get the best parameter set based on a metric.
    Uses the default metric from settings if none provided.
    """
    # Use default metric from settings if none provided
    if not metric_name:
        metric_name = settings.default_metric
        higher_is_better = settings.default_metric_higher_is_better
        
    best_params = get_best_parameter_set_by_metric(metric_name, higher_is_better)
    if not best_params:
        raise HTTPException(
            status_code=404, 
            detail=f"No parameter sets found with metric '{metric_name}'"
        )
    return best_params


# После @router.get("/parameter-sets/best", response_model=ParameterSetResponse) 
# добавляем эндпоинты для списка параметров и удаления

@router.get("/parameter-sets", response_model=List[ParameterSetResponse])
async def get_parameter_sets_endpoint(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of parameter sets to return"),
    api_key: bool = Depends(get_current_api_key_validated)
):
    """
    Get a list of all parameter sets.
    """
    params_list = get_parameter_sets(limit=limit)
    if not params_list:
        return []
        
    return params_list


@router.post("/parameter-sets/delete", response_model=DeleteResponse)
async def delete_parameter_sets(
    request: DeleteIdsRequest,
    api_key: bool = Depends(get_current_api_key_validated)
):
    """
    Delete multiple parameter sets by their IDs.
    Active parameter sets cannot be deleted.
    """
    if not request.ids:
        return DeleteResponse(successful=0, failed=0, errors=["No IDs provided"])
        
    result = delete_parameter_sets_by_ids(request.ids)
    return DeleteResponse(
        successful=result["successful"],
        failed=result["failed"],
        errors=result["errors"]
    )


# Model Endpoints
@router.get("/models/active", response_model=ModelResponse)
async def get_active_model_endpoint(api_key: bool = Depends(get_current_api_key_validated)):
    """Get the currently active model."""
    active_model = get_active_model()
    if not active_model:
        raise HTTPException(status_code=404, detail="No active model found")
    return active_model


@router.post("/models/{model_id}/set-active")
async def activate_model(model_id: str, api_key: bool = Depends(get_current_api_key_validated)):
    """Set a model as active."""
    if set_model_active(model_id):
        return {"success": True, "message": f"Model {model_id} set as active"}
    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")


@router.get("/models/best", response_model=ModelResponse)
async def get_best_model(
    metric_name: str = Query(None, description="Metric name to use for comparison"),
    higher_is_better: bool = Query(True, description="Whether higher values are better"),
    api_key: bool = Depends(get_current_api_key_validated)
):
    """
    Get the best model based on a metric.
    Uses the default metric from settings if none provided.
    """
    # Use default metric from settings if none provided
    if not metric_name:
        metric_name = settings.default_metric
        higher_is_better = settings.default_metric_higher_is_better
        
    best_model = get_best_model_by_metric(metric_name, higher_is_better)
    if not best_model:
        raise HTTPException(
            status_code=404, 
            detail=f"No models found with metric '{metric_name}'"
        )
    return best_model


@router.get("/models/recent", response_model=List[ModelResponse])
async def get_recent_models_endpoint(
    limit: int = Query(5, ge=1, le=100, description="Maximum number of models to return"),
    api_key: bool = Depends(get_current_api_key_validated)
):
    """Get the most recent models, ordered by creation date."""
    models = get_recent_models(limit)
    if not models:
        return []
        
    # Convert to response format
    result = []
    for model in models:
        # model is a tuple of (model_id, job_id, model_path, created_at, metadata)
        # Destructure tuple values
        model_id, job_id, model_path, created_at, metadata_json = model
        import json
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        result.append(ModelResponse(
            model_id=model_id,
            model_path=model_path,
            is_active=False,  # We don't have this info from get_recent_models
            metadata=metadata
        ))
    
    # Try to mark the active model
    active_model = get_active_model()
    if active_model:
        for model in result:
            if model.model_id == active_model["model_id"]:
                model.is_active = True
                break
                
    return result


# После @router.get("/models/recent", response_model=List[ModelResponse])
# добавляем эндпоинты для списка моделей и удаления

@router.get("/models", response_model=List[ModelResponse])
async def get_all_models_endpoint(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of models to return"),
    api_key: bool = Depends(get_current_api_key_validated)
):
    """
    Get a list of all models.
    """
    models_list = get_all_models(limit=limit)
    if not models_list:
        return []
        
    return models_list


@router.post("/models/delete", response_model=DeleteResponse)
async def delete_models(
    request: DeleteIdsRequest,
    api_key: bool = Depends(get_current_api_key_validated)
):
    """
    Delete multiple models by their IDs and their associated files.
    Active models cannot be deleted.
    """
    if not request.ids:
        return DeleteResponse(successful=0, failed=0, errors=["No IDs provided"])
        
    result = delete_models_by_ids(request.ids)
    return DeleteResponse(
        successful=result["successful"],
        failed=result["failed"],
        errors=result["errors"]
    ) 