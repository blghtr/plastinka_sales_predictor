"""
API endpoints for working with models and parameter sets.
"""

import json as jsonlib
import logging
import os

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)

from deployment.app.config import get_settings
from deployment.app.db.database import (
    create_job,
    create_model_record,
    delete_configs_by_ids,
    delete_models_by_ids,
    get_active_config,
    get_active_model,
    get_all_models,
    get_best_config_by_metric,
    get_best_model_by_metric,
    get_configs,
    get_job,
    get_recent_models,
    set_config_active,
    set_model_active,
)
from deployment.app.models.api_models import (
    ConfigCreateRequest,
    ConfigResponse,
    DeleteIdsRequest,
    DeleteResponse,
    ModelResponse,
)
from deployment.app.services.auth import get_current_api_key_validated

router = APIRouter(
    prefix="/api/v1/models-configs",
    tags=["models-configs"],
    responses={
        200: {"description": "Success"},
        404: {"description": "Not Found"},
        500: {"description": "Server Error"},
    },
)

logger = logging.getLogger("plastinka.api.model_params")


# Config Endpoints
@router.get("/configs/active", response_model=ConfigResponse)
async def get_active_config_endpoint(
    api_key: bool = Depends(get_current_api_key_validated),
):
    """Get the currently active config."""
    active_config = get_active_config()
    if not active_config:
        raise HTTPException(status_code=404, detail="No active config found")
    return active_config


@router.post("/configs/{config_id}/set-active")
async def activate_config(
    config_id: str, api_key: bool = Depends(get_current_api_key_validated)
):
    """Set a config as active."""
    if set_config_active(config_id):
        return {"success": True, "message": f"Config {config_id} set as active"}
    raise HTTPException(status_code=404, detail=f"Config {config_id} not found")


@router.get("/configs/best", response_model=ConfigResponse)
async def get_best_config(
    metric_name: str = Query(None, description="Metric name to use for comparison"),
    higher_is_better: bool = Query(
        True, description="Whether higher values are better"
    ),
    api_key: bool = Depends(get_current_api_key_validated),
):
    """
    Get the best config based on a metric.
    Uses the default metric from settings if none provided.
    """
    # Use default metric from settings if none provided
    if not metric_name:
        settings = get_settings()
        metric_name = settings.default_metric
        higher_is_better = settings.default_metric_higher_is_better

    best_config = get_best_config_by_metric(metric_name, higher_is_better)
    if not best_config:
        raise HTTPException(
            status_code=404, detail=f"No configs found with metric '{metric_name}'"
        )
    return best_config


# Endpoint to list and delete configs


@router.get("/configs", response_model=list[ConfigResponse])
async def get_configs_endpoint(
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of configs to return"
    ),
    api_key: bool = Depends(get_current_api_key_validated),
):
    """
    Get a list of all configs.
    """
    configs_list = get_configs(limit=limit)
    if not configs_list:
        return []

    return configs_list


@router.post("/configs/delete", response_model=DeleteResponse)
async def delete_configs(
    request: DeleteIdsRequest, api_key: bool = Depends(get_current_api_key_validated)
):
    """
    Delete multiple configs by their IDs.
    Active configs cannot be deleted.
    """
    if not request.ids:
        return DeleteResponse(successful=0, failed=0, errors=["No IDs provided"])

    result = delete_configs_by_ids(request.ids)
    return DeleteResponse(
        successful=result["successful"],
        failed=result["failed"],
        errors=result["errors"],
    )


# Model Endpoints
@router.get("/models/active", response_model=ModelResponse)
async def get_active_model_endpoint(
    api_key: bool = Depends(get_current_api_key_validated),
):
    """Get the currently active model."""
    active_model = get_active_model()
    if not active_model:
        raise HTTPException(status_code=404, detail="No active model found")
    return active_model


@router.post("/models/{model_id}/set-active")
async def activate_model(
    model_id: str, api_key: bool = Depends(get_current_api_key_validated)
):
    """Set a model as active."""
    if set_model_active(model_id):
        return {"success": True, "message": f"Model {model_id} set as active"}
    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")


@router.get("/models/best", response_model=ModelResponse)
async def get_best_model(
    metric_name: str = Query(None, description="Metric name to use for comparison"),
    higher_is_better: bool = Query(
        True, description="Whether higher values are better"
    ),
    api_key: bool = Depends(get_current_api_key_validated),
):
    """
    Get the best model based on a metric.
    Uses the default metric from settings if none provided.
    """
    # Use default metric from settings if none provided
    if not metric_name:
        settings = get_settings()
        metric_name = settings.default_metric
        higher_is_better = settings.default_metric_higher_is_better

    best_model = get_best_model_by_metric(metric_name, higher_is_better)
    if not best_model:
        raise HTTPException(
            status_code=404, detail=f"No models found with metric '{metric_name}'"
        )
    return best_model


@router.get("/models/recent", response_model=list[ModelResponse])
async def get_recent_models_endpoint(
    limit: int = Query(
        5, ge=1, le=100, description="Maximum number of models to return"
    ),
    api_key: bool = Depends(get_current_api_key_validated),
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

        result.append(
            ModelResponse(
                model_id=model_id,
                model_path=model_path,
                is_active=False,  # We don't have this info from get_recent_models
                metadata=metadata,
            )
        )

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


@router.get("/models", response_model=list[ModelResponse])
async def get_all_models_endpoint(
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of models to return"
    ),
    api_key: bool = Depends(get_current_api_key_validated),
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
    request: DeleteIdsRequest, api_key: bool = Depends(get_current_api_key_validated)
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
        errors=result["errors"],
    )


# --- Upload/Create Config Endpoint ---
@router.post("/configs/upload", response_model=ConfigResponse)
async def upload_config(
    request: ConfigCreateRequest = Body(...),
    api_key: bool = Depends(get_current_api_key_validated),
):
    """Upload (create) a new config."""
    from deployment.app.db.database import auto_activate_best_config_if_enabled, create_or_get_config

    try:
        config_id = create_or_get_config(
            request.json_payload, is_active=request.is_active
        )
        
        # Auto-activate best config if enabled in settings (unless user explicitly set this one as active)
        if not request.is_active:
            try:
                activated = auto_activate_best_config_if_enabled()
                if activated:
                    logger.info(f"Auto-activated best config after manual config upload: {config_id}")
            except Exception as e:
                logger.warning(f"Failed to auto-activate best config after manual upload: {e}")
        
        return ConfigResponse(
            config_id=config_id,
            configs=request.json_payload,
            is_active=request.is_active,
        )
    except Exception as e:
        logger.error(f"Failed to upload config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload config: {e}"
        ) from e


# --- Upload/Create Model Endpoint ---
@router.post("/models/upload", response_model=ModelResponse)
async def upload_model(
    model_file: UploadFile = File(..., description="Model onnx file"),
    model_id: str = Form(..., description="Unique model identifier"),
    job_id: str | None = Form(
        None, description="Job ID that produced the model (optional)"
    ),
    is_active: bool = Form(False, description="Set as active after creation"),
    created_at: str = Form(None, description="Creation timestamp (ISO format)"),
    metadata: str = Form(None, description="Model metadata as JSON string"),
    api_key: bool = Depends(get_current_api_key_validated),
):
    from datetime import datetime

    try:
        # --- Работа с job_id ---
        used_job_id = job_id
        if job_id is None:
            # Создаем job типа manual_upload, статус completed, id сгенерируется
            try:
                new_job_id = create_job(
                    job_type="manual_upload",
                    parameters={"uploaded_model_filename": model_file.filename},
                    status="completed",
                )
                used_job_id = new_job_id
                logger.info(f"Created manual_upload job with job_id={used_job_id}")
            except Exception as job_create_exc:
                logger.error(f"Failed to create manual_upload job: {job_create_exc}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create manual_upload job: {job_create_exc}",
                ) from job_create_exc
        else:
            # Проверяем существование job_id
            job = None
            try:
                job = get_job(job_id)
            except Exception as e:
                logger.warning(f"Error checking job existence for job_id={job_id}: {e}")
                job = None
            if not job:
                raise HTTPException(
                    status_code=400,
                    detail=f"Provided job_id '{job_id}' does not exist.",
                )
        # --- Сохраняем файл ---
        file_ext = os.path.splitext(model_file.filename)[1]
        save_path = os.path.join(
            get_settings().model_storage_dir, f"{model_id}{file_ext}"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            content = await model_file.read()
            f.write(content)
        # --- Парсим metadata ---
        meta_dict = None
        if metadata:
            try:
                meta_dict = jsonlib.loads(metadata)
            except Exception as e:
                logger.error(f"Failed to parse metadata JSON: {e}")
                if os.path.exists(save_path):
                    try:
                        os.remove(save_path)
                    except Exception as cleanup_e:
                        logger.error(
                            f"Failed to clean up orphaned model file {save_path}: {cleanup_e}"
                        )
                raise HTTPException(
                    status_code=400, detail="Invalid metadata JSON"
                ) from e
        created_at_val = created_at or datetime.now().isoformat()
        try:
            create_model_record(
                model_id=model_id,
                model_path=save_path,
                job_id=used_job_id,
                created_at=created_at_val,
                metadata=meta_dict,
                is_active=is_active,
            )
            
            # Auto-activate best model if enabled in settings (unless user explicitly set this one as active)
            if not is_active:
                try:
                    from deployment.app.db.database import auto_activate_best_model_if_enabled
                    activated = auto_activate_best_model_if_enabled()
                    if activated:
                        logger.info(f"Auto-activated best model after manual model upload: {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to auto-activate best model after manual upload: {e}")
                    
        except Exception as db_exc:
            logger.error(
                f"Failed to create model record for model_id={model_id}: {db_exc}"
            )
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception as cleanup_e:
                    logger.error(
                        f"Failed to clean up orphaned model file {save_path}: {cleanup_e}"
                    )
            raise HTTPException(
                status_code=500, detail=f"Failed to upload model: {db_exc}"
            ) from db_exc
        return ModelResponse(
            model_id=model_id,
            model_path=save_path,
            is_active=is_active,
            metadata=meta_dict,
            created_at=created_at_val,
            job_id=used_job_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        try:
            if "save_path" in locals() and save_path and os.path.exists(save_path):
                os.remove(save_path)
        except Exception as cleanup_e:
            logger.error(
                f"Failed to clean up orphaned model file {save_path}: {cleanup_e}"
            )
        raise HTTPException(
            status_code=500, detail=f"Failed to upload model: {e}"
        ) from e
