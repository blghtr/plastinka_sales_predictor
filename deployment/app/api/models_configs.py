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
    Path,
    Query,
    UploadFile,
)
from fastapi import status

from deployment.app.config import get_settings
from deployment.app.db.database import (
    auto_activate_best_config_if_enabled,
    auto_activate_best_model_if_enabled,
)
from deployment.app.models.api_models import (
    ConfigCreateRequest,
    ConfigResponse,
    DeleteIdsRequest,
    DeleteResponse,
    ModelResponse,
    ErrorDetailResponse
)
from deployment.app.services.auth import get_current_api_key_validated
from deployment.app.dependencies import DataAccessLayer, get_dal_for_general_user
from deployment.app.utils.error_handling import ErrorDetail

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
@router.get("/configs/active", response_model=ConfigResponse,
             summary="Get the currently active hyperparameter configuration.")
async def get_active_config_endpoint(
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Retrieves the full details of the parameter set that is currently marked as active.
    This configuration is used by default for new training jobs.
    Returns a 404 error if no configuration is currently active.
    """
    active_config = dal.get_active_config()
    if not active_config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ErrorDetailResponse(
            message="No active config found",
            code="no_active_config",
            status_code=status.HTTP_404_NOT_FOUND,
        ).model_dump())
    return active_config


@router.post("/configs/{config_id}/set-active",
             summary="Set a specific hyperparameter configuration as active.")
async def activate_config(
    config_id: str = Path(..., description="The unique identifier (ID) of the configuration to activate."),
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Marks a chosen parameter set as the active one for future training jobs.
    This deactivates any previously active configuration.
    """
    if dal.set_config_active(config_id):
        return {"success": True, "message": f"Config {config_id} set as active"}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ErrorDetailResponse(
        message=f"Config {config_id} not found",
        code="config_not_found",
        status_code=status.HTTP_404_NOT_FOUND,
    ).model_dump())


@router.get("/configs/best", response_model=ConfigResponse,
             summary="Get the best hyperparameter configuration based on a metric.")
async def get_best_config(
    metric_name: str = Query(
        None,
        description="The name of the metric to use for comparison (e.g., \"val_MIWS_MIC_Ratio\"). If omitted, the default metric from settings will be used."
    ),
    higher_is_better: bool = Query(
        True, 
        description="A boolean flag indicating the desired direction of the metric. Set to `true` if higher values are better, `false` otherwise."
    ),
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Finds and returns the configuration that has the best recorded performance
    for a specific metric. If no metric is specified, it uses the default
    metric defined in the application settings.
    """
    # Use default metric from settings if none provided
    if not metric_name:
        settings = get_settings()
        metric_name = settings.default_metric
        higher_is_better = settings.default_metric_higher_is_better

    best_config = dal.get_best_config_by_metric(metric_name, higher_is_better)
    if not best_config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ErrorDetailResponse(
            message=f"No configs found with metric '{metric_name}'",
            code="no_configs_found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"metric_name": metric_name}
        ).model_dump())
    return best_config


# Endpoint to list and delete configs


@router.get("/configs", response_model=list[ConfigResponse],
             summary="Get a list of all available hyperparameter configurations.")
async def get_configs_endpoint(
    limit: int = Query(
        100, ge=1, le=1000, description="The maximum number of configurations to return in the list."
    ),
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Retrieves a paginated list of all saved hyperparameter configurations.
    """
    configs_list = dal.get_configs(limit=limit)
    if not configs_list:
        return []

    return configs_list


@router.post("/configs/delete", response_model=DeleteResponse,
             summary="Delete one or more hyperparameter configurations.")
async def delete_configs(
    request: DeleteIdsRequest = Body(..., description="A JSON object containing a list of configuration IDs to delete."),
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Deletes the specified configurations by their IDs. The active configuration
    cannot be deleted.
    """
    if not request.ids:
        # Modified to return HTTPException directly, as DeleteResponse might not be suitable for empty request body validation error
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ErrorDetailResponse(
            message="No IDs provided for deletion.",
            code="no_ids_provided",
            status_code=status.HTTP_400_BAD_REQUEST,
        ).model_dump())

    result = dal.delete_configs_by_ids(request.ids)
    return DeleteResponse(
        successful=result["successful"],
        failed=result["failed"],
        errors=result["errors"],
    )


# Model Endpoints
@router.get("/models/active", response_model=ModelResponse, summary="Get the currently active model.")
async def get_active_model_endpoint(
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Retrieves the details of the model currently marked as active.
    This model is used for generating predictions.
    Returns a 404 error if no model is active.
    """
    active_model = dal.get_active_model()
    if not active_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ErrorDetailResponse(
            message="No active model found",
            code="no_active_model",
            status_code=status.HTTP_404_NOT_FOUND,
        ).model_dump())
    return active_model


@router.post("/models/{model_id}/set-active", summary="Set a specific model as active.")
async def activate_model(
    model_id: str = Path(..., description="The unique identifier (ID) of the model to activate."),
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Marks a chosen model as the active one for generating predictions.
    This deactivates any previously active model.
    """
    if dal.set_model_active(model_id, deactivate_others=True):
        return {"success": True, "message": f"Model {model_id} set as active"}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ErrorDetailResponse(
        message=f"Model {model_id} not found",
        code="model_not_found",
        status_code=status.HTTP_404_NOT_FOUND,
    ).model_dump())


@router.get("/models/best", response_model=ModelResponse, summary="Get the best model based on a performance metric.")
async def get_best_model(
    metric_name: str = Query(
        None,
        description="The name of the metric to use for comparison (e.g., \"val_MIWS_MIC_Ratio\"). If omitted, the default metric from settings will be used."
    ),
    higher_is_better: bool = Query(
        True, 
        description="A boolean flag indicating the desired direction of the metric. Set to `true` if higher values are better, `false` otherwise."
    ),
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Finds and returns the model that has the best recorded performance
    for a specific metric from its training results. If no metric is specified,
    it uses the default metric from the application settings.
    """
    # Use default metric from settings if none provided
    if not metric_name:
        settings = get_settings()
        metric_name = settings.default_metric
        higher_is_better = settings.default_metric_higher_is_better

    best_model = dal.get_best_model_by_metric(metric_name, higher_is_better)
    if not best_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ErrorDetailResponse(
            message=f"No models found with metric '{metric_name}'",
            code="no_models_found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"metric_name": metric_name}
        ).model_dump())
    return best_model


@router.get("/models/recent", response_model=list[ModelResponse], summary="Get a list of the most recently created models.")
async def get_recent_models_endpoint(
    limit: int = Query(
        5, ge=1, le=100, description="The maximum number of recent models to return."
    ),
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """Retrieves a list of the most recent models, ordered by their creation date."""
    models = dal.get_recent_models(limit)
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
    active_model = dal.get_active_model()
    if active_model:
        for model in result:
            if model.model_id == active_model["model_id"]:
                model.is_active = True
                break

    return result


# После @router.get("/models/recent", response_model=List[ModelResponse])
# добавляем эндпоинты для списка моделей и удаления


@router.get("/models", response_model=list[ModelResponse], summary="Get a list of all available models.")
async def get_all_models_endpoint(
    limit: int = Query(
        100, ge=1, le=1000, description="The maximum number of models to return in the list."
    ),
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Retrieves a paginated list of all saved models in the system.
    """
    models_list = dal.get_all_models(limit=limit)
    if not models_list:
        return []

    return models_list


@router.post("/models/delete", response_model=DeleteResponse, summary="Delete one or more models.")
async def delete_models(
    request: DeleteIdsRequest = Body(..., description="A JSON object containing a list of model IDs to delete."),
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Deletes the specified models by their IDs, including their associated model files from storage.
    The active model cannot be deleted.
    """
    if not request.ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ErrorDetailResponse(
            message="No IDs provided for deletion.",
            code="no_ids_provided",
            status_code=status.HTTP_400_BAD_REQUEST,
        ).model_dump())

    result = dal.delete_models_by_ids(request.ids)
    return DeleteResponse(
        successful=result.get("deleted_count", 0),
        failed=len(result.get("failed_deletions", [])),
        errors=[{"id": mid, "reason": "Failed to delete file"} for mid in result.get("failed_deletions", [])],
    )


# --- Upload/Create Config Endpoint ---
@router.post("/configs/upload", response_model=ConfigResponse, summary="Create a new hyperparameter configuration.")
async def upload_config(
    request: ConfigCreateRequest = Body(
        ..., 
        description="A JSON object containing the configuration payload (`json_payload`) and a boolean flag (`is_active`) to set it as active."
    ),
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Uploads a new set of hyperparameters and saves it as a configuration.
    Optionally, it can be set as the active configuration upon creation.
    """
    try:
        config_id = dal.create_or_get_config(
            request.json_payload, is_active=request.is_active, source="manual_upload"
        )
                
        return ConfigResponse(
            config_id=config_id,
            config=request.json_payload,
            is_active=request.is_active,
        )
    except Exception as e:
        logger.error(f"Failed to upload config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetailResponse(
                message=f"Failed to upload config: {e}",
                code="config_upload_failed",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                details={"error": str(e)}
            ).model_dump()
        ) from e


# --- Upload/Create Model Endpoint ---
@router.post("/models/upload", response_model=ModelResponse, summary="Upload a new model file.")
async def upload_model(
    model_file: UploadFile = File(..., description="The model file to be uploaded (e.g., `model.onnx`)."),
    model_id: str = Form(..., description="A unique identifier for the new model."),
    job_id: str | None = Form(None, description="The optional ID of the training job that produced this model."),
    is_active: bool = Form(False, description="If `true`, sets the model as active immediately after upload."),
    created_at: str = Form(None, description="An optional ISO format timestamp for when the model was created. Defaults to now."),
    metadata: str = Form(None, description="An optional JSON string containing metadata about the model."),
    api_key: bool = Depends(get_current_api_key_validated),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Uploads a model file (e.g., in ONNX format) and creates a corresponding model record in the database.
    Allows associating the model with a job, setting it as active, and embedding metadata.
    """
    from datetime import datetime

    try:
        # --- Работа с job_id ---
        used_job_id = job_id
        if job_id is None:
            # Создаем job типа manual_upload, статус completed, id сгенерируется
            try:
                new_job_id = dal.create_job(
                    job_type="manual_upload",
                    parameters={"uploaded_model_filename": model_file.filename},
                    status="completed",
                )
                used_job_id = new_job_id
                logger.info(f"Created manual_upload job with job_id={used_job_id}")
            except Exception as job_create_exc:
                logger.error(f"Failed to create manual_upload job: {job_create_exc}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=ErrorDetailResponse(
                        message=f"Failed to create manual_upload job: {job_create_exc}",
                        code="manual_job_creation_failed",
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        details={"error": str(job_create_exc)}
                    ).model_dump()
                ) from job_create_exc
        else:
            # Проверяем существование job_id
            job = None
            try:
                job = dal.get_job(job_id)
            except Exception as e:
                logger.warning(f"Error checking job existence for job_id={{job_id}}: {e}")
                job = None
            if not job:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ErrorDetailResponse(
                        message=f"Provided job_id '{job_id}' does not exist.",
                        code="job_id_not_found",
                        status_code=status.HTTP_400_BAD_REQUEST,
                        details={"job_id": job_id}
                    ).model_dump()
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
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ErrorDetailResponse(
                        message="Invalid metadata JSON",
                        code="invalid_metadata_json",
                        status_code=status.HTTP_400_BAD_REQUEST,
                        details={"error": str(e)}
                    ).model_dump()
                ) from e
        created_at_val = created_at or datetime.now().isoformat()
        try:
            dal.create_model_record(
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
                    activated = dal.auto_activate_best_model_if_enabled()
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
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorDetailResponse(
                    message=f"Failed to upload model: {db_exc}",
                    code="model_upload_failed",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    details={"error": str(db_exc)}
                ).model_dump()
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetailResponse(
                message=f"Failed to upload model: {e}",
                code="model_upload_failed",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                details={"error": str(e)}
            ).model_dump()
        ) from e
