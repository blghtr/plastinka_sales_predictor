import json
import logging
import os
import shutil
from datetime import date, datetime
from pathlib import Path as PathLibPath
from typing import Any

import aiofiles
import io
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi import (
    status as fastapi_status,
)

from deployment.app.config import get_settings
from deployment.app.db.database import (
    DatabaseError,
)
from deployment.app.models.api_models import (
    DataUploadFormParameters,
    DataUploadResponse,
    JobDetails,
    JobResponse,
    JobsList,
    JobStatus,
    JobType,
    ReportParams,
    ReportResponse,
    TrainingParams,
    TrainingResponse,
    TuningParams,
)
from deployment.app.services.auth import get_unified_auth
from deployment.app.services.data_processor import process_data_files
from deployment.app.services.datasphere_service import run_job
from deployment.app.services.report_service import generate_report
from deployment.app.utils.error_handling import AppValidationError, ErrorDetail
from deployment.app.utils.validation import validate_data_file_upload
from deployment.app.utils.validation import (
    validate_sales_file,
    validate_stock_file,
)

from ..db.data_access_layer import DataAccessLayer  # Import for type hinting
from ..dependencies import (  # Import the DAL dependency
    get_dal_for_general_user,
)

logger = logging.getLogger("plastinka.api")

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


async def _save_uploaded_file(uploaded_file: UploadFile, directory: PathLibPath) -> PathLibPath:
    """Helper function to save UploadFile asynchronously."""
    file_path = directory / uploaded_file.filename
    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await uploaded_file.read(1024 * 1024):  # Read in chunks
                await out_file.write(content)
        await uploaded_file.seek(
            0
        )  # Reset pointer if needed elsewhere (though likely not)
    except Exception as e:
        logger.error(
            f"Failed to save file {uploaded_file.filename} to {directory}: {e}",
            exc_info=True,
        )
        # Optionally remove partially saved file
        if file_path.exists():
            os.remove(file_path)
        raise  # Re-raise the exception to be caught by the main handler
    return file_path


def get_next_month(dataset_end_date) -> date:
    """
    Возвращает первый день месяца, следующего за dataset_end_date.

    Args:
        dataset_end_date (str | date | datetime): Дата окончания датасета (строка в ISO-формате или объект date/datetime)

    Returns:
        date: Первый день следующего месяца

    Пример:
        >>> get_next_month("2024-03-15")
        datetime.date(2024, 4, 1)
        >>> get_next_month(date(2024, 12, 31))
        datetime.date(2025, 1, 1)
    """
    if isinstance(dataset_end_date, str):
        dt = datetime.fromisoformat(dataset_end_date.replace("Z", "+00:00"))
    elif isinstance(dataset_end_date, datetime):
        dt = dataset_end_date
    elif isinstance(dataset_end_date, date):
        dt = datetime.combine(dataset_end_date, datetime.min.time())
    else:
        raise ValueError("dataset_end_date must be str, date, or datetime")
    # Переход на следующий месяц
    if dt.month == 12:
        return date(dt.year + 1, 1, 1)
    else:
        return date(dt.year, dt.month + 1, 1)


@router.post("/data-upload", response_model=DataUploadResponse,
             summary="Submit a job to upload and process sales and stock data.")
async def create_data_upload_job(
    request: Request,
    background_tasks: BackgroundTasks,
    stock_file: UploadFile = File(..., description="An Excel or CSV file containing stock data."),
    sales_files: list[UploadFile] = File(
        ..., description="One or more Excel or CSV files containing sales data."
    ),
    params: DataUploadFormParameters = Depends(DataUploadFormParameters.as_form), # Changed from cutoff_date: str = Form(...) to params: DataUploadFormParameters = Depends(...)
    x_api_key_valid: dict[str, Any] = Depends(get_unified_auth),
    dal: DataAccessLayer = Depends(get_dal_for_general_user), # Inject DAL
):
    """
    Accepts stock and sales data files, validates them, and queues a background job
    to process the data and store it in the database.
    """
    job_id = None
    temp_job_dir = None
    
    # Prepare parameters for lock hashing and job creation
    prospective_params = {
        "stock_file": stock_file.filename,
        "sales_files": [f.filename for f in sales_files],
        "overwrite": params.overwrite,
    }

    # Enforce refractory: job_type + parameter hash (before any processing)
    acquired, retry_after = dal.try_acquire_job_submission_lock(
        JobType.DATA_UPLOAD.value, prospective_params
    )
    if not acquired:
        detail = {
            "message": "A similar data_upload job was submitted recently. Please retry later.",
            "code": "job_refractory_active",
            "retry_after_seconds": retry_after,
        }
        raise HTTPException(
            status_code=fastapi_status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(retry_after)},
        )
    
    stock_file_content = io.BytesIO(await stock_file.read())
    try:
        # Validate stock file content
        is_valid_stock, stock_error = validate_stock_file(
            stock_file_content, 
            stock_file.filename
        )
        if not is_valid_stock:
            raise AppValidationError(
                message=f"Invalid stock file: {stock_error}",
                details={"filename": stock_file.filename},
            )

        # Reset file position
        await stock_file.seek(0)

        # Validate sales files
        for i, sales_file in enumerate(sales_files):
            # Validate sales file format and size
            sales_file_content = io.BytesIO(await sales_file.read())
            # Validate sales file content
            is_valid_sales, sales_error = validate_sales_file(
                sales_file_content, 
                sales_file.filename
            )
            if not is_valid_sales:
                raise AppValidationError(
                    message=f"Invalid sales file ({sales_file.filename}): {sales_error}",
                    details={"filename": sales_file.filename, "index": i},
                )
            # Reset file position
            await sales_file.seek(0)

        # Create a new job *after* lock acquisition
        job_id = dal.create_job(
            JobType.DATA_UPLOAD,
            parameters=prospective_params,
        )

        # Создаем уникальную временную директорию для этого задания
        # Используем базовую директорию из настроек
        base_temp_dir = PathLibPath(get_settings().temp_upload_dir)
        base_temp_dir.mkdir(
            parents=True, exist_ok=True
        )  # Убедимся, что базовая директория существует
        temp_job_dir = base_temp_dir / job_id
        temp_job_dir.mkdir(exist_ok=False)  # Создаем уникальную директорию задания

        # Создаем поддиректорию для файлов продаж
        sales_dir = temp_job_dir / "sales"
        sales_dir.mkdir(exist_ok=True)

        # Сохраняем файлы асинхронно
        saved_stock_path = await _save_uploaded_file(stock_file, temp_job_dir)
        saved_sales_paths = []
        for sales_file in sales_files:
            saved_path = await _save_uploaded_file(sales_file, sales_dir)
            saved_sales_paths.append(str(saved_path))

        # Start the background task with file paths
        background_tasks.add_task(
            process_data_files,
            job_id=job_id,
            stock_file_path=str(saved_stock_path),
            sales_files_paths=saved_sales_paths,
            temp_dir_path=str(temp_job_dir),  # Передаем путь для очистки
            dal=dal, # Передаем dal
        )

        logger.info(
            f"Created data upload job {job_id} with files: {stock_file.filename} and {len(sales_files)} sales files"
        )
        return DataUploadResponse(job_id=job_id, status=JobStatus.PENDING)

    except AppValidationError as e:
        error = ErrorDetail(
            message=e.message,
            code="validation_error",
            status_code=fastapi_status.HTTP_400_BAD_REQUEST,
            details=e.details,
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump(),
        ) from e
    except DatabaseError as e:
        error = ErrorDetail(
            message="Failed to create data upload job due to database error",
            code="database_error",
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": str(e)},
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump(),
        ) from e
    except FileExistsError as e:
        # temp_job_dir and job_id are guaranteed to be set if error is from temp_job_dir.mkdir()
        detailed_error_reason = (
            f"Path {temp_job_dir} for job {job_id} already exists. Original error: {e}"
        )
        logger.error(
            f"FileExistsError for job {job_id}: Initial error for path {temp_job_dir}. Original: {e}",
            exc_info=True,
        )

        if temp_job_dir.is_dir():  # Check if the conflicting path is a directory
            try:
                dir_contents = os.listdir(temp_job_dir)
                if not dir_contents:
                    detailed_error_reason = (
                        f"Temporary directory {temp_job_dir} for job {job_id} already existed but was empty. "
                        "This might be a quick retry or incomplete cleanup."
                    )
                    logger.warning(detailed_error_reason)
                else:
                    # Log only the first few items to prevent overly long log messages
                    preview_contents = dir_contents[:5]
                    has_more_items = "..." if len(dir_contents) > 5 else ""
                    detailed_error_reason = (
                        f"Temporary directory {temp_job_dir} for job {job_id} already existed and was NOT empty "
                        f"(contains {len(dir_contents)} items: {preview_contents}{has_more_items}). This indicates a conflict."
                    )
                    logger.error(detailed_error_reason)
            except Exception as list_dir_exc:
                detailed_error_reason = (
                    f"Temporary directory {temp_job_dir} for job {job_id} existed, "
                    f"but an error occurred while checking its contents: {list_dir_exc}."
                )
                logger.error(detailed_error_reason, exc_info=True)
        elif temp_job_dir.is_file():
            detailed_error_reason = f"Path {temp_job_dir} for job {job_id} already existed as a FILE, not a directory."
            logger.error(detailed_error_reason)
        else:  # Exists but is not a dir or file (e.g. broken symlink) or check failed
            detailed_error_reason = (
                f"Path {temp_job_dir} for job {job_id} already existed, but it's not a regular file or directory, "
                f"or its status is inaccessible. Original error: {e}"
            )
            logger.error(detailed_error_reason)

        # Fail the job with the detailed reason
        if job_id:
            dal.update_job_status(
                job_id, JobStatus.FAILED.value, error_message=detailed_error_reason
            )

        error = ErrorDetail(
            message="Failed to initialize job resources: A path conflict occurred.",
            code="job_resource_conflict",  # More specific error code
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={
                "job_id": job_id,
                "path": str(temp_job_dir),
                "reason": detailed_error_reason,
                "original_exception": str(e),
            },
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump()
        ) from e
    except Exception as e:
        logger.error(
            f"Unexpected error in data-upload for job {job_id or 'unknown'}: {str(e)}",
            exc_info=True,
        )
        # Если ошибка произошла после создания задания, помечаем его как FAILED
        if job_id and not dal.get_job(job_id)["status"] == JobStatus.FAILED.value:
            dal.update_job_status(
                job_id,
                JobStatus.FAILED.value,
                error_message=f"Unexpected error during setup: {str(e)}",
            )
        # Очищаем временную директорию, если она была создана
        if temp_job_dir and temp_job_dir.exists():
            shutil.rmtree(temp_job_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory {temp_job_dir} after error.")

        error = ErrorDetail(
            message="Failed to create data upload job",
            code="internal_error",
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump()
        ) from e


@router.post("/training", response_model=TrainingResponse, summary="Submit a job to train a new model.")
async def create_training_job(
    request: Request,
    background_tasks: BackgroundTasks,
    params: TrainingParams | None = Body(None, description="An optional JSON object to specify `dataset_start_date` and `dataset_end_date` for the training data."),
    x_api_key_valid: dict[str, Any] = Depends(get_unified_auth),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
):
    """
    Initiates a model training job using the currently active hyperparameter configuration.
    The training dataset date range can be optionally specified. If not, the system
    determines the date range automatically based on available data.
    """
    logger.info("Received request to create training job")
    if params is None:
        params = TrainingParams()
    dataset_start_date = params.dataset_start_date
    dataset_end_date = params.dataset_end_date

    try:
        # 1. Determine adjusted training end date automatically
        dataset_end_date = dal.adjust_dataset_boundaries(
            start_date=dataset_start_date,
            end_date=dataset_end_date,
        )
        logger.info(
            f"Determined dataset_end_date: {dataset_end_date if dataset_end_date else 'None'}"
        )

        # 2. Get the effective configuration
        config = dal.get_effective_config(get_settings(), logger)
        if config is None:
            raise HTTPException(
                status_code=fastapi_status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "No active config and no best config by metric available",
                    "code": "no_config_available",
                },
            )
        logger.info(f"Using configuration: {config['config_id']}")

        # 3. Prepare job parameters for refractory check
        job_params = {"config_id": config["config_id"]}
        if dataset_start_date:
            job_params["dataset_start_date"] = dataset_start_date
        if dataset_end_date:
            job_params["dataset_end_date"] = dataset_end_date
            job_params["prediction_month"] = get_next_month(dataset_end_date)

        # Enforce refractory: job_type + parameter hash (after config validation)
        acquired, retry_after = dal.try_acquire_job_submission_lock(JobType.TRAINING.value, job_params)
        if not acquired:
            raise HTTPException(
                status_code=fastapi_status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": "A similar training job was submitted recently. Please retry later.",
                    "code": "job_refractory_active",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        job_id = dal.create_job(JobType.TRAINING, parameters=job_params)
        logger.info(f"Job record created with ID: {job_id}")

        # 4. Start the background task with the *adjusted* end date
        background_tasks.add_task(
            run_job,
            job_id=job_id,
            config=config["config"],
            config_id=config["config_id"],
            dataset_start_date=params.dataset_start_date,
            dataset_end_date=dataset_end_date,  # Use the adjusted date
            dal=dal, # Pass the dal
        )
        logger.info(f"Background task added for job ID: {job_id}")

        return TrainingResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            parameter_set_id=config["config_id"],
            using_active_parameters=True,
        )

    except HTTPException as e:
        logger.warning(
            f"HTTPException caught in create_training_job: {e.status_code} - {e.detail}"
        )
        raise e
    except ValueError as e:
        logger.warning(f"Validation error in create_training_job: {e}")
        error = ErrorDetail(
            message=str(e),
            code="invalid_date_range",
            status_code=fastapi_status.HTTP_400_BAD_REQUEST,
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code, detail=error.to_response_model().model_dump()
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in create_training_job: {e}", exc_info=True
        )
        error = ErrorDetail(
            message="An unexpected internal server error occurred while initiating the job.",
            code="internal_server_error",
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": str(e)},
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump(),
        ) from e


@router.post("/tuning", response_model=JobResponse, summary="Submit a hyperparameter tuning job.")
async def create_tuning_job(
    request: Request,
    background_tasks: BackgroundTasks,
    params: TuningParams | None = Body(None, description="An optional JSON object to specify `mode` (`lite` or `full`), `time_budget_s`, `dataset_start_date`, and `dataset_end_date`."),
    x_api_key_valid: dict[str, Any] = Depends(get_unified_auth),
    dal: DataAccessLayer = Depends(get_dal_for_general_user), # Inject DAL
):
    """
    Starts a hyperparameter tuning process. It can be run in `lite` or `full` mode.
    The system uses historical configurations to seed the tuning process.
    """
    logger.info("Received request to create tuning job")
    if params is None:
        params = TuningParams()
    
    try:
        # Determine adjusted training end date automatically
        dataset_start_date = params.dataset_start_date
        dataset_end_date = params.dataset_end_date
        mode = params.mode
        time_budget_s = params.time_budget_s
        dataset_end_date = dal.adjust_dataset_boundaries(
            start_date=dataset_start_date,
            end_date=dataset_end_date,
        )
        logger.info(
            f"Determined adjusted dataset_end_date: {dataset_end_date.isoformat() if dataset_end_date else 'None'}"
        )

        config = dal.get_effective_config(get_settings(), logger)
        if config is None:
            raise HTTPException(
                status_code=fastapi_status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "No active config and no best config by metric available",
                    "code": "no_config_available",
                    "status_code": fastapi_status.HTTP_400_BAD_REQUEST,
                    "details": None,
                },
            )
        logger.info(f"Using configuration: {config['config_id']}")

        job_params = {
            "mode": mode or get_settings().tuning.mode,
        }
        if dataset_start_date:
            job_params["dataset_start_date"] = dataset_start_date
        if dataset_end_date:
            job_params["dataset_end_date"] = dataset_end_date

        # Enforce refractory: job_type + parameter hash (after config validation)
        acquired, retry_after = dal.try_acquire_job_submission_lock(JobType.TUNING.value, job_params)
        if not acquired:
            raise HTTPException(
                status_code=fastapi_status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": "A similar tuning job was submitted recently. Please retry later.",
                    "code": "job_refractory_active",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # 3. Создание задания в БД
        job_id = dal.create_job(JobType.TUNING, parameters=job_params)
        logger.info(f"Tuning job created: {job_id}")

        # 4. Подготовка динамических параметров задачи
        additional_params: dict[str, int | str] = {}
        if mode:
            additional_params["mode"] = mode
        if time_budget_s is not None:
            additional_params["time_budget_s"] = time_budget_s
        elif mode and mode == "full":
            additional_params["time_budget_s"] = 7200
        else:
            additional_params["time_budget_s"] = 3600

        background_tasks.add_task(
            run_job,
            job_id=job_id,
            config=config["config"],
            config_id=config["config_id"],
            job_type="tune",
            dataset_start_date=dataset_start_date,
            dataset_end_date=dataset_end_date,
            additional_job_params=additional_params,
            dal=dal, # Pass DAL to background task
        )

        return JobResponse(job_id=job_id, status=JobStatus.PENDING)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_tuning_job: {e}", exc_info=True)
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Internal server error",
                "code": "internal_server_error",
                "status_code": fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
                "details": {"error": str(e)},
            },
        ) from e

@router.post("/reports", response_model=ReportResponse, summary="Generate a prediction report.")
async def create_prediction_report_job(
    request: Request,
    params: ReportParams = Body(..., description="A JSON object specifying the `report_type`, `prediction_month`, and optional `filters`."),
    x_api_key_valid: dict[str, Any] = Depends(get_unified_auth),
    dal: DataAccessLayer = Depends(get_dal_for_general_user), # Inject DAL
):
    """
    Creates and returns a report based on prediction results for a specified month.
    If no month is provided, it defaults to the latest month with available predictions.
    The report can be filtered and is returned as a CSV string.
    """
    try:
        prediction_month = params.prediction_month
        if prediction_month is None:
            # If no month is provided, get the latest one from prediction_results
            latest_month = dal.get_latest_prediction_month()
            if not latest_month:
                raise HTTPException(
                    status_code=fastapi_status.HTTP_404_NOT_FOUND,
                    detail={
                        "message": "No predictions found. Cannot generate a report for the latest month.",
                        "code": "no_predictions_found",
                    },
                )
            prediction_month = latest_month
            logger.info(f"Prediction month not provided for report, determined latest available month: {prediction_month}")

        # Update params with the determined month to pass to the report generator
        params.prediction_month = prediction_month

        logger.info(
            f"Received request to generate report for month: {prediction_month.strftime('%Y-%m')}"
        )

        # Generate the report directly
        report_df = generate_report(params=params, dal=dal) # Pass DAL to report generator

        # Convert DataFrame to CSV string
        csv_data = report_df.to_csv(index=False)

        # Create and return the response
        return ReportResponse(
            report_type=params.report_type.value,
            prediction_month=prediction_month.strftime("%Y-%m"),
            records_count=len(report_df),
            csv_data=csv_data,
            generated_at=datetime.now(),
            filters_applied=params.filters,
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions that were intentionally raised for specific error handling
        raise e
    except Exception as e:
        logger.error(f"Failed to create report: {e}", exc_info=True)
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Failed to create report",
                "code": "internal_server_error",
                "details": {"error": str(e)},
            },
        ) from e


@router.get("/{job_id}", response_model=JobDetails, summary="Get the status and details of a specific job.")
async def get_job_status(
    request: Request,
    job_id: str = Path(..., description="The unique identifier of the job."),
    x_api_key_valid: dict[str, Any] = Depends(get_unified_auth),
    dal: DataAccessLayer = Depends(get_dal_for_general_user), # Inject DAL
):
    """
    Retrieves the current status, progress, and other details of a job by its ID.
    If the job is completed, the response will include the results.
    """
    try:
        job = dal.get_job(job_id)

        if not job:
            raise HTTPException(
                status_code=fastapi_status.HTTP_404_NOT_FOUND,
                detail={
                    "message": f"Job with ID {job_id} not found",
                    "code": "job_not_found",
                    "status_code": fastapi_status.HTTP_404_NOT_FOUND,
                    "details": None,
                },
            )

        # Build basic response
        response = JobDetails(
            job_id=job["job_id"],
            job_type=job["job_type"],
            status=job["status"],
            created_at=datetime.fromisoformat(job["created_at"]),
            updated_at=datetime.fromisoformat(job["updated_at"]),
            progress=job["progress"] or 0.0,
            error=job["error_message"],
        )

        # Add result data if job is completed and has a result
        if job["status"] == JobStatus.COMPLETED.value and job["result_id"]:
            result = {}

            if job["job_type"] == JobType.DATA_UPLOAD.value:
                data_result = dal.get_data_upload_result(job["result_id"])
                if data_result:
                    result = {
                        "records_processed": data_result["records_processed"],
                        "features_generated": json.loads(
                            data_result["features_generated"]
                        ),
                        "processing_run_id": data_result["processing_run_id"],
                    }

            elif job["job_type"] == JobType.TRAINING.value:
                training_result = dal.get_training_results(result_id=job["result_id"])
                if training_result:
                    # Handle JSON deserialization for metrics
                    metrics = training_result.get("metrics", None)
                    if isinstance(metrics, str):
                        try:
                            metrics = json.loads(metrics)
                        except json.JSONDecodeError:
                            metrics = {}

                    # Get parameters from the job record
                    parameters = {}
                    if job.get("parameters", None):
                        try:
                            parameters = json.loads(job["parameters"])
                        except json.JSONDecodeError:
                            parameters = {}

                    result = {
                        "model_id": training_result["model_id"],
                        "metrics": metrics,
                        "parameters": parameters,  # Use parameters from job record
                        "config_id": training_result.get("config_id"),
                        "duration": training_result.get("duration"),
                    }

            elif job["job_type"] == JobType.PREDICTION.value:
                prediction_result = dal.get_prediction_result(job["result_id"])
                if prediction_result:
                    result = {
                        "model_id": prediction_result["model_id"],
                        "prediction_month": prediction_result["prediction_month"],
                        "output_path": prediction_result["output_path"],
                        "summary_metrics": json.loads(
                            prediction_result["summary_metrics"]
                        ),
                    }

            elif job["job_type"] == JobType.REPORT.value:
                report_result = dal.get_report_result(job["result_id"])
                if report_result:
                    # Ensure all fields for ReportResponse are populated if they exist in DB
                    result = {
                        "report_type": report_result["report_type"],
                        "prediction_month": report_result.get("prediction_month"),
                        "records_count": report_result.get("records_count"),
                        "csv_data": report_result.get("csv_data"),
                        "has_enriched_metrics": report_result.get("has_enriched_metrics"),
                        "enriched_columns": json.loads(report_result.get("enriched_columns", "[]")),
                        "generated_at": report_result.get("generated_at"),
                        "filters_applied": json.loads(report_result.get("filters_applied", "{}")),
                        "parameters": report_result["parameters"],
                        "output_path": report_result["output_path"],
                    }

            response.result = result

        return response

    except HTTPException:
        # Let built-in handlers deal with HTTP exceptions
        raise
    except AppValidationError as e:
        error = ErrorDetail(
            message=str(e),
            code="validation_error",
            status_code=fastapi_status.HTTP_400_BAD_REQUEST,
            details=e.details,
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump(),
        ) from e
    except DatabaseError as e:
        error = ErrorDetail(
            message=f"Failed to retrieve job {job_id}",
            code="database_error",
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump(),
        ) from e
    except ValueError as e:
        error = ErrorDetail(
            message=f"Invalid input data: {str(e)}",
            code="invalid_input_data",
            status_code=fastapi_status.HTTP_400_BAD_REQUEST,
            details={"error": str(e)},
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump(),
        ) from e
    except Exception as e:
        logger.error(
            f"Unexpected error in get_job_status for job {job_id}: {str(e)}",
            exc_info=True,
        )
        error = ErrorDetail(
            message=f"Failed to retrieve job {job_id}",
            code="internal_error",
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump()
        )


@router.get("", response_model=JobsList, summary="List all jobs with optional filtering.")
async def list_all_jobs(
    request: Request,
    job_type: JobType | None = Query(None, description="The type of job to filter by (e.g., `training`, `data_upload`)."),
    status: JobStatus | None = Query(None, description="The status of the job to filter by (e.g., `pending`, `completed`, `failed`)."),
    limit: int = Query(100, ge=1, le=1000, description="The maximum number of jobs to return."),
    x_api_key_valid: dict[str, Any] = Depends(get_unified_auth),
    dal: DataAccessLayer = Depends(get_dal_for_general_user), # Inject DAL
):
    """
    Retrieves a list of all jobs, which can be filtered by `job_type` and `status`.
    """
    try:
        jobs_data = dal.list_jobs(
            job_type=job_type.value if job_type else None,
            status=status.value if status else None,
            limit=limit,
        )

        # Convert job data to JobDetails objects
        jobs = []
        for job in jobs_data:
            job_details = JobDetails(
                job_id=job["job_id"],
                job_type=job["job_type"],
                status=job["status"],
                created_at=datetime.fromisoformat(job["created_at"]),
                updated_at=datetime.fromisoformat(job["updated_at"]),
                progress=job["progress"] or 0.0,
                error=job["error_message"],
            )
            jobs.append(job_details)

        return JobsList(jobs=jobs, total=len(jobs))

    except DatabaseError as e:
        logger.error(f"Unexpected error in list_all_jobs: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message="Failed to list jobs",
            code="database_error",
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump()
        )
    except Exception as e:
        logger.error(f"Unexpected error in list_all_jobs: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message="Failed to list jobs",
            code="internal_error",
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error(request)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump()
        )
