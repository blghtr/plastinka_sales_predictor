import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import aiofiles
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from deployment.app.config import get_settings
from deployment.app.db.database import (
    DatabaseError,
    create_job,
    get_data_upload_result,
    get_effective_config,
    get_job,
    get_prediction_result,
    get_report_result,
    get_training_result,
    list_jobs,
    update_job_status,
)
from deployment.app.models.api_models import (
    DataUploadResponse,
    JobDetails,
    JobResponse,
    JobsList,
    JobStatus,
    JobType,
    PredictionParams,
    PredictionResponse,
    ReportParams,
    TrainingResponse,
)
from deployment.app.services.auth import get_current_api_key_validated
from deployment.app.services.data_processor import process_data_files
from deployment.app.services.datasphere_service import run_job
from deployment.app.services.report_service import run_report_job
from deployment.app.utils.error_handling import ErrorDetail
from deployment.app.utils.file_validation import validate_data_file_upload
from deployment.app.utils.validation import (
    ValidationError,
    validate_date_format,
    validate_historical_date_range,
    validate_sales_file,
    validate_stock_file,
)

logger = logging.getLogger("plastinka.api")

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


async def _save_uploaded_file(uploaded_file: UploadFile, directory: Path) -> Path:
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


@router.post("/data-upload", response_model=DataUploadResponse)
async def create_data_upload_job(
    request: Request,
    background_tasks: BackgroundTasks,
    stock_file: UploadFile = File(..., description="Excel (.xlsx, .xls) or CSV file with stock data"),
    sales_files: list[UploadFile] = File(
        ..., description="Excel (.xlsx, .xls) or CSV files with sales data"
    ),
    cutoff_date: str = Form(
        "30.09.2022", description="Cutoff date for data processing (DD.MM.YYYY)"
    ),
    api_key: bool = Depends(get_current_api_key_validated),
):
    """
    Submit a job to process data files.

    This will:
    1. Upload the files to a temporary location
    2. Process them using the existing pipeline
    3. Store processed features in the database

    Returns a job ID that can be used to check the job status.
    """
    job_id = None
    temp_job_dir = None
    try:
        # Validate cutoff date
        is_valid_date, parsed_date = validate_date_format(cutoff_date)
        if not is_valid_date:
            raise ValidationError(
                message="Invalid cutoff date format. Expected format: DD.MM.YYYY",
                details={"cutoff_date": cutoff_date},
            )

        # Validate stock file format and size
        await validate_data_file_upload(stock_file)

        # Validate stock file content
        stock_content = await stock_file.read()
        is_valid_stock, stock_error = validate_stock_file(stock_content, stock_file.filename)
        if not is_valid_stock:
            raise ValidationError(
                message=f"Invalid stock file: {stock_error}",
                details={"filename": stock_file.filename},
            )

        # Reset file position
        await stock_file.seek(0)

        # Validate sales files
        for i, sales_file in enumerate(sales_files):
            # Validate sales file format and size
            await validate_data_file_upload(sales_file)

            # Validate sales file content
            sales_content = await sales_file.read()
            is_valid_sales, sales_error = validate_sales_file(sales_content, sales_file.filename)
            if not is_valid_sales:
                raise ValidationError(
                    message=f"Invalid sales file ({sales_file.filename}): {sales_error}",
                    details={"filename": sales_file.filename, "index": i},
                )
            # Reset file position
            await sales_file.seek(0)

        # Create a new job *before* creating the temp directory
        job_id = create_job(
            JobType.DATA_UPLOAD,
            parameters={
                "stock_file": stock_file.filename,
                "sales_files": [f.filename for f in sales_files],
                "cutoff_date": cutoff_date,
            },
        )

        # Создаем уникальную временную директорию для этого задания
        # Используем базовую директорию из настроек
        base_temp_dir = Path(get_settings().temp_upload_dir)
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
            cutoff_date=cutoff_date,
            temp_dir_path=str(temp_job_dir),  # Передаем путь для очистки
        )

        logger.info(
            f"Created data upload job {job_id} with files: {stock_file.filename} and {len(sales_files)} sales files"
        )
        return DataUploadResponse(job_id=job_id, status=JobStatus.PENDING)

    except ValidationError:
        # ValidationError is handled by the app_validation_exception_handler
        raise
    except DatabaseError as e:
        logger.error(f"Database error in data-upload: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message="Failed to create data upload job due to database error",
            code="database_error",
            status_code=500,
            details={"error": str(e)},
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
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
            update_job_status(
                job_id, JobStatus.FAILED.value, error_message=detailed_error_reason
            )

        error = ErrorDetail(
            message="Failed to initialize job resources: A path conflict occurred.",
            code="job_resource_conflict",  # More specific error code
            status_code=500,
            details={
                "job_id": job_id,
                "path": str(temp_job_dir),
                "reason": detailed_error_reason,
                "original_exception": str(e),
            },
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
    except Exception as e:
        logger.error(
            f"Unexpected error in data-upload for job {job_id or 'unknown'}: {str(e)}",
            exc_info=True,
        )
        # Если ошибка произошла после создания задания, помечаем его как FAILED
        if job_id and not get_job(job_id)["status"] == JobStatus.FAILED.value:
            update_job_status(
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
            status_code=500,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())


@router.post("/training", response_model=TrainingResponse)
async def create_training_job(
    request: Request,
    background_tasks: BackgroundTasks,
    dataset_start_date: str | None = None,
    dataset_end_date: str | None = None,
    api_key: bool = Depends(get_current_api_key_validated),
):
    """
    Submit a job to train a model using the active parameter set.

    This will:
    1. Prepare the training dataset using the specified date range
    2. Train the model using the active parameter set or best parameter set
    3. Save the trained model

    Returns a job ID that can be used to check the job status.

    Returns:
        HTTPException: If there's no active parameter set and no best parameter set by metric.
    """
    logger.info("Received request to create training job")
    try:
        # 1. Валидация и парсинг дат
        parsed_start_date = None
        parsed_end_date = None

        if dataset_start_date:
            is_valid, parsed_start_date = validate_date_format(
                dataset_start_date, format_str="%d.%m.%Y"
            )
            if not is_valid:
                logger.warning(f"Invalid start date format: {dataset_start_date}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid start date format. Expected DD.MM.YYYY.",
                )
        if dataset_end_date:
            is_valid, parsed_end_date = validate_date_format(
                dataset_end_date, format_str="%d.%m.%Y"
            )
            if not is_valid:
                logger.warning(f"Invalid end date format: {dataset_end_date}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid end date format. Expected DD.MM.YYYY.",
                )

        # 2. Логическая валидация диапазона дат (если обе даты заданы)
        if parsed_start_date and parsed_end_date:
            is_valid, error_msg, _, _ = validate_historical_date_range(
                parsed_start_date, parsed_end_date, format_str="%d.%m.%Y"
            )
            if not is_valid:
                logger.warning(f"Invalid date range: {error_msg}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg
                )

        # 3. Получение конфига
        config = get_effective_config(get_settings(), logger=logger)
        if config is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active config and no best config by metric available",
            )
        logger.info(f"Using configuration: {config['config_id']}")

        # 4. Создание задания в БД
        job_params = {"config_id": config["config_id"]}
        if parsed_start_date:
            job_params["dataset_start_date"] = parsed_start_date
        if parsed_end_date:
            job_params["dataset_end_date"] = parsed_end_date

        job_id = create_job(JobType.TRAINING, parameters=job_params)
        logger.info(f"Job record created with ID: {job_id}")

        # 5. Запуск фоновой задачи
        background_tasks.add_task(
            run_job,
            job_id=job_id,
            training_config=config["config"],
            config_id=config["config_id"],
            dataset_start_date=parsed_start_date,
            dataset_end_date=parsed_end_date,
        )
        logger.info(f"Background task added for job ID: {job_id}")

        return TrainingResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            parameter_set_id=config["config_id"],
            using_active_parameters=True,
        )

    except HTTPException as e:
        # Re-raise HTTPExceptions that were intentionally raised (e.g., from get_effective_config)
        logger.warning(
            f"HTTPException caught in create_training_job: {e.status_code} - {e.detail}"
        )
        raise e
    except ValueError as e:
        # Catch ValueErrors, potentially from date parsing issues before the function signature,
        # or custom validation added inside.
        logger.error(f"ValueError in create_training_job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input data: {e}"
        ) from e
    # Assuming DatabaseError is a specific exception type raised by your database layer functions
    except Exception as e:  # Catching general Exception for database or other unexpected errors during sync part
        # Log any other unexpected errors with traceback
        logger.error(
            f"An unexpected error occurred in create_training_job: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred while initiating the job.",
        ) from e


@router.post("/prediction", response_model=PredictionResponse)
async def create_prediction_job(
    request: Request,
    params: PredictionParams,
    background_tasks: BackgroundTasks,
    api_key: bool = Depends(get_current_api_key_validated),
):
    """
    Submit a job to generate predictions.

    This will:
    1. Load the model specified by model_id
    2. Generate predictions
    3. Save the prediction results

    Returns a job ID that can be used to check the job status.
    """
    try:
        raise NotImplementedError("Prediction job is not implemented yet")

    except DatabaseError as e:
        logger.error(
            f"Database error in prediction job creation: {str(e)}", exc_info=True
        )
        error = ErrorDetail(
            message="Failed to create prediction job due to database error",
            code="database_error",
            status_code=500,
            details={"error": str(e)},
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
    except Exception as e:
        logger.error(
            f"Unexpected error in prediction job creation: {str(e)}", exc_info=True
        )
        error = ErrorDetail(
            message="Failed to create prediction job",
            code="internal_error",
            status_code=500,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())


@router.post("/reports", response_model=JobResponse)
async def create_prediction_report_job(
    request: Request,
    params: ReportParams,
    background_tasks: BackgroundTasks,
    api_key: bool = Depends(get_current_api_key_validated),
):
    """
    Submit a job to generate a prediction report.

    This will:
    1. Create a job record in the database.
    2. Add a background task to generate the report.

    Returns a job ID that can be used to check the job status.
    """
    try:
        logger.info(
            f"Received request to generate report for month: {params.prediction_month.strftime('%Y-%m')}"
        )

        # Create a job record
        job_id = create_job(job_type=JobType.REPORT, parameters=params.model_dump())

        # Add the report generation to background tasks
        background_tasks.add_task(run_report_job, job_id=job_id, params=params)

        logger.info(f"Report job created with ID: {job_id}")
        return JobResponse(job_id=job_id, status=JobStatus.PENDING)

    except Exception as e:
        logger.error(f"Failed to create report job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create report job",
        ) from e


@router.get("/{job_id}", response_model=JobDetails)
async def get_job_status(
    request: Request,
    job_id: str,
    api_key: bool = Depends(get_current_api_key_validated),
):
    """
    Get the status and details of a job.

    If the job is completed, this will also include the results.
    """
    try:
        job = get_job(job_id)

        if not job:
            raise HTTPException(
                status_code=404, detail=f"Job with ID {job_id} not found"
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
                data_result = get_data_upload_result(job["result_id"])
                if data_result:
                    result = {
                        "records_processed": data_result["records_processed"],
                        "features_generated": json.loads(
                            data_result["features_generated"]
                        ),
                        "processing_run_id": data_result["processing_run_id"],
                    }

            elif job["job_type"] == JobType.TRAINING.value:
                training_result = get_training_result(job["result_id"])
                if training_result:
                    result = {
                        "model_id": training_result["model_id"],
                        "metrics": json.loads(training_result["metrics"]),
                        "parameters": json.loads(training_result["parameters"]),
                        "duration": training_result["duration"],
                    }

            elif job["job_type"] == JobType.PREDICTION.value:
                prediction_result = get_prediction_result(job["result_id"])
                if prediction_result:
                    result = {
                        "model_id": prediction_result["model_id"],
                        "prediction_date": prediction_result["prediction_date"],
                        "output_path": prediction_result["output_path"],
                        "summary_metrics": json.loads(
                            prediction_result["summary_metrics"]
                        ),
                    }

            elif job["job_type"] == JobType.REPORT.value:
                report_result = get_report_result(job["result_id"])
                if report_result:
                    result = {
                        "report_type": report_result["report_type"],
                        "parameters": json.loads(report_result["parameters"]),
                        "output_path": report_result["output_path"],
                    }

            response.result = result

        return response

    except HTTPException:
        # Let built-in handlers deal with HTTP exceptions
        raise
    except DatabaseError as e:
        logger.error(
            f"Database error in get_job_status for job {job_id}: {str(e)}",
            exc_info=True,
        )
        error = ErrorDetail(
            message=f"Failed to retrieve job {job_id}",
            code="internal_error",
            status_code=500,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
    except Exception as e:
        logger.error(
            f"Unexpected error in get_job_status for job {job_id}: {str(e)}",
            exc_info=True,
        )
        error = ErrorDetail(
            message=f"Failed to retrieve job {job_id}",
            code="internal_error",
            status_code=500,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())


@router.get("", response_model=JobsList)
async def list_all_jobs(
    request: Request,
    job_type: JobType | None = None,
    status: JobStatus | None = None,
    limit: int = Query(100, ge=1, le=1000),
    api_key: bool = Depends(get_current_api_key_validated),
):
    """
    List all jobs with optional filtering by type and status.
    """
    try:
        jobs_data = list_jobs(
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
            code="internal_error",
            status_code=500,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
    except Exception as e:
        logger.error(f"Unexpected error in list_all_jobs: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message="Failed to list jobs",
            code="internal_error",
            status_code=500,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
