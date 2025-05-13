from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends, Query, Body, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
from datetime import datetime
import uuid
import json
import logging
from pathlib import Path
import aiofiles
import shutil

from app.models.api_models import (
    JobResponse, JobDetails, JobsList, JobStatus, JobType,
    DataUploadResponse, TrainingParams, TrainingResponse,
    PredictionParams, PredictionResponse, ReportParams, ReportResponse
)
from app.db.database import (
    create_job, update_job_status, get_job, list_jobs,
    get_data_upload_result, get_training_result, get_prediction_result, get_report_result,
    create_data_upload_result, create_training_result, create_prediction_result, create_report_result,
    DatabaseError, get_active_parameter_set, get_best_parameter_set_by_metric
)
from app.services.data_processor import process_data_files
from app.services.datasphere_service import run_job
from app.services.report_service import generate_report
from app.utils.validation import validate_stock_file, validate_sales_file, validate_date_format, ValidationError
from app.utils.file_validation import validate_excel_file_upload
from app.utils.error_handling import ErrorDetail
import debugpy
from app.config import settings

logger = logging.getLogger("plastinka.api")

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


async def _save_uploaded_file(uploaded_file: UploadFile, directory: Path) -> Path:
    """Helper function to save UploadFile asynchronously."""
    file_path = directory / uploaded_file.filename
    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            while content := await uploaded_file.read(1024 * 1024):  # Read in chunks
                await out_file.write(content)
        await uploaded_file.seek(0) # Reset pointer if needed elsewhere (though likely not)
    except Exception as e:
        logger.error(f"Failed to save file {uploaded_file.filename} to {directory}: {e}", exc_info=True)
        # Optionally remove partially saved file
        if file_path.exists():
            os.remove(file_path)
        raise # Re-raise the exception to be caught by the main handler
    return file_path


@router.post("/data-upload", response_model=DataUploadResponse)
async def create_data_upload_job(
    request: Request,
    background_tasks: BackgroundTasks,
    stock_file: UploadFile = File(..., description="Excel file with stock data"),
    sales_files: List[UploadFile] = File(..., description="Excel files with sales data"),
    cutoff_date: str = Form("30.09.2022", description="Cutoff date for data processing (DD.MM.YYYY)")
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
                details={"cutoff_date": cutoff_date}
            )
        
        # Validate stock file format and size
        await validate_excel_file_upload(stock_file)
        
        # Validate stock file content
        stock_content = await stock_file.read()
        is_valid_stock, stock_error = validate_stock_file(stock_content)
        if not is_valid_stock:
            raise ValidationError(
                message=f"Invalid stock file: {stock_error}",
                details={"filename": stock_file.filename}
            )
        
        # Reset file position
        await stock_file.seek(0)
        
        # Validate sales files
        for i, sales_file in enumerate(sales_files):
            # Validate sales file format and size
            await validate_excel_file_upload(sales_file)
            
            # Validate sales file content
            sales_content = await sales_file.read()
            is_valid_sales, sales_error = validate_sales_file(sales_content)
            if not is_valid_sales:
                raise ValidationError(
                    message=f"Invalid sales file ({sales_file.filename}): {sales_error}",
                    details={"filename": sales_file.filename, "index": i}
                )
            # Reset file position
            await sales_file.seek(0)
        
        # Create a new job *before* creating the temp directory
        job_id = create_job(
            JobType.DATA_UPLOAD,
            parameters={
                "stock_file": stock_file.filename,
                "sales_files": [f.filename for f in sales_files],
                "cutoff_date": cutoff_date
            }
        )
        
        # Создаем уникальную временную директорию для этого задания
        # Используем базовую директорию из настроек
        base_temp_dir = Path(settings.temp_upload_dir)
        base_temp_dir.mkdir(parents=True, exist_ok=True) # Убедимся, что базовая директория существует
        temp_job_dir = base_temp_dir / job_id
        temp_job_dir.mkdir(exist_ok=False) # Создаем уникальную директорию задания
        
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
            temp_dir_path=str(temp_job_dir) # Передаем путь для очистки
        )
        
        logger.info(f"Created data upload job {job_id} with files: {stock_file.filename} and {len(sales_files)} sales files")
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
            details={"error": str(e)}
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
    except FileExistsError as e:
        logger.error(f"Temporary directory for job {job_id} already exists: {e}", exc_info=True)
        # Если задание уже создано, но директорию создать не удалось, возможно, стоит отменить задание
        if job_id: update_job_status(job_id, JobStatus.FAILED.value, error_message="Failed to create temporary directory")
        error = ErrorDetail(message="Failed to initialize job resources", code="internal_error", status_code=500, details={"error": str(e)})
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
    except Exception as e:
        logger.error(f"Unexpected error in data-upload for job {job_id or 'unknown'}: {str(e)}", exc_info=True)
        # Если ошибка произошла после создания задания, помечаем его как FAILED
        if job_id and not get_job(job_id)['status'] == JobStatus.FAILED.value:
             update_job_status(job_id, JobStatus.FAILED.value, error_message=f"Unexpected error during setup: {str(e)}")
        # Очищаем временную директорию, если она была создана
        if temp_job_dir and temp_job_dir.exists():
            shutil.rmtree(temp_job_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory {temp_job_dir} after error.")
            
        error = ErrorDetail(
            message="Failed to create data upload job",
            code="internal_error",
            status_code=500,
            details={"error": str(e)},
            exception=e
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())


@router.post("/training", response_model=TrainingResponse)
async def create_training_job(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Submit a job to train a model using the active parameter set.
    
    This will:
    1. Prepare the training dataset
    2. Train the model using the active parameter set or best parameter set
    3. Save the trained model
    
    Returns a job ID that can be used to check the job status.
    
    Raises:
        HTTPException: If there's no active parameter set and no best parameter set by metric.
    """
    try:
        # Check if active parameter set exists
        active_params_data = get_active_parameter_set()
        active_parameter_set_id = None
        
        if active_params_data:
            active_parameter_set_id = active_params_data["parameter_set_id"]
            logger.info(f"Found active parameter set: {active_parameter_set_id}")
        else:
            # Check if there's a best parameter set by metric
            metric_name = settings.default_metric
            best_params_data = get_best_parameter_set_by_metric(
                metric_name, 
                settings.default_metric_higher_is_better
            )
            
            if not best_params_data:
                error_msg = "No active parameter set and no best parameter set by metric available"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=400,
                    detail=error_msg
                )
            
            active_parameter_set_id = best_params_data["parameter_set_id"]
            logger.info(f"Using best parameter set by {metric_name}: {active_parameter_set_id}")
        
        # Create a new job
        job_id = create_job(
            JobType.TRAINING,
            parameters={"use_active_parameters": True, "parameter_set_id": active_parameter_set_id}
        )
        
        # Start the background task
        background_tasks.add_task(
            run_job,
            job_id=job_id
        )
        
        logger.info(f"Created training job {job_id} using parameter set {active_parameter_set_id}")
        return TrainingResponse(
            job_id=job_id, 
            status=JobStatus.PENDING,
            parameter_set_id=active_parameter_set_id,
            using_active_parameters=True
        )
    except DatabaseError as e:
        logger.error(f"Database error in training job creation: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message="Failed to create training job due to database error",
            code="database_error",
            status_code=500,
            details={"error": str(e)}
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
    except Exception as e:
        logger.error(f"Unexpected error in training job creation: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message="Failed to create training job",
            code="internal_error",
            status_code=500,
            details={"error": str(e)},
            exception=e
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())


@router.post("/prediction", response_model=PredictionResponse)
async def create_prediction_job(
    request: Request,
    params: PredictionParams,
    background_tasks: BackgroundTasks
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
        # Create a new job
        job_id = create_job(
            JobType.PREDICTION,
            parameters=params.model_dump()
        )
        
        # Start the background task
        #background_tasks.add_task(
        #    generate_predictions,
        #    job_id=job_id,
        #    params=params
        #)
        
        logger.info(f"Created prediction job {job_id} with model_id: {params.model_id}")
        return PredictionResponse(job_id=job_id, status=JobStatus.PENDING)
        
    except DatabaseError as e:
        logger.error(f"Database error in prediction job creation: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message="Failed to create prediction job due to database error",
            code="database_error",
            status_code=500,
            details={"error": str(e)}
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
    except Exception as e:
        logger.error(f"Unexpected error in prediction job creation: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message="Failed to create prediction job",
            code="internal_error",
            status_code=500,
            details={"error": str(e)},
            exception=e
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())


@router.post("/reports", response_model=ReportResponse)
async def create_report_job(
    request: Request,
    params: ReportParams,
    background_tasks: BackgroundTasks
):
    """
    Submit a job to generate a report.
    
    This will:
    1. Prepare the data needed for the report
    2. Generate the report
    3. Save the report
    
    Returns a job ID that can be used to check the job status.
    """
    try:
        # Create a new job
        job_id = create_job(
            JobType.REPORT,
            parameters=params.model_dump()
        )
        
        # Start the background task
        background_tasks.add_task(
            generate_report,
            job_id=job_id,
            params=params
        )
        
        logger.info(f"Created report job {job_id} of type: {params.report_type}")
        return ReportResponse(job_id=job_id, status=JobStatus.PENDING)
        
    except DatabaseError as e:
        logger.error(f"Database error in report job creation: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message="Failed to create report job due to database error",
            code="database_error",
            status_code=500,
            details={"error": str(e)}
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
    except Exception as e:
        logger.error(f"Unexpected error in report job creation: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message="Failed to create report job",
            code="internal_error",
            status_code=500,
            details={"error": str(e)},
            exception=e
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())


@router.get("/{job_id}", response_model=JobDetails)
async def get_job_status(request: Request, job_id: str):
    """
    Get the status and details of a job.
    
    If the job is completed, this will also include the results.
    """
    try:
        job = get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
        
        # Build basic response
        response = JobDetails(
            job_id=job["job_id"],
            job_type=job["job_type"],
            status=job["status"],
            created_at=datetime.fromisoformat(job["created_at"]),
            updated_at=datetime.fromisoformat(job["updated_at"]),
            progress=job["progress"] or 0.0,
            error=job["error_message"]
        )
        
        # Add result data if job is completed and has a result
        if job["status"] == JobStatus.COMPLETED.value and job["result_id"]:
            result = {}
            
            if job["job_type"] == JobType.DATA_UPLOAD.value:
                data_result = get_data_upload_result(job["result_id"])
                if data_result:
                    result = {
                        "records_processed": data_result["records_processed"],
                        "features_generated": json.loads(data_result["features_generated"]),
                        "processing_run_id": data_result["processing_run_id"]
                    }
                    
            elif job["job_type"] == JobType.TRAINING.value:
                training_result = get_training_result(job["result_id"])
                if training_result:
                    result = {
                        "model_id": training_result["model_id"],
                        "metrics": json.loads(training_result["metrics"]),
                        "parameters": json.loads(training_result["parameters"]),
                        "duration": training_result["duration"]
                    }
                    
            elif job["job_type"] == JobType.PREDICTION.value:
                prediction_result = get_prediction_result(job["result_id"])
                if prediction_result:
                    result = {
                        "model_id": prediction_result["model_id"],
                        "prediction_date": prediction_result["prediction_date"],
                        "output_path": prediction_result["output_path"],
                        "summary_metrics": json.loads(prediction_result["summary_metrics"])
                    }
                    
            elif job["job_type"] == JobType.REPORT.value:
                report_result = get_report_result(job["result_id"])
                if report_result:
                    result = {
                        "report_type": report_result["report_type"],
                        "parameters": json.loads(report_result["parameters"]),
                        "output_path": report_result["output_path"]
                    }
            
            response.result = result
            
        return response
        
    except HTTPException:
        # Let built-in handlers deal with HTTP exceptions
        raise
    except DatabaseError as e:
        logger.error(f"Database error in get_job_status for job {job_id}: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message=f"Failed to retrieve job {job_id} due to database error",
            code="database_error",
            status_code=500,
            details={"error": str(e)}
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())
    except Exception as e:
        logger.error(f"Unexpected error in get_job_status for job {job_id}: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message=f"Failed to retrieve job {job_id}",
            code="internal_error",
            status_code=500,
            details={"error": str(e)},
            exception=e
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict())


@router.get("", response_model=JobsList)
async def list_all_jobs(
    request: Request,
    job_type: Optional[JobType] = None,
    status: Optional[JobStatus] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """
    List all jobs with optional filtering by type and status.
    """
    try:
        jobs_data = list_jobs(
            job_type=job_type.value if job_type else None,
            status=status.value if status else None,
            limit=limit
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
                error=job["error_message"]
            )
            jobs.append(job_details)
        
        return JobsList(jobs=jobs, total=len(jobs))
        
    except DatabaseError as e:
        logger.error(f"Database error in list_all_jobs: {str(e)}", exc_info=True)
        error = ErrorDetail(
            message="Failed to list jobs due to database error",
            code="database_error",
            status_code=500,
            details={"error": str(e)}
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
            exception=e
        )
        error.log_error(request)
        return JSONResponse(status_code=500, content=error.to_dict()) 