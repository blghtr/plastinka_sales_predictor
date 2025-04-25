from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends, Query, Body, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
from datetime import datetime
import uuid
import json
import logging

from app.models.api_models import (
    JobResponse, JobDetails, JobsList, JobStatus, JobType,
    DataUploadResponse, TrainingParams, TrainingResponse,
    PredictionParams, PredictionResponse, ReportParams, ReportResponse
)
from app.db.database import (
    create_job, update_job_status, get_job, list_jobs,
    get_data_upload_result, get_training_result, get_prediction_result, get_report_result,
    create_data_upload_result, create_training_result, create_prediction_result, create_report_result,
    DatabaseError
)
from app.services.data_processor import process_data_files
from app.services.training_service import train_model
from app.services.prediction_service import generate_predictions
from app.services.report_service import generate_report
from app.utils.validation import validate_stock_file, validate_sales_file, validate_date_format, ValidationError
from app.utils.file_validation import validate_excel_file_upload
from app.utils.error_handling import ErrorDetail

logger = logging.getLogger("plastinka.api")

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


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
        
        # Create a new job
        job_id = create_job(
            JobType.DATA_UPLOAD,
            parameters={
                "stock_file": stock_file.filename,
                "sales_files": [f.filename for f in sales_files],
                "cutoff_date": cutoff_date
            }
        )
        
        # Start the background task
        background_tasks.add_task(
            process_data_files,
            job_id=job_id,
            stock_file=stock_file,
            sales_files=sales_files,
            cutoff_date=cutoff_date
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
    except Exception as e:
        logger.error(f"Unexpected error in data-upload: {str(e)}", exc_info=True)
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
    params: TrainingParams,
    background_tasks: BackgroundTasks
):
    """
    Submit a job to train a model.
    
    This will:
    1. Prepare the training dataset
    2. Train the model with the provided parameters
    3. Save the trained model
    
    Returns a job ID that can be used to check the job status.
    """
    try:
        # Create a new job
        job_id = create_job(
            JobType.TRAINING,
            parameters=params.model_dump()
        )
        
        # Start the background task
        background_tasks.add_task(
            train_model,
            job_id=job_id,
            params=params
        )
        
        logger.info(f"Created training job {job_id} with parameters: {params.model_dump()}")
        return TrainingResponse(job_id=job_id, status=JobStatus.PENDING)
        
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
        background_tasks.add_task(
            generate_predictions,
            job_id=job_id,
            params=params
        )
        
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