import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import shutil
import pandas as pd
from fastapi import UploadFile

# Import our custom modules
from deployment.app.db.database import update_job_status, create_data_upload_result
from deployment.app.db.feature_storage import save_features
from deployment.app.models.api_models import JobStatus

# Import the necessary functions from the original codebase
from plastinka_sales_predictor.data_preparation import process_data, get_stock_features

async def process_data_files(
    job_id: str,
    stock_file_path: str,
    sales_files_paths: List[str],
    cutoff_date: str,
    temp_dir_path: str
) -> None:
    """
    Process uploaded files to extract features.
    
    Args:
        job_id: ID of the job
        stock_file_path: Path to the saved stock file
        sales_files_paths: List of paths to the saved sales files
        cutoff_date: Cutoff date for processing (DD.MM.YYYY)
        temp_dir_path: Path to the temporary directory used for this job
    """
    temp_dir = Path(temp_dir_path)
    stock_path = Path(stock_file_path)
    
    try:
        # Update job status to running
        update_job_status(job_id, JobStatus.RUNNING.value, progress=0)
        
        # Check if files exist
        if not stock_path.exists():
            raise FileNotFoundError(f"Stock file not found at {stock_path}")
        for p in sales_files_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Sales file not found at {p}")
        
        update_job_status(job_id, JobStatus.RUNNING.value, progress=20)
        
        # Process the data using the existing pipeline
        sales_dir_path = temp_dir / "sales"
        features = process_data(
            stock_path=str(stock_path),
            sales_path=str(sales_dir_path),
            cutoff_date=cutoff_date
        )
        
        update_job_status(job_id, JobStatus.RUNNING.value, progress=80)
        
        # Save features using our SQL feature storage
        stock_filename = stock_path.name
        sales_filenames = [Path(p).name for p in sales_files_paths]
        source_files = ", ".join([stock_filename] + sales_filenames)
        run_id = save_features(
            features,
            cutoff_date,
            source_files,
            store_type='sql'
        )
        
        # Create result record
        result_id = create_data_upload_result(
            job_id=job_id,
            records_processed=sum(df.shape[0] for df in features.values() if hasattr(df, 'shape')),
            features_generated=list(features.keys()),
            processing_run_id=run_id
        )
        
        # Update job as completed
        update_job_status(
            job_id,
            JobStatus.COMPLETED.value,
            progress=100,
            result_id=result_id
        )
        
    except Exception as e:
        # Update job as failed with error message
        update_job_status(
            job_id,
            JobStatus.FAILED.value,
            error_message=str(e)
        )
        # Re-raise for logging
        raise
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


# Удаляем функцию save_uploaded_file, так как она больше не нужна
# async def save_uploaded_file(uploaded_file: UploadFile, directory: Path) -> Path:
#     ... 