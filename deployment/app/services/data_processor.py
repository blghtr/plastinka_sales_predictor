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
    stock_file: UploadFile,
    sales_files: List[UploadFile],
    cutoff_date: str
) -> None:
    """
    Process uploaded files to extract features.
    
    Args:
        job_id: ID of the job
        stock_file: Uploaded stock file
        sales_files: List of uploaded sales files
        cutoff_date: Cutoff date for processing (DD.MM.YYYY)
    """
    try:
        # Update job status to running
        update_job_status(job_id, JobStatus.RUNNING.value, progress=0)
        
        # Create temporary directory for files
        temp_dir = Path("deployment/data/uploads") / job_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save files to temp directory
        stock_path = await save_uploaded_file(stock_file, temp_dir)
        
        # Create sales directory
        sales_dir = temp_dir / "sales"
        sales_dir.mkdir(exist_ok=True)
        
        # Save sales files
        for sales_file in sales_files:
            await save_uploaded_file(sales_file, sales_dir)
        
        # Update progress
        update_job_status(job_id, JobStatus.RUNNING.value, progress=20)
        
        # Process the data using the existing pipeline
        features = process_data(
            stock_path=str(stock_path),
            sales_path=str(sales_dir),
            cutoff_date=cutoff_date
        )
        
        # Update progress
        update_job_status(job_id, JobStatus.RUNNING.value, progress=60)
        
        # Get stock features (availability and confidence)
        if 'stock' in features and 'change' in features:
            stock_features = get_stock_features(
                features['stock'],
                features['change']
            )
            
            # Add stock features to the features dictionary
            features['availability'] = stock_features.loc[:, pd.IndexSlice['availability', :]]
            features['confidence'] = stock_features.loc[:, pd.IndexSlice['confidence', :]]
        
        # Update progress
        update_job_status(job_id, JobStatus.RUNNING.value, progress=80)
        
        # Save features using our SQL feature storage
        source_files = ", ".join([stock_file.filename] + [f.filename for f in sales_files])
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


async def save_uploaded_file(uploaded_file: UploadFile, directory: Path) -> Path:
    """
    Save an uploaded file to a directory.
    
    Args:
        uploaded_file: The uploaded file
        directory: Directory to save the file in
        
    Returns:
        Path to the saved file
    """
    file_path = directory / uploaded_file.filename
    
    # Read file content
    content = await uploaded_file.read()
    
    # Write to file
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Reset file pointer for potential reuse
    await uploaded_file.seek(0)
    
    return file_path 