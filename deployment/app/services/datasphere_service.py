from datetime import datetime
import time
import os
import json
import uuid
import asyncio
import logging
from deployment.app.db.database import update_job_status, create_training_result
from deployment.app.models.api_models import JobStatus, TrainingParams
from deployment.app.config import settings
from deployment.datasphere.client import DataSphereClient
from deployment.datasphere.prepare_datasets import get_datasets
import shutil

# Set up logger
logger = logging.getLogger(__name__)

# TODO: Закончить и исправить тесты
async def run_job(job_id: str, params: TrainingParams) -> None:
    """
    Runs a job using the specified parameters and Yandex Datasphere.
    
    Args:
        job_id: ID of the job
        params: Training parameters
    """
    
    try:
        # Update job status to running
        update_job_status(job_id, JobStatus.RUNNING.value, progress=0)
        
        # Get datasets (binaries are saved to job's directory)
        train_dataset, val_dataset = get_datasets(params)
        
        update_job_status(
            job_id, 
            JobStatus.RUNNING.value, 
            progress=10
        )
        
        # Initialize DataSphere client
        client = DataSphereClient(**settings.datasphere.client)  # TODO Implement datasphere settings

        # Submit job to DataSphere
        job_id = client.submit_job_cli(settings.datasphere.job_config)
        # Poll for function execution completion
        completed = False
        max_polls = 30  # Maximum number of status checks
        poll_interval = 5  # Time between polls in seconds
        polls = 0
        while not completed and polls < max_polls:
            # Wait before checking status
            await asyncio.sleep(poll_interval)
            
            # Get execution status
            current_status = client.get_job_status(job_id)

            # TODO:Update job progress based on cloud function status. But how?
            
            
            # Check if function execution is complete
            if current_status.lower() in ["completed", "success"]:
                completed = True
                
                # Get the result from the execution
                client.download_job_files(job_id, settings.datasphere.output_dir)
                
                # TODO: read results and save to db
                
            elif current_status in ["failed", "error"]:
                # Function execution failed
                client.download_job_files(
                    job_id, 
                    settings.datasphere.output_dir,
                    with_diagnostics=True,
                    with_logs=True
                )
                
                # TODO: read results and save to db
                
                # Update job as failed
                update_job_status(
                    job_id,
                    JobStatus.FAILED.value,
                    error_message=error_message
                )
                
                raise RuntimeError(f"Cloud function execution failed: {error_message}")
            
            polls += 1
        
        # If we've reached the maximum number of polls and the function hasn't completed
        if not completed:
            update_job_status(
                job_id,
                JobStatus.FAILED.value,
                error_message="Cloud function execution timed out"
            )
            raise TimeoutError("Cloud function execution timed out")
        
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
        # Clean up local storage
        datasets_dir = os.path.join(settings.datasphere.output_dir, 'datasets')
        if os.path.exists(datasets_dir):
            shutil.rmtree(datasets_dir)
        logger.info(f"Datasets directory cleaned up: {datasets_dir}")
