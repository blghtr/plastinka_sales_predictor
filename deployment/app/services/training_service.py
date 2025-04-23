from datetime import datetime
import time
import os
import json
import uuid
import asyncio
import logging

from deployment.app.db.database import update_job_status, create_training_result
from deployment.app.models.api_models import JobStatus, TrainingParams
from deployment.app.cloud_integration.client.function_client import CloudFunctionClient
from deployment.app.cloud_integration.client.storage_client import CloudStorageClient

# Import from the original codebase
from plastinka_sales_predictor.training_utils import (
    train_model_wrapper
)

# Set up logger
logger = logging.getLogger(__name__)

async def train_model(job_id: str, params: TrainingParams) -> None:
    """
    Train a model using the specified parameters and Yandex Cloud Functions.
    
    Args:
        job_id: ID of the job
        params: Training parameters
    """
    function_client = CloudFunctionClient()
    storage_client = CloudStorageClient()
    
    try:
        # Update job status to running
        update_job_status(job_id, JobStatus.RUNNING.value, progress=0)
        
        # Prepare dataset for cloud function
        # In a real implementation, we would need to prepare the dataset 
        # and upload it to cloud storage
        input_dataset_path = None
        output_model_path = None
        
        try:
            # Create a dummy dataset file for demonstration
            # In a real implementation, this would use actual dataset files
            tmp_dataset_path = f"/tmp/dataset_{job_id}.csv"
            with open(tmp_dataset_path, 'w') as f:
                f.write("date,value\n")
                # Add sample data
                for i in range(100):
                    f.write(f"2023-01-{i+1:02d},{i*10}\n")
            
            # Upload dataset to cloud storage
            input_dataset_path = storage_client.upload_file(
                file_path=tmp_dataset_path,
                object_type='dataset',
                job_id=job_id,
                metadata={"job_id": job_id}
            )
            
            # Define output path for model
            output_model_path = f"models/model_{job_id}.pt"
            
            update_job_status(job_id, JobStatus.RUNNING.value, progress=10, 
                             error_message="Dataset prepared and uploaded to cloud storage")
            
        except Exception as e:
            raise RuntimeError(f"Failed to prepare dataset: {str(e)}")
        
        # Prepare cloud function parameters
        training_params = {
            "model_type": params.model_type,
            "input_chunk_length": params.input_chunk_length,
            "output_chunk_length": params.output_chunk_length,
            "max_epochs": params.max_epochs,
            "learning_rate": params.learning_rate,
            "batch_size": params.batch_size,
            "additional_params": params.additional_params or {}
        }
        
        # Prepare storage paths for cloud function
        storage_paths = {
            "input": input_dataset_path,
            "output": output_model_path,
            "models": output_model_path
        }
        
        # Invoke cloud function
        execution_id = function_client.invoke_training_function(
            job_id=job_id,
            training_params=training_params,
            storage_paths=storage_paths
        )
        
        update_job_status(job_id, JobStatus.RUNNING.value, progress=20, 
                         error_message=f"Training job submitted to cloud function (execution_id: {execution_id})")
        
        # Poll for function execution completion
        completed = False
        max_polls = 30  # Maximum number of status checks
        poll_interval = 5  # Time between polls in seconds
        polls = 0
        
        while not completed and polls < max_polls:
            # Wait before checking status
            await asyncio.sleep(poll_interval)
            
            # Get execution status
            status_data = function_client.get_execution_status(execution_id)
            current_status = status_data.get("status", "unknown")
            
            # Update job progress based on cloud function status
            progress = status_data.get("progress", {}).get("percentage", 20)
            current_step = status_data.get("progress", {}).get("current_step", "Processing")
            
            # Ensure progress is at least 20%
            progress = max(20, progress)
            
            update_job_status(job_id, JobStatus.RUNNING.value, progress=progress, 
                             error_message=f"Cloud function: {current_step}")
            
            # Check if function execution is complete
            if current_status in ["completed", "success"]:
                completed = True
                
                # Get the result from the execution
                result = status_data.get("result", {})
                
                # Create the result record
                model_info = result.get("model_info", {})
                model_id = model_info.get("model_id", str(uuid.uuid4())[:8])
                metrics = model_info.get("metrics", {})
                training_details = result.get("training_details", {})
                
                # Calculate duration from training details
                start_time = status_data.get("start_time", "")
                end_time = status_data.get("end_time", "")
                
                if start_time and end_time:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    duration = (end_dt - start_dt).total_seconds()
                else:
                    duration = training_details.get("training_time_seconds", 0)
                
                # Create result
                result_id = create_training_result(
                    job_id=job_id,
                    model_id=model_id,
                    metrics=metrics,
                    parameters=params.dict(),
                    duration=int(duration),
                    model_path=result.get("storage_path", "")
                )
                
                # Update job as completed
                update_job_status(
                    job_id,
                    JobStatus.COMPLETED.value,
                    progress=100,
                    result_id=result_id
                )
                
            elif current_status in ["failed", "error"]:
                # Function execution failed
                error_details = status_data.get("error", {})
                error_message = error_details.get("message", "Unknown error in cloud function")
                
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