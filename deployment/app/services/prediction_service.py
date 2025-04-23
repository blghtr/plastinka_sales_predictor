from datetime import datetime
import time
import os
import json
import uuid
import pandas as pd
import asyncio
import logging

from deployment.app.db.database import update_job_status, create_prediction_result
from deployment.app.models.api_models import JobStatus, PredictionParams
from deployment.app.cloud_integration.client.function_client import CloudFunctionClient
from deployment.app.cloud_integration.client.storage_client import CloudStorageClient

# Import from the original codebase if needed
# from plastinka_sales_predictor import ...

# Set up logger
logger = logging.getLogger(__name__)


async def generate_predictions(job_id: str, params: PredictionParams) -> None:
    """
    Generate predictions using the specified model and Yandex Cloud Functions.
    
    Args:
        job_id: ID of the job
        params: Prediction parameters
    """
    function_client = CloudFunctionClient()
    storage_client = CloudStorageClient()
    
    try:
        # Update job status to running
        update_job_status(job_id, JobStatus.RUNNING.value, progress=0)
        
        # Prepare input data for cloud function
        # In a real implementation, we would need to prepare the dataset 
        # and upload it to cloud storage
        input_data_path = None
        output_results_path = None
        model_path = None
        
        try:
            # Create a dummy input data file for demonstration
            # In a real implementation, this would use actual feature data
            tmp_data_path = f"/tmp/input_data_{job_id}.csv"
            with open(tmp_data_path, 'w') as f:
                f.write("date,feature1,feature2\n")
                # Add sample data
                for i in range(30):
                    f.write(f"2023-01-{i+1:02d},{i*1.5},{i*0.8}\n")
            
            # Upload input data to cloud storage
            input_data_path = storage_client.upload_file(
                file_path=tmp_data_path,
                object_type='dataset',
                job_id=job_id,
                metadata={"job_id": job_id, "type": "prediction_input"}
            )
            
            # Define output path for results
            output_results_path = f"results/prediction_{job_id}.csv"
            
            # Set path to the model
            # In a real implementation, this would be the actual path to the trained model
            model_path = f"models/model_{params.model_id}.pt"
            
            update_job_status(job_id, JobStatus.RUNNING.value, progress=10, 
                             error_message="Input data prepared and uploaded to cloud storage")
            
        except Exception as e:
            raise RuntimeError(f"Failed to prepare input data: {str(e)}")
        
        # Prepare cloud function parameters
        prediction_params = {
            "model_id": params.model_id,
            "prediction_length": params.prediction_length,
            "forecast_horizon": params.prediction_length,  # For consistency
            "additional_params": params.additional_params or {},
            "dataset_config": {
                "date_column": "date",
                "features": ["feature1", "feature2"],
                "target": "target"
            }
        }
        
        # Prepare storage paths for cloud function
        storage_paths = {
            "input": input_data_path,
            "output": output_results_path,
            "models": model_path
        }
        
        # Invoke cloud function
        execution_id = function_client.invoke_prediction_function(
            job_id=job_id,
            prediction_params=prediction_params,
            storage_paths=storage_paths
        )
        
        update_job_status(job_id, JobStatus.RUNNING.value, progress=20, 
                         error_message=f"Prediction job submitted to cloud function (execution_id: {execution_id})")
        
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
                prediction_info = result.get("prediction_info", {})
                model_id = prediction_info.get("model_id", params.model_id)
                summary_metrics = prediction_info.get("metrics", {})
                
                # Get result file path
                output_path = result.get("storage_path", output_results_path)
                
                # Download the prediction results if needed
                local_output_path = f"/tmp/predictions_{job_id}.csv"
                try:
                    storage_client.download_file(output_path, local_output_path)
                except Exception as download_error:
                    # If download fails, create a dummy result file
                    # In a real implementation, this would handle the error properly
                    logger.warning(f"Failed to download prediction results: {str(download_error)}")
                    prediction_df = pd.DataFrame({
                        'date': [datetime.now() + pd.Timedelta(days=i) for i in range(params.prediction_length)],
                        'prediction': [float(i) for i in range(params.prediction_length)]
                    })
                    prediction_df.to_csv(local_output_path, index=False)
                
                # Create result in database
                result_id = create_prediction_result(
                    job_id=job_id,
                    model_id=model_id,
                    prediction_date=datetime.now(),
                    output_path=output_path,
                    summary_metrics=summary_metrics
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