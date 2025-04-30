"""
Yandex Cloud Function for model training.

This function handles training of time series models for sales prediction.
It downloads the dataset from cloud storage, trains the model, and uploads
the model back to cloud storage.
"""
import os
import json
import time
import uuid
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import boto3
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('training_function')


class TrainingFunction:
    """
    Cloud function handler for model training.
    """

    def __init__(self):
        """Initialize the training function."""
        self.api_endpoint = os.environ.get('API_ENDPOINT', '')
        self.api_key = os.environ.get('API_KEY', '')
        self.storage_bucket = os.environ.get('STORAGE_BUCKET', '')
        self.storage_access_key = os.environ.get('STORAGE_ACCESS_KEY', '')
        self.storage_secret_key = os.environ.get('STORAGE_SECRET_KEY', '')
        self.storage_endpoint = os.environ.get('STORAGE_ENDPOINT', 'https://storage.yandexcloud.net')
        self.storage_region = os.environ.get('STORAGE_REGION', 'ru-central1')
        
        # Set log level from environment variable
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        logger.setLevel(getattr(logging, log_level))
        
        # Initialize progress tracking
        self.progress = {
            "percentage": 0,
            "current_step": "Initializing",
            "steps_completed": 0,
            "steps_total": 5  # Total number of steps in training process
        }
        
        # Initialize S3 client
        self._init_s3_client()
    
    def _init_s3_client(self):
        """Initialize the S3 client for cloud storage access."""
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.storage_endpoint,
            aws_access_key_id=self.storage_access_key,
            aws_secret_access_key=self.storage_secret_key,
            region_name=self.storage_region
        )
    
    def handle(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Main function handler.
        
        Args:
            event: Event data containing job parameters
            context: Function context
            
        Returns:
            Dict containing job result
        """
        job_id = event.get('job_id')
        callback_url = event.get('callback_url')
        storage_paths = event.get('storage_paths', {})
        training_params = event.get('training_params', {})
        
        # Validate required parameters
        if not job_id or not callback_url or not storage_paths or not training_params:
            error = {
                "code": "INVALID_INPUT",
                "message": "Missing required parameters",
                "details": {
                    "job_id": job_id is not None,
                    "callback_url": callback_url is not None,
                    "storage_paths": bool(storage_paths),
                    "training_params": bool(training_params)
                }
            }
            return self._create_error_response(job_id, error)
        
        try:
            # Send initial status update
            self._send_status_update(job_id, callback_url, "running")
            
            # Process training request
            result = self._process_training_request(job_id, storage_paths, training_params)
            
            # Send final status update
            self._send_status_update(job_id, callback_url, "success", result=result)
            
            # Return success response
            return {
                "job_id": job_id,
                "status": "success",
                "result": result
            }
            
        except Exception as e:
            logger.exception(f"Error processing training request for job {job_id}")
            
            # Create error details
            error = {
                "code": "TRAINING_ERROR",
                "message": str(e),
                "details": {
                    "exception_type": type(e).__name__,
                    "traceback": logger.handlers[0].formatter.formatException()
                }
            }
            
            # Send error status update
            self._send_status_update(job_id, callback_url, "error", error=error)
            
            # Return error response
            return self._create_error_response(job_id, error)
    
    def _process_training_request(self, job_id: str, storage_paths: Dict[str, str], 
                                 training_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the training request.
        
        Args:
            job_id: Job ID
            storage_paths: Cloud storage paths
            training_params: Training parameters
            
        Returns:
            Training result data
        """
        # Extract paths
        input_path = storage_paths.get('input')
        output_path = storage_paths.get('output')
        models_path = storage_paths.get('models')
        
        if not input_path or not output_path:
            raise ValueError("Missing required storage paths 'input' or 'output'")
        
        # Step 1: Download dataset
        self._update_progress(1, 5, "Downloading dataset")
        dataset = self._download_dataset(input_path)
        
        # Step 2: Prepare data
        self._update_progress(2, 5, "Preparing data")
        train_data, val_data = self._prepare_data(dataset, training_params)
        
        # Step 3: Train model
        self._update_progress(3, 5, "Training model")
        model, training_details = self._train_model(train_data, val_data, training_params)
        
        # Step 4: Evaluate model
        self._update_progress(4, 5, "Evaluating model")
        metrics = self._evaluate_model(model, val_data, training_params)
        
        # Step 5: Save and upload model
        self._update_progress(5, 5, "Saving model")
        model_path = self._save_model(model, models_path, training_params)
        
        # Prepare result
        model_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Build model info
        model_info = {
            "model_id": model_id,
            "model_type": training_params.get('model_type'),
            "created_at": timestamp,
            "input_features": training_params.get('dataset_config', {}).get('features', []),
            "hyperparameters": training_params.get('hyperparameters', {}),
            "metrics": metrics
        }
        
        # Return result
        return {
            "storage_path": model_path,
            "model_info": model_info,
            "training_details": training_details
        }
    
    def _download_dataset(self, input_path: str) -> pd.DataFrame:
        """
        Download dataset from cloud storage.
        
        Args:
            input_path: Path to dataset in cloud storage
            
        Returns:
            DataFrame containing the dataset
        """
        try:
            # Create local temp directory
            local_path = '/tmp/dataset.csv'
            
            # Download file
            self.s3_client.download_file(
                Bucket=self.storage_bucket,
                Key=input_path,
                Filename=local_path
            )
            
            # Read dataset based on file extension
            if input_path.endswith('.csv'):
                df = pd.read_csv(local_path)
            elif input_path.endswith('.parquet'):
                df = pd.read_parquet(local_path)
            elif input_path.endswith('.json'):
                df = pd.read_json(local_path)
            else:
                raise ValueError(f"Unsupported file format for {input_path}")
            
            logger.info(f"Downloaded dataset from {input_path} with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.exception(f"Error downloading dataset from {input_path}")
            
            # Enhance error with specific error code
            error = ValueError(f"Failed to download dataset: {str(e)}")
            error.code = "STORAGE_ERROR"
            raise error
    
    def _prepare_data(self, dataset: pd.DataFrame, training_params: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Prepare data for training.
        
        Args:
            dataset: Input dataset
            training_params: Training parameters
            
        Returns:
            Tuple of (train_data, val_data)
        """
        # This is a simplified implementation
        # In a real function, this would implement time series specific processing
        
        dataset_config = training_params.get('dataset_config', {})
        
        # Get target and feature columns
        target_column = dataset_config.get('target_column')
        features = dataset_config.get('features', [])
        date_column = dataset_config.get('date_column')
        
        if not target_column or not date_column:
            raise ValueError("Missing required dataset_config parameters 'target_column' or 'date_column'")
        
        # Convert date column to datetime
        dataset[date_column] = pd.to_datetime(dataset[date_column])
        
        # Sort by date
        dataset = dataset.sort_values(by=date_column).reset_index(drop=True)
        
        # Get train/val split dates
        train_start = pd.to_datetime(dataset_config.get('train_start'))
        train_end = pd.to_datetime(dataset_config.get('train_end'))
        val_start = pd.to_datetime(dataset_config.get('val_start', train_end))
        val_end = pd.to_datetime(dataset_config.get('val_end', None))
        
        # Split data
        train_mask = (dataset[date_column] >= train_start) & (dataset[date_column] <= train_end)
        train_data = dataset[train_mask].copy()
        
        if val_end is not None:
            val_mask = (dataset[date_column] > train_end) & (dataset[date_column] <= val_end)
        else:
            val_mask = dataset[date_column] > train_end
        
        val_data = dataset[val_mask].copy() if val_mask.any() else None
        
        logger.info(f"Prepared data with {len(train_data)} training samples and "
                  f"{len(val_data) if val_data is not None else 0} validation samples")
        
        # In a real function, this would return appropriate data structures for the model
        # For simplicity, we're just returning the DataFrames
        return train_data, val_data
    
    def _train_model(self, train_data: Any, val_data: Any, 
                    training_params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Train the model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            training_params: Training parameters
            
        Returns:
            Tuple of (model, training_details)
        """
        # This is a placeholder implementation
        # In a real function, this would implement actual model training
        
        model_type = training_params.get('model_type')
        hyperparameters = training_params.get('hyperparameters', {})
        training_config = training_params.get('training_config', {})
        
        # Extract training parameters
        epochs = training_config.get('epochs', 10)
        batch_size = training_config.get('batch_size', 32)
        early_stopping = training_config.get('early_stopping', False)
        patience = training_config.get('patience', 3)
        
        # Simulate training process
        start_time = time.time()
        
        # Create mock training results
        mock_epochs_completed = epochs if not early_stopping else min(epochs, np.random.randint(5, epochs))
        mock_best_epoch = np.random.randint(1, mock_epochs_completed)
        
        # Simulate training process with progress updates
        for epoch in range(mock_epochs_completed):
            # Update progress within the training step
            progress_in_training = (epoch + 1) / epochs
            overall_progress = 2 + progress_in_training  # Step 3 (training) progress
            self._update_progress(
                overall_progress, 5, 
                f"Training model - Epoch {epoch + 1}/{epochs}"
            )
            time.sleep(0.2)  # Simulate computation time
        
        # Create model (placeholder)
        model = {"type": model_type, "params": hyperparameters}
        
        # Training details
        training_details = {
            "epochs_completed": mock_epochs_completed,
            "training_time_seconds": int(time.time() - start_time),
            "early_stopping_triggered": mock_epochs_completed < epochs,
            "best_epoch": mock_best_epoch
        }
        
        logger.info(f"Trained {model_type} model in {training_details['training_time_seconds']} seconds")
        
        return model, training_details
    
    def _evaluate_model(self, model: Any, val_data: Any, 
                       training_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            model: Trained model
            val_data: Validation data
            training_params: Training parameters
            
        Returns:
            Evaluation metrics
        """
        # This is a placeholder implementation
        # In a real function, this would implement actual model evaluation
        
        # Create mock metrics
        metrics = {
            "training_loss": np.random.uniform(0.01, 0.1),
            "validation_loss": np.random.uniform(0.05, 0.2),
            "training_metrics": {
                "mae": np.random.uniform(0.05, 0.5),
                "rmse": np.random.uniform(0.1, 1.0),
                "mape": np.random.uniform(5.0, 20.0)
            }
        }
        
        if val_data is not None:
            metrics["validation_metrics"] = {
                "mae": np.random.uniform(0.1, 0.6),
                "rmse": np.random.uniform(0.2, 1.2),
                "mape": np.random.uniform(7.0, 25.0)
            }
        
        logger.info(f"Evaluated model with validation RMSE: {metrics.get('validation_metrics', {}).get('rmse', 'N/A')}")
        
        return metrics
    
    def _save_model(self, model: Any, models_path: Optional[str], 
                   training_params: Dict[str, Any]) -> str:
        """
        Save and upload the model to cloud storage.
        
        Args:
            model: Trained model
            models_path: Base path for models in cloud storage
            training_params: Training parameters
            
        Returns:
            Path to the saved model in cloud storage
        """
        # Determine model path
        if not models_path:
            # If no models path provided, create one
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = training_params.get('model_type', 'default')
            models_path = f"models/{model_type}_{timestamp}"
        
        # Ensure models_path ends with .pt or .pth
        if not (models_path.endswith('.pt') or models_path.endswith('.pth')):
            models_path = f"{models_path}.pt"
        
        # Save model locally
        local_path = '/tmp/model.pt'
        
        # In a real function, this would serialize the actual model
        # For now, we save a JSON representation
        with open(local_path, 'w') as f:
            json.dump(model, f)
        
        # Upload to cloud storage
        try:
            self.s3_client.upload_file(
                Filename=local_path,
                Bucket=self.storage_bucket,
                Key=models_path,
                ExtraArgs={'ContentType': 'application/octet-stream'}
            )
            
            logger.info(f"Saved model to {models_path}")
            return models_path
            
        except Exception as e:
            logger.exception(f"Error uploading model to {models_path}")
            
            # Enhance error with specific error code
            error = ValueError(f"Failed to upload model: {str(e)}")
            error.code = "STORAGE_ERROR"
            raise error
    
    def _update_progress(self, steps_completed: float, steps_total: int, 
                        current_step: str) -> None:
        """
        Update progress tracking.
        
        Args:
            steps_completed: Number of steps completed (can be fractional)
            steps_total: Total number of steps
            current_step: Description of current step
        """
        percentage = min(100.0, (steps_completed / steps_total) * 100)
        
        self.progress = {
            "percentage": percentage,
            "current_step": current_step,
            "steps_completed": int(steps_completed),
            "steps_total": steps_total
        }
        
        logger.info(f"Progress: {percentage:.1f}% - {current_step}")
    
    def _send_status_update(self, job_id: str, callback_url: str, status: str,
                           result: Optional[Dict[str, Any]] = None,
                           error: Optional[Dict[str, Any]] = None) -> None:
        """
        Send status update to the callback URL.
        
        Args:
            job_id: Job ID
            callback_url: Callback URL
            status: Status ('running', 'success', or 'error')
            result: Result data (for success status)
            error: Error data (for error status)
        """
        if not callback_url:
            logger.warning(f"No callback URL provided for job {job_id}")
            return
        
        # Prepare update payload
        payload = {
            "job_id": job_id,
            "status": status,
            "progress": self.progress,
            "logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": f"Status update: {status}"
                }
            ]
        }
        
        # Add result or error if provided
        if status == "success" and result:
            payload["result"] = result
        elif status == "error" and error:
            payload["error"] = error
        
        # Send the request
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Function-Type": "training",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(
                callback_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to send status update to {callback_url}: "
                             f"Status code {response.status_code}")
                logger.debug(f"Response content: {response.text}")
            else:
                logger.info(f"Sent {status} status update to {callback_url}")
                
        except Exception as e:
            logger.warning(f"Error sending status update to {callback_url}: {str(e)}")
    
    def _create_error_response(self, job_id: Optional[str], error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an error response.
        
        Args:
            job_id: Job ID
            error: Error details
            
        Returns:
            Error response object
        """
        return {
            "job_id": job_id or "unknown",
            "status": "error",
            "error": error
        }


# Initialize the function handler
training_function = TrainingFunction()

# Entry point for cloud function
def handler(event, context):
    return training_function.handle(event, context) 