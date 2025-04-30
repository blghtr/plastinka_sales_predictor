"""
Yandex Cloud Function for model prediction.

This function handles predictions using trained time series models.
It downloads the model from cloud storage, loads the dataset, generates
predictions, and uploads the results back to cloud storage.
"""
import os
import json
import time
import uuid
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import boto3
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prediction_function')


class PredictionFunction:
    """
    Cloud function handler for model prediction.
    """

    def __init__(self):
        """Initialize the prediction function."""
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
            "steps_total": 4  # Total number of steps in prediction process
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
        prediction_params = event.get('prediction_params', {})
        
        # Validate required parameters
        if not job_id or not callback_url or not storage_paths or not prediction_params:
            error = {
                "code": "INVALID_INPUT",
                "message": "Missing required parameters",
                "details": {
                    "job_id": job_id is not None,
                    "callback_url": callback_url is not None,
                    "storage_paths": bool(storage_paths),
                    "prediction_params": bool(prediction_params)
                }
            }
            return self._create_error_response(job_id, error)
        
        try:
            # Send initial status update
            self._send_status_update(job_id, callback_url, "running")
            
            # Process prediction request
            result = self._process_prediction_request(job_id, storage_paths, prediction_params)
            
            # Send final status update
            self._send_status_update(job_id, callback_url, "success", result=result)
            
            # Return success response
            return {
                "job_id": job_id,
                "status": "success",
                "result": result
            }
            
        except Exception as e:
            logger.exception(f"Error processing prediction request for job {job_id}")
            
            # Create error details
            error = {
                "code": "PREDICTION_ERROR",
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
    
    def _process_prediction_request(self, job_id: str, storage_paths: Dict[str, str], 
                                   prediction_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the prediction request.
        
        Args:
            job_id: Job ID
            storage_paths: Cloud storage paths
            prediction_params: Prediction parameters
            
        Returns:
            Prediction result data
        """
        # Extract paths
        input_path = storage_paths.get('input')
        output_path = storage_paths.get('output')
        models_path = storage_paths.get('models')
        
        if not input_path or not output_path or not models_path:
            raise ValueError("Missing required storage paths 'input', 'output', or 'models'")
        
        # Extract model ID
        model_id = prediction_params.get('model_id')
        if not model_id:
            raise ValueError("Missing required parameter 'model_id'")
        
        # Step 1: Download model
        self._update_progress(1, 4, "Loading model")
        model = self._load_model(models_path)
        
        # Step 2: Download dataset
        self._update_progress(2, 4, "Loading dataset")
        dataset = self._load_dataset(input_path, prediction_params)
        
        # Step 3: Generate predictions
        self._update_progress(3, 4, "Generating predictions")
        predictions, metrics = self._generate_predictions(model, dataset, prediction_params)
        
        # Step 4: Save and upload results
        self._update_progress(4, 4, "Saving results")
        result_path = self._save_results(predictions, output_path, prediction_params)
        
        # Prepare result
        timestamp = datetime.now().isoformat()
        
        # Extract forecast horizon
        forecast_horizon = prediction_params.get('forecast_horizon', 0)
        
        # Determine prediction date range
        dataset_config = prediction_params.get('dataset_config', {})
        date_column = dataset_config.get('date_column')
        
        # Calculate prediction start/end dates
        if dataset is not None and date_column and len(dataset) > 0:
            # Get the last date from the dataset
            history_end = dataset[date_column].max()
            if isinstance(history_end, str):
                history_end = pd.to_datetime(history_end)
            
            # Prediction starts after the last history date
            prediction_start = (history_end + timedelta(days=1)).isoformat()
            
            # Prediction ends after forecast_horizon periods
            # This is simplified - in reality, the period increment would depend on data frequency
            prediction_end = (history_end + timedelta(days=forecast_horizon)).isoformat()
        else:
            prediction_start = timestamp
            prediction_end = (pd.to_datetime(timestamp) + timedelta(days=forecast_horizon)).isoformat()
        
        # Build prediction info
        prediction_info = {
            "model_id": model_id,
            "created_at": timestamp,
            "forecast_horizon": forecast_horizon,
            "prediction_start": prediction_start,
            "prediction_end": prediction_end,
            "metrics": metrics
        }
        
        # Build prediction details
        prediction_details = {
            "prediction_time_seconds": int(time.time() - self.start_time),
            "data_points": len(predictions) if isinstance(predictions, pd.DataFrame) else 0,
            "feature_importance": self._get_feature_importance(model, prediction_params)
        }
        
        # Return result
        return {
            "storage_path": result_path,
            "prediction_info": prediction_info,
            "prediction_details": prediction_details
        }
    
    def _load_model(self, models_path: str) -> Any:
        """
        Load model from cloud storage.
        
        Args:
            models_path: Path to model in cloud storage
            
        Returns:
            Loaded model
        """
        try:
            # Create local temp file
            local_path = '/tmp/model.pt'
            
            # Download file
            self.s3_client.download_file(
                Bucket=self.storage_bucket,
                Key=models_path,
                Filename=local_path
            )
            
            # In a real function, this would deserialize the actual model
            # For now, we load a JSON representation
            with open(local_path, 'r') as f:
                model = json.load(f)
            
            logger.info(f"Loaded model from {models_path}")
            return model
            
        except Exception as e:
            logger.exception(f"Error loading model from {models_path}")
            
            # Enhance error with specific error code
            error = ValueError(f"Failed to load model: {str(e)}")
            error.code = "MODEL_ERROR"
            raise error
    
    def _load_dataset(self, input_path: str, prediction_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Load dataset from cloud storage.
        
        Args:
            input_path: Path to dataset in cloud storage
            prediction_params: Prediction parameters
            
        Returns:
            DataFrame containing the dataset
        """
        try:
            # Create local temp file
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
            
            # Process dataset based on configuration
            dataset_config = prediction_params.get('dataset_config', {})
            date_column = dataset_config.get('date_column')
            
            if date_column:
                # Convert date column to datetime
                df[date_column] = pd.to_datetime(df[date_column])
                
                # Filter by date range if specified
                history_start = dataset_config.get('history_start')
                history_end = dataset_config.get('history_end')
                
                if history_start:
                    df = df[df[date_column] >= pd.to_datetime(history_start)]
                
                if history_end:
                    df = df[df[date_column] <= pd.to_datetime(history_end)]
                
                # Sort by date
                df = df.sort_values(by=date_column).reset_index(drop=True)
            
            logger.info(f"Loaded dataset from {input_path} with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.exception(f"Error loading dataset from {input_path}")
            
            # Enhance error with specific error code
            error = ValueError(f"Failed to load dataset: {str(e)}")
            error.code = "DATA_ERROR"
            raise error
    
    def _generate_predictions(self, model: Any, dataset: pd.DataFrame, 
                             prediction_params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate predictions using the model.
        
        Args:
            model: Loaded model
            dataset: Input dataset
            prediction_params: Prediction parameters
            
        Returns:
            Tuple of (predictions DataFrame, metrics dict)
        """
        # This is a placeholder implementation
        # In a real function, this would use the actual model to generate predictions
        
        # Start timing
        self.start_time = time.time()
        
        # Get forecast horizon
        forecast_horizon = prediction_params.get('forecast_horizon', 30)
        
        # Get dataset configuration
        dataset_config = prediction_params.get('dataset_config', {})
        target_column = dataset_config.get('target_column')
        date_column = dataset_config.get('date_column')
        
        if not target_column or not date_column:
            raise ValueError("Missing required dataset_config parameters 'target_column' or 'date_column'")
        
        # Get prediction configuration
        prediction_config = prediction_params.get('prediction_config', {})
        include_history = prediction_config.get('include_history', False)
        return_intervals = prediction_config.get('return_intervals', False)
        interval_width = prediction_config.get('interval_width', 0.95)
        
        # Get the last date in the dataset
        last_date = dataset[date_column].max()
        
        # Create forecast dates
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq='D'  # Assuming daily data - would be determined from data in practice
        )
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({date_column: forecast_dates})
        
        # Generate mock predictions
        last_values = dataset[target_column].iloc[-5:].values
        trend = np.mean(np.diff(last_values))
        base = last_values[-1]
        
        # Generate forecast with some randomness and trend
        forecast = np.array([
            base + trend * i + np.random.normal(0, base * 0.05) 
            for i in range(1, forecast_horizon + 1)
        ])
        predictions[target_column] = forecast
        
        # Add prediction intervals if requested
        if return_intervals:
            # Calculate confidence intervals (simplified)
            std_dev = np.std(last_values) * 1.5  # Increased for forecasting uncertainty
            z_score = 1.96  # Approximately for 95% confidence
            if interval_width != 0.95:
                # Adjust z-score for different confidence levels (simplified)
                z_score = z_score * interval_width / 0.95
            
            # Add lower and upper bounds
            predictions[f"{target_column}_lower"] = predictions[target_column] - z_score * std_dev
            predictions[f"{target_column}_upper"] = predictions[target_column] + z_score * std_dev
        
        # Include history if requested
        if include_history:
            history = dataset[[date_column, target_column]].copy()
            predictions = pd.concat([history, predictions], ignore_index=True)
        
        # Calculate evaluation metrics (if historical data is used for prediction window)
        metrics = {
            "mae": np.random.uniform(0.1, 0.6),
            "rmse": np.random.uniform(0.2, 1.2),
            "mape": np.random.uniform(7.0, 25.0)
        }
        
        # Simulate computation time
        time.sleep(forecast_horizon * 0.02)  # More horizons take more time
        
        logger.info(f"Generated predictions for {forecast_horizon} periods")
        
        return predictions, metrics
    
    def _save_results(self, predictions: pd.DataFrame, output_path: str, 
                     prediction_params: Dict[str, Any]) -> str:
        """
        Save prediction results to cloud storage.
        
        Args:
            predictions: Prediction results
            output_path: Base path for output in cloud storage
            prediction_params: Prediction parameters
            
        Returns:
            Path to the saved results in cloud storage
        """
        # Ensure output_path includes file extension
        if not any(output_path.endswith(ext) for ext in ['.csv', '.json', '.parquet']):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = prediction_params.get('model_id', 'unknown')
            output_path = f"{output_path}/prediction_{model_id}_{timestamp}.csv"
        
        # Create local results file
        local_path = '/tmp/predictions.csv'
        
        # Save predictions to local file
        predictions.to_csv(local_path, index=False)
        
        # Upload to cloud storage
        try:
            self.s3_client.upload_file(
                Filename=local_path,
                Bucket=self.storage_bucket,
                Key=output_path,
                ExtraArgs={'ContentType': 'text/csv'}
            )
            
            logger.info(f"Saved prediction results to {output_path}")
            return output_path
            
        except Exception as e:
            logger.exception(f"Error saving prediction results to {output_path}")
            
            # Enhance error with specific error code
            error = ValueError(f"Failed to save prediction results: {str(e)}")
            error.code = "STORAGE_ERROR"
            raise error
    
    def _get_feature_importance(self, model: Any, prediction_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Get feature importance for the model (placeholder implementation).
        
        Args:
            model: Loaded model
            prediction_params: Prediction parameters
            
        Returns:
            Dict mapping feature names to importance scores
        """
        # This is a placeholder implementation
        # In a real function, this would extract actual feature importance from the model
        
        dataset_config = prediction_params.get('dataset_config', {})
        features = dataset_config.get('features', [])
        
        if not features:
            return {}
        
        # Generate random importance scores that sum to 1
        importance_values = np.random.dirichlet(np.ones(len(features)))
        
        return {feature: float(importance) for feature, importance in zip(features, importance_values)}
    
    def _update_progress(self, steps_completed: int, steps_total: int, 
                        current_step: str) -> None:
        """
        Update progress tracking.
        
        Args:
            steps_completed: Number of steps completed
            steps_total: Total number of steps
            current_step: Description of current step
        """
        percentage = min(100.0, (steps_completed / steps_total) * 100)
        
        self.progress = {
            "percentage": percentage,
            "current_step": current_step,
            "steps_completed": steps_completed,
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
                "X-Function-Type": "prediction",
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
prediction_function = PredictionFunction()

# Entry point for cloud function
def handler(event, context):
    return prediction_function.handle(event, context) 