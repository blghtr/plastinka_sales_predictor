"""
Client for interacting with Yandex Cloud Functions.
Handles function invocation and status tracking.
"""
import os
import json
import uuid
import time
import logging
import requests
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import backoff

from deployment.app.cloud_integration.config.cloud_config import cloud_settings
from deployment.app.db.database import get_db_connection

logger = logging.getLogger(__name__)


class CloudFunctionClient:
    """Client for interacting with Yandex Cloud Functions."""
    
    def __init__(self):
        """Initialize the cloud function client."""
        self.api_gateway_url = cloud_settings.functions.api_gateway_url
        self.api_key = cloud_settings.functions.api_key
        self.callback_url = cloud_settings.callback_url
        self.callback_auth_token = cloud_settings.callback_auth_token
        self.max_retries = cloud_settings.functions.max_retries
        self.request_timeout = cloud_settings.functions.request_timeout
        
        # Ensure API gateway URL is properly formatted
        if self.api_gateway_url and not self.api_gateway_url.endswith('/'):
            self.api_gateway_url += '/'
    
    def invoke_training_function(self, job_id: str, 
                                training_params: Dict[str, Any],
                                storage_paths: Dict[str, str]) -> str:
        """
        Invoke the training cloud function.
        
        Args:
            job_id: Unique job identifier
            training_params: Model training parameters
            storage_paths: Cloud storage paths for input/output
            
        Returns:
            Execution ID for tracking the function
        """
        # Register function in database
        function_id = self._ensure_function_registered('training')
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        # Prepare request payload
        payload = {
            "job_id": job_id,
            "callback_url": self.callback_url,
            "storage_paths": storage_paths,
            "training_params": training_params
        }
        
        # Store execution in database
        self._store_function_execution(
            execution_id=execution_id,
            function_id=function_id,
            job_id=job_id,
            request_payload=payload
        )
        
        # Invoke the function
        try:
            self._invoke_function(
                function_type='training',
                payload=payload,
                execution_id=execution_id
            )
            
            logger.info(f"Invoked training function for job {job_id} with execution ID {execution_id}")
            return execution_id
            
        except Exception as e:
            # Update execution status to failed
            self._update_execution_status(
                execution_id=execution_id,
                status='failed',
                error_message=str(e)
            )
            logger.error(f"Error invoking training function for job {job_id}: {str(e)}")
            raise
    
    def invoke_prediction_function(self, job_id: str, 
                                  prediction_params: Dict[str, Any],
                                  storage_paths: Dict[str, str]) -> str:
        """
        Invoke the prediction cloud function.
        
        Args:
            job_id: Unique job identifier
            prediction_params: Model prediction parameters
            storage_paths: Cloud storage paths for input/output
            
        Returns:
            Execution ID for tracking the function
        """
        # Register function in database
        function_id = self._ensure_function_registered('prediction')
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        # Prepare request payload
        payload = {
            "job_id": job_id,
            "callback_url": self.callback_url,
            "storage_paths": storage_paths,
            "prediction_params": prediction_params
        }
        
        # Store execution in database
        self._store_function_execution(
            execution_id=execution_id,
            function_id=function_id,
            job_id=job_id,
            request_payload=payload
        )
        
        # Invoke the function
        try:
            self._invoke_function(
                function_type='prediction',
                payload=payload,
                execution_id=execution_id
            )
            
            logger.info(f"Invoked prediction function for job {job_id} with execution ID {execution_id}")
            return execution_id
            
        except Exception as e:
            # Update execution status to failed
            self._update_execution_status(
                execution_id=execution_id,
                status='failed',
                error_message=str(e)
            )
            logger.error(f"Error invoking prediction function for job {job_id}: {str(e)}")
            raise
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get the current status of a function execution.
        
        Args:
            execution_id: Function execution ID
            
        Returns:
            Execution status details
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the execution record
        cursor.execute(
            """
            SELECT e.execution_id, e.function_id, e.job_id, e.status, 
                   e.start_time, e.end_time, e.error_message,
                   f.function_type
            FROM cloud_function_executions e
            JOIN cloud_functions f ON e.function_id = f.function_id
            WHERE e.execution_id = ?
            """,
            (execution_id,)
        )
        execution = cursor.fetchone()
        
        if not execution:
            conn.close()
            raise ValueError(f"Execution {execution_id} not found")
        
        # Get the latest status update
        cursor.execute(
            """
            SELECT timestamp, status, progress_percentage, current_step,
                   steps_completed, steps_total, error_code
            FROM cloud_function_status_updates
            WHERE execution_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (execution_id,)
        )
        status_update = cursor.fetchone()
        
        # Build the status object
        status_data = {
            "execution_id": execution[0],
            "function_type": execution[7],
            "job_id": execution[2],
            "status": execution[3],
            "start_time": execution[4],
            "end_time": execution[5],
        }
        
        if status_update:
            status_data.update({
                "latest_update": status_update[0],
                "latest_status": status_update[1],
                "progress": {
                    "percentage": status_update[2],
                    "current_step": status_update[3],
                    "steps_completed": status_update[4],
                    "steps_total": status_update[5]
                }
            })
            
            if status_update[6]:  # Error code
                status_data["error"] = {
                    "code": status_update[6],
                    "message": execution[6]  # Error message from execution record
                }
        elif execution[6]:  # Error message in execution record
            status_data["error"] = {
                "message": execution[6]
            }
        
        conn.close()
        return status_data
    
    def get_execution_logs(self, execution_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get logs for a function execution.
        
        Args:
            execution_id: Function execution ID
            limit: Maximum number of log entries to return
            
        Returns:
            List of log entries
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT timestamp, status, logs
            FROM cloud_function_status_updates
            WHERE execution_id = ? AND logs IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (execution_id, limit)
        )
        
        logs = []
        for row in cursor.fetchall():
            if row[2]:  # logs field
                try:
                    log_entries = json.loads(row[2])
                    for entry in log_entries:
                        entry['update_time'] = row[0]
                        entry['status'] = row[1]
                        logs.append(entry)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse logs for execution {execution_id}")
        
        conn.close()
        return logs[:limit]
    
    def update_status_from_callback(self, execution_id: str, 
                                   callback_data: Dict[str, Any],
                                   auth_token: str) -> bool:
        """
        Update execution status from a callback from the cloud function.
        
        Args:
            execution_id: Function execution ID
            callback_data: Status data from the cloud function
            auth_token: Authentication token for verification
            
        Returns:
            True if update was successful
        """
        # Verify auth token
        if auth_token != self.callback_auth_token:
            logger.warning(f"Invalid auth token in callback for execution {execution_id}")
            return False
        
        # Get required fields
        job_id = callback_data.get("job_id")
        status = callback_data.get("status")
        progress = callback_data.get("progress", {})
        logs = callback_data.get("logs", [])
        error = callback_data.get("error")
        
        # Validate the execution exists and matches the job ID
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT job_id FROM cloud_function_executions WHERE execution_id = ?",
            (execution_id,)
        )
        
        result = cursor.fetchone()
        if not result:
            conn.close()
            logger.warning(f"Execution {execution_id} not found in callback")
            return False
        
        if result[0] != job_id:
            conn.close()
            logger.warning(f"Job ID mismatch in callback: expected {result[0]}, got {job_id}")
            return False
        
        # Store the status update
        timestamp = datetime.now().isoformat()
        
        cursor.execute(
            """
            INSERT INTO cloud_function_status_updates
            (execution_id, timestamp, status, progress_percentage, current_step,
             steps_completed, steps_total, logs, error_code, error_message, error_details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                execution_id,
                timestamp,
                status,
                progress.get("percentage"),
                progress.get("current_step"),
                progress.get("steps_completed"),
                progress.get("steps_total"),
                json.dumps(logs) if logs else None,
                error.get("code") if error else None,
                error.get("message") if error else None,
                json.dumps(error.get("details")) if error and "details" in error else None
            )
        )
        
        # Update the execution status if needed
        if status in ('success', 'error'):
            self._update_execution_status(
                execution_id=execution_id,
                status='completed' if status == 'success' else 'failed',
                error_message=error.get("message") if error else None,
                response_payload=callback_data
            )
        
        # Update job progress in the jobs table
        if progress.get("percentage") is not None:
            cursor.execute(
                """
                UPDATE jobs
                SET progress = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (progress.get("percentage"), timestamp, job_id)
            )
        
        # If function completed, update job status and result
        if status == 'success' and 'result' in callback_data:
            # Update job status to completed
            cursor.execute(
                """
                UPDATE jobs
                SET status = 'completed', updated_at = ?
                WHERE job_id = ? AND status <> 'completed'
                """,
                (timestamp, job_id)
            )
            
            # Handle function-specific result processing
            cursor.execute(
                """
                SELECT function_type 
                FROM cloud_functions f 
                JOIN cloud_function_executions e ON f.function_id = e.function_id
                WHERE e.execution_id = ?
                """,
                (execution_id,)
            )
            
            function_type = cursor.fetchone()[0]
            
            if function_type == 'training':
                self._process_training_result(job_id, callback_data.get("result", {}))
            elif function_type == 'prediction':
                self._process_prediction_result(job_id, callback_data.get("result", {}))
        
        # If function failed, update job status to failed
        elif status == 'error':
            cursor.execute(
                """
                UPDATE jobs
                SET status = 'failed', error_message = ?, updated_at = ?
                WHERE job_id = ? AND status <> 'completed'
                """,
                (error.get("message") if error else "Unknown error", timestamp, job_id)
            )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated status for execution {execution_id} to {status}")
        return True
    
    @backoff.on_exception(backoff.expo,
                         (requests.exceptions.RequestException,),
                         max_tries=3,
                         giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500)
    def _invoke_function(self, function_type: str, payload: Dict[str, Any], 
                        execution_id: str) -> Dict[str, Any]:
        """
        Invoke a cloud function with retry logic.
        
        Args:
            function_type: Type of function ('training' or 'prediction')
            payload: Function parameters
            execution_id: Execution ID for tracking
            
        Returns:
            Function response
        """
        if not self.api_gateway_url:
            raise ValueError("API Gateway URL not configured")
        
        # Get function URL
        function_name = self._get_function_name(function_type)
        function_url = f"{self.api_gateway_url}{function_name}"
        
        headers = {
            "Content-Type": "application/json",
            "X-Execution-ID": execution_id
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Track start time
        start_time = datetime.now().isoformat()
        
        # Update execution status to running
        self._update_execution_status(
            execution_id=execution_id,
            status='running',
            start_time=start_time
        )
        
        # Make the request
        response = requests.post(
            function_url,
            headers=headers,
            json=payload,
            timeout=self.request_timeout
        )
        
        # Handle errors
        response.raise_for_status()
        
        # Parse response
        response_data = response.json()
        
        return response_data
    
    def _ensure_function_registered(self, function_type: str) -> str:
        """
        Ensure the function is registered in the database.
        
        Args:
            function_type: Type of function ('training' or 'prediction')
            
        Returns:
            Function ID
        """
        function_name = self._get_function_name(function_type)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if function exists
        cursor.execute(
            "SELECT function_id FROM cloud_functions WHERE function_type = ? AND function_name = ?",
            (function_type, function_name)
        )
        
        result = cursor.fetchone()
        if result:
            function_id = result[0]
        else:
            # Create new function record
            function_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            cursor.execute(
                """
                INSERT INTO cloud_functions
                (function_id, function_type, function_name, created_at, last_updated, version, status, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    function_id,
                    function_type,
                    function_name,
                    timestamp,
                    timestamp,
                    "1.0",
                    "active",
                    json.dumps({})
                )
            )
            
            conn.commit()
        
        conn.close()
        return function_id
    
    def _store_function_execution(self, execution_id: str, function_id: str,
                                 job_id: str, request_payload: Dict[str, Any]) -> None:
        """
        Store a function execution in the database.
        
        Args:
            execution_id: Unique execution ID
            function_id: Function ID
            job_id: Job ID
            request_payload: Function request payload
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert execution record
        cursor.execute(
            """
            INSERT INTO cloud_function_executions
            (execution_id, function_id, job_id, status, request_payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                execution_id,
                function_id,
                job_id,
                "pending",
                json.dumps(request_payload)
            )
        )
        
        conn.commit()
        conn.close()
    
    def _update_execution_status(self, execution_id: str, status: str,
                               error_message: Optional[str] = None,
                               start_time: Optional[str] = None,
                               end_time: Optional[str] = None,
                               response_payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the status of a function execution.
        
        Args:
            execution_id: Execution ID
            status: New status ('pending', 'running', 'completed', 'failed')
            error_message: Error message if status is 'failed'
            start_time: Execution start time
            end_time: Execution end time
            response_payload: Function response payload
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        update_fields = ["status"]
        update_values = [status]
        
        if error_message is not None:
            update_fields.append("error_message")
            update_values.append(error_message)
        
        if start_time is not None:
            update_fields.append("start_time")
            update_values.append(start_time)
        
        if end_time is not None:
            update_fields.append("end_time")
            update_values.append(end_time)
        elif status in ('completed', 'failed'):
            # Set end_time automatically if not provided
            update_fields.append("end_time")
            update_values.append(datetime.now().isoformat())
        
        if response_payload is not None:
            update_fields.append("response_payload")
            update_values.append(json.dumps(response_payload))
        
        # Build update query
        update_query = f"""
            UPDATE cloud_function_executions
            SET {', '.join(f'{field} = ?' for field in update_fields)}
            WHERE execution_id = ?
        """
        
        # Add execution_id to values
        update_values.append(execution_id)
        
        cursor.execute(update_query, update_values)
        conn.commit()
        conn.close()
    
    def _get_function_name(self, function_type: str) -> str:
        """
        Get the function name for a given type.
        
        Args:
            function_type: Type of function ('training' or 'prediction')
            
        Returns:
            Function name
        """
        if function_type == 'training':
            return cloud_settings.functions.training_function_name
        elif function_type == 'prediction':
            return cloud_settings.functions.prediction_function_name
        else:
            raise ValueError(f"Unknown function type: {function_type}")
    
    def _process_training_result(self, job_id: str, result: Dict[str, Any]) -> None:
        """
        Process a training function result.
        
        Args:
            job_id: Job ID
            result: Training result data
        """
        if not result:
            return
        
        # Extract model info
        model_info = result.get("model_info", {})
        if not model_info:
            return
        
        # Store in training_results table
        conn = get_db_connection()
        cursor = conn.cursor()
        
        result_id = str(uuid.uuid4())
        model_id = model_info.get("model_id", str(uuid.uuid4()))
        
        # Get training details
        training_details = result.get("training_details", {})
        duration = training_details.get("training_time_seconds", 0)
        
        cursor.execute(
            """
            INSERT INTO training_results
            (result_id, job_id, model_id, metrics, parameters, duration)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                job_id,
                model_id,
                json.dumps(model_info.get("metrics", {})),
                json.dumps(model_info.get("hyperparameters", {})),
                duration
            )
        )
        
        # Update job with result ID
        cursor.execute(
            """
            UPDATE jobs
            SET result_id = ?
            WHERE job_id = ?
            """,
            (result_id, job_id)
        )
        
        conn.commit()
        conn.close()
    
    def _process_prediction_result(self, job_id: str, result: Dict[str, Any]) -> None:
        """
        Process a prediction function result.
        
        Args:
            job_id: Job ID
            result: Prediction result data
        """
        if not result:
            return
        
        # Extract prediction info
        prediction_info = result.get("prediction_info", {})
        if not prediction_info:
            return
        
        # Store in prediction_results table
        conn = get_db_connection()
        cursor = conn.cursor()
        
        result_id = str(uuid.uuid4())
        model_id = prediction_info.get("model_id", "")
        storage_path = result.get("storage_path", "")
        
        cursor.execute(
            """
            INSERT INTO prediction_results
            (result_id, job_id, model_id, prediction_date, output_path, summary_metrics)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                job_id,
                model_id,
                datetime.now().isoformat(),
                storage_path,
                json.dumps(prediction_info.get("metrics", {}))
            )
        )
        
        # Update job with result ID
        cursor.execute(
            """
            UPDATE jobs
            SET result_id = ?
            WHERE job_id = ?
            """,
            (result_id, job_id)
        )
        
        conn.commit()
        conn.close()


# Create a global instance of the function client
function_client = CloudFunctionClient() 