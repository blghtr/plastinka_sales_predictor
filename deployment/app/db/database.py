import sqlite3
import json
import uuid
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from contextlib import contextmanager
import hashlib

# from app.config import settings
from deployment.app.config import settings # Corrected absolute import

logger = logging.getLogger("plastinka.database")

# Use database path from settings
DB_PATH = settings.db.path

class DatabaseError(Exception):
    """Exception raised for database errors."""
    def __init__(self, message: str, query: str = None, params: tuple = None, original_error: Exception = None):
        self.message = message
        self.query = query
        self.params = params
        self.original_error = original_error
        super().__init__(self.message)

def get_db_connection():
    """Get a connection to the SQLite database"""
    try:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}", exc_info=True)
        raise DatabaseError(f"Database connection failed: {str(e)}", original_error=e)

def dict_factory(cursor, row):
    """Convert row to dictionary"""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

def execute_query(query: str, params: tuple = (), fetchall: bool = False, connection: sqlite3.Connection = None) -> Union[List[Dict], Dict, None]:
    """
    Execute a query and optionally return results
    
    Args:
        query: SQL query with placeholders (?, :name)
        params: Parameters for the query
        fetchall: Whether to fetch all results or just one
        connection: Optional existing database connection to use
        
    Returns:
        Query results as dict or list of dicts, or None for operations
        
    Raises:
        DatabaseError: If database operation fails
    """
    conn = None
    cursor = None
    conn_created = False
    
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        
        cursor.execute(query, params)
        
        if query.strip().upper().startswith(("SELECT", "PRAGMA")):
            if fetchall:
                result = cursor.fetchall()
            else:
                result = cursor.fetchone()
        else:
            conn.commit()
            result = None
            
        return result
        
    except sqlite3.Error as e:
        if conn and conn_created:
            conn.rollback()
        
        # Log with limited parameter data for security
        safe_params = "..." if params else "()"
        logger.error(f"Database error in query: {query[:100]} with params: {safe_params}: {str(e)}", exc_info=True)
        
        raise DatabaseError(
            message=f"Database operation failed: {str(e)}",
            query=query,
            params=params,
            original_error=e
        )
    finally:
        if cursor:
            cursor.close()
        if conn and conn_created:
            conn.close()

def execute_many(query: str, params_list: List[tuple], connection: sqlite3.Connection = None) -> None:
    """
    Execute a query with multiple parameter sets
    
    Args:
        query: SQL query with placeholders
        params_list: List of parameter tuples
        connection: Optional existing database connection to use
        
    Raises:
        DatabaseError: If database operation fails
    """
    if not params_list:
        return
    
    conn = None
    cursor = None
    conn_created = False
    
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        cursor = conn.cursor()
        
        cursor.executemany(query, params_list)
        conn.commit()
        
    except sqlite3.Error as e:
        if conn and conn_created:
            conn.rollback()
        
        logger.error(f"Database error in executemany: {query[:100]}, params count: {len(params_list)}: {str(e)}", exc_info=True)
        
        raise DatabaseError(
            message=f"Batch database operation failed: {str(e)}",
            query=query,
            original_error=e
        )
    finally:
        if cursor:
            cursor.close()
        if conn and conn_created:
            conn.close()

# Job-related database functions

def generate_id() -> str:
    """Generate a unique ID for jobs or results"""
    return str(uuid.uuid4())

def create_job(job_type: str, parameters: Dict[str, Any] = None, connection: sqlite3.Connection = None) -> str:
    """
    Create a new job record and return the job ID
    
    Args:
        job_type: Type of job (from JobType enum)
        parameters: Dictionary of job parameters
        connection: Optional existing database connection to use
        
    Returns:
        Generated job ID
    """
    job_id = generate_id()
    now = datetime.now().isoformat()
    
    query = """
    INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, parameters, progress)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    params = (
        job_id,
        job_type,
        'pending',
        now,
        now,
        json.dumps(parameters) if parameters else None,
        0
    )
    
    try:
        execute_query(query, params, connection=connection)
        logger.info(f"Created new job: {job_id} of type {job_type}")
        return job_id
    except DatabaseError as e:
        logger.error(f"Failed to create job: {str(e)}")
        raise

def update_job_status(job_id: str, status: str, progress: float = None, 
                      result_id: str = None, error_message: str = None,
                      connection: sqlite3.Connection = None) -> None:
    """
    Update job status and related fields
    
    Args:
        job_id: ID of the job to update
        status: New job status
        progress: Optional progress value (0-100)
        result_id: Optional result ID if job completed
        error_message: Optional error message if job failed
        connection: Optional existing database connection to use
    """
    now = datetime.now().isoformat()
    
    # Build query and parameters list
    query_parts = ["UPDATE jobs SET updated_at = ?"]
    params = [now]
    
    if status:
        query_parts.append("status = ?")
        params.append(status)
    
    if progress is not None:
        query_parts.append("progress = ?")
        params.append(progress)
    
    if result_id:
        query_parts.append("result_id = ?")
        params.append(result_id)
    
    if error_message:
        query_parts.append("error_message = ?")
        params.append(error_message)
    
    query = ", ".join(query_parts) + " WHERE job_id = ?"
    params.append(job_id)
    
    try:
        execute_query(query, tuple(params), connection=connection)
        logger.info(f"Updated job {job_id}: status={status}, progress={progress}")
    except DatabaseError as e:
        logger.error(f"Failed to update job {job_id}: {str(e)}")
        raise

def get_job(job_id: str, connection: sqlite3.Connection = None) -> Dict:
    """
    Get job details by ID
    
    Args:
        job_id: Job ID to retrieve
        connection: Optional existing database connection to use
        
    Returns:
        Job details dictionary or None if not found
    """
    query = "SELECT * FROM jobs WHERE job_id = ?"
    try:
        result = execute_query(query, (job_id,), connection=connection)
        return result
    except DatabaseError as e:
        logger.error(f"Failed to get job {job_id}: {str(e)}")
        raise

def list_jobs(job_type: str = None, status: str = None, limit: int = 100, connection: sqlite3.Connection = None) -> List[Dict]:
    """
    List jobs with optional filters
    
    Args:
        job_type: Optional job type filter
        status: Optional status filter
        limit: Maximum number of jobs to return
        connection: Optional existing database connection to use
        
    Returns:
        List of job dictionaries
    """
    query = "SELECT * FROM jobs WHERE 1=1"
    params = []
    
    if job_type:
        query += " AND job_type = ?"
        params.append(job_type)
    
    if status:
        query += " AND status = ?"
        params.append(status)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    try:
        results = execute_query(query, tuple(params), fetchall=True, connection=connection)
        return results or []
    except DatabaseError as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise

# Result-related functions

def create_data_upload_result(job_id: str, records_processed: int, 
                              features_generated: List[str], 
                              processing_run_id: int, 
                              connection: sqlite3.Connection = None) -> str:
    """
    Create a data upload result record
    
    Args:
        job_id: Associated job ID
        records_processed: Number of records processed
        features_generated: List of feature types generated
        processing_run_id: ID of the processing run
        connection: Optional existing database connection to use
        
    Returns:
        Generated result ID
    """
    result_id = generate_id()
    
    query = """
    INSERT INTO data_upload_results (result_id, job_id, records_processed, 
                                    features_generated, processing_run_id)
    VALUES (?, ?, ?, ?, ?)
    """
    
    params = (
        result_id, 
        job_id,
        records_processed,
        json.dumps(features_generated),
        processing_run_id
    )
    
    try:
        execute_query(query, params, connection=connection)
        return result_id
    except DatabaseError:
        logger.error(f"Failed to create data upload result for job {job_id}")
        raise

def create_or_get_parameter_set(
    parameters_dict: Dict[str, Any],
    is_active: bool = False
) -> str:
    """
    Creates a parameter set record if it doesn't exist, based on a hash of the parameters.
    If `is_active` is True, this set will be marked active, deactivating others.
    Returns the parameter_set_id.
    
    Args:
        parameters_dict: Dictionary of parameters
        is_active: Whether to explicitly set this parameter set as active
        
    Returns:
        Parameter set ID (hash)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create a stable JSON string representation for hashing
    params_json = json.dumps(parameters_dict, sort_keys=True, ensure_ascii=False)
    parameter_set_id = hashlib.sha256(params_json.encode('utf-8')).hexdigest()
    
    now = datetime.utcnow()
    
    try:
        cursor.execute(
            "SELECT parameter_set_id FROM parameter_sets WHERE parameter_set_id = ?",
            (parameter_set_id,)
        )
        existing = cursor.fetchone()
        
        if not existing:
            # New parameter set
            cursor.execute(
                """
                INSERT INTO parameter_sets (parameter_set_id, parameters, created_at, is_active)
                VALUES (?, ?, ?, ?)
                """,
                (parameter_set_id, params_json, now, is_active) # Set active status directly if requested
            )
            logger.info(f"Created new parameter set: {parameter_set_id}")
            
            # If created as active, ensure others are deactivated
            if is_active:
                cursor.execute(
                    "UPDATE parameter_sets SET is_active = 0 WHERE parameter_set_id != ?",
                    (parameter_set_id,)
                )
                logger.info(f"Parameter set {parameter_set_id} created as active.")
        
        # If it exists and is_active is True, make it active
        elif is_active:
            # Need to check if it's *already* active to avoid unnecessary updates
            cursor.execute(
                "SELECT is_active FROM parameter_sets WHERE parameter_set_id = ?",
                (parameter_set_id,)
            )
            current_status = cursor.fetchone()
            if not current_status or not current_status[0]: # If not found or not active
                set_parameter_set_active(parameter_set_id, connection=conn)
            
        conn.commit()

    except sqlite3.Error as e:
        logger.error(f"Database error in create_or_get_parameter_set: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        
    return parameter_set_id

def get_active_parameter_set(connection: sqlite3.Connection = None) -> Optional[Dict[str, Any]]:
    """
    Returns the currently active parameter set or None if none is active.
    
    Returns:
        Dictionary with parameter_set_id and parameters fields if an active set exists,
        otherwise None.
    """
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT parameter_set_id, parameters, default_metric_name, default_metric_value
            FROM parameter_sets 
            WHERE is_active = 1 
            LIMIT 1
            """
        )
        
        result = cursor.fetchone()
        if result:
            parameters = json.loads(result[1]) if result[1] else {}
            
            return {
                "parameter_set_id": result[0],
                "parameters": parameters,
                "default_metric_name": result[2],
                "default_metric_value": result[3]
            }
        return None
        
    except sqlite3.Error as e:
        logger.error(f"Database error in get_active_parameter_set: {e}")
        return None
    finally:
        if conn_created and conn:
            conn.close()

def set_parameter_set_active(parameter_set_id: str, deactivate_others: bool = True, connection: sqlite3.Connection = None) -> bool:
    """
    Sets a parameter set as active and optionally deactivates others.
    
    Args:
        parameter_set_id: The parameter set ID to activate
        deactivate_others: Whether to deactivate all other parameter sets
        connection: Optional existing database connection to use
        
    Returns:
        True if successful, False otherwise
    """
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        cursor = conn.cursor()
        
        # First check if parameter set exists
        cursor.execute(
            "SELECT 1 FROM parameter_sets WHERE parameter_set_id = ?",
            (parameter_set_id,)
        )
        
        if not cursor.fetchone():
            logger.error(f"Parameter set {parameter_set_id} not found")
            return False
            
        if deactivate_others:
            cursor.execute("UPDATE parameter_sets SET is_active = 0")
            
        cursor.execute(
            "UPDATE parameter_sets SET is_active = 1 WHERE parameter_set_id = ?",
            (parameter_set_id,)
        )
        
        conn.commit()
        logger.info(f"Set parameter set {parameter_set_id} as active")
        return True
        
    except sqlite3.Error as e:
        if conn_created and conn:
            conn.rollback()
        logger.error(f"Error setting parameter set {parameter_set_id} as active: {e}")
        return False
    finally:
        if conn_created and conn:
            conn.close()

def get_best_parameter_set_by_metric(metric_name: str, higher_is_better: bool = True, connection: sqlite3.Connection = None) -> Optional[Dict[str, Any]]:
    """
    Returns the parameter set with the best metric value based on training_results.
    
    Args:
        metric_name: The name of the metric to use for evaluation
        higher_is_better: True if higher values of the metric are better, False otherwise
        connection: Optional existing database connection to use
        
    Returns:
        Dictionary with parameter_set_id, parameters, and metrics fields if a best set exists,
        otherwise None.
    """
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        conn.row_factory = dict_factory # Use dict_factory
        cursor = conn.cursor()
        
        order_direction = "DESC" if higher_is_better else "ASC"
        
        # Join parameter_sets and training_results to find the best metrics
        # Ensure parameter_set_id is not NULL in training_results
        query = f"""
            SELECT 
                ps.parameter_set_id, 
                ps.parameters, 
                tr.metrics,
                JSON_EXTRACT(tr.metrics, '$.{metric_name}') as metric_value
            FROM training_results tr
            JOIN parameter_sets ps ON tr.parameter_set_id = ps.parameter_set_id
            WHERE tr.parameter_set_id IS NOT NULL AND JSON_VALID(tr.metrics) = 1 AND JSON_EXTRACT(tr.metrics, '$.{metric_name}') IS NOT NULL
            ORDER BY metric_value {order_direction}
            LIMIT 1
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result:
            # Parse JSON fields
            if result.get('parameters'):
                 try:
                     result['parameters'] = json.loads(result['parameters'])
                 except json.JSONDecodeError:
                     logger.warning(f"Could not decode parameters JSON for parameter_set {result['parameter_set_id']}")
                     result['parameters'] = {} # Set to empty dict on error
            else:
                result['parameters'] = {}

            if result.get('metrics'):
                 try:
                     result['metrics'] = json.loads(result['metrics'])
                 except json.JSONDecodeError:
                     logger.warning(f"Could not decode metrics JSON for parameter_set {result['parameter_set_id']} in training_results")
                     result['metrics'] = {} # Set to empty dict on error
            else:
                result['metrics'] = {}
                
            # Remove the extracted metric_value helper column
            result.pop('metric_value', None) 
            return result
            
        return None
        
    except sqlite3.Error as e:
        logger.error(f"Database error in get_best_parameter_set_by_metric: {e}", exc_info=True)
        return None
    finally:
        if conn_created and conn:
            conn.close() # Correct indentation

def create_model_record(
    model_id: str, 
    job_id: str, 
    model_path: str, 
    created_at: datetime, 
    metadata: Optional[Dict[str, Any]] = None, 
    is_active: bool = False
) -> None:
    """
    Creates a record for a trained model artifact.
    If `is_active` is True, this model will be marked active, deactivating others.
    
    Args:
        model_id: Unique identifier for the model
        job_id: ID of the job that produced the model
        model_path: Path to the model file
        created_at: Creation timestamp
        metadata: Optional metadata for the model
        is_active: Whether to explicitly set this model as active
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    metadata_json = json.dumps(metadata) if metadata else None
    
    try:
        # Create the model record, setting is_active based on explicit parameter ONLY
        cursor.execute(
            """
            INSERT INTO models (model_id, job_id, model_path, created_at, metadata, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model_id, job_id, model_path, created_at, metadata_json, is_active)
        )
        
        # If explicitly set as active, deactivate others
        if is_active:
            cursor.execute(
                """
                UPDATE models 
                SET is_active = 0 
                WHERE model_id != ?
                """,
                (model_id,)
            )
            logger.info(f"Model {model_id} created as active.")
            
        conn.commit()
        logger.info(f"Created model record for model_id: {model_id}")
    except sqlite3.Error as e:
        logger.error(f"Database error in create_model_record for model_id {model_id}: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def get_active_model(connection: sqlite3.Connection = None) -> Optional[Dict[str, Any]]:
    """
    Returns the currently active model or None if none is active.
    
    Args:
        connection: Optional existing database connection to use
        
    Returns:
        Dictionary with model information if an active model exists, 
        otherwise None.
    """
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT model_id, model_path, metadata 
            FROM models 
            WHERE is_active = 1 
            LIMIT 1
            """
        )
        
        result = cursor.fetchone()
        if result:
            metadata = json.loads(result[2]) if result[2] else {}
            
            return {
                "model_id": result[0],
                "model_path": result[1],
                "metadata": metadata
            }
        return None
        
    except sqlite3.Error as e:
        logger.error(f"Database error in get_active_model: {e}")
        return None
    finally:
        if conn_created and conn:
            conn.close()

def set_model_active(model_id: str, deactivate_others: bool = True, connection: sqlite3.Connection = None) -> bool:
    """
    Sets a model as active and optionally deactivates others.
    
    Args:
        model_id: The model ID to activate
        deactivate_others: Whether to deactivate all other models
        connection: Optional existing database connection to use
        
    Returns:
        True if successful, False otherwise
    """
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        cursor = conn.cursor()
        
        # First check if model exists
        cursor.execute(
            "SELECT 1 FROM models WHERE model_id = ?", 
            (model_id,)
        )
        
        if not cursor.fetchone():
            logger.error(f"Model {model_id} not found")
            return False
            
        if deactivate_others:
            cursor.execute("UPDATE models SET is_active = 0")
            
        cursor.execute(
            "UPDATE models SET is_active = 1 WHERE model_id = ?",
            (model_id,)
        )
        
        conn.commit()
        logger.info(f"Set model {model_id} as active")
        return True
        
    except sqlite3.Error as e:
        if conn_created and conn:
            conn.rollback()
        logger.error(f"Error setting model {model_id} as active: {e}")
        return False
    finally:
        if conn_created and conn:
            conn.close()

def get_best_model_by_metric(metric_name: str, higher_is_better: bool = True, connection: sqlite3.Connection = None) -> Optional[Dict[str, Any]]:
    """
    Returns the model with the best metric value based on training_results.
    
    Args:
        metric_name: The name of the metric to use for evaluation
        higher_is_better: True if higher values of the metric are better, False otherwise
        connection: Optional existing database connection to use
        
    Returns:
        Dictionary with model information if a best model exists, otherwise None.
    """
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        conn.row_factory = dict_factory # Use dict_factory for easier access
        cursor = conn.cursor()
        
        order_direction = "DESC" if higher_is_better else "ASC"
        
        # Join models and training_results to find the best metrics
        # Ensure model_id is not NULL in training_results
        query = f"""
            SELECT 
                m.model_id, 
                m.model_path, 
                m.metadata, 
                tr.metrics,
                JSON_EXTRACT(tr.metrics, '$.{metric_name}') as metric_value
            FROM training_results tr
            JOIN models m ON tr.model_id = m.model_id
            WHERE tr.model_id IS NOT NULL AND JSON_VALID(tr.metrics) = 1 AND JSON_EXTRACT(tr.metrics, '$.{metric_name}') IS NOT NULL
            ORDER BY metric_value {order_direction}
            LIMIT 1
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result:
            # No need to parse JSON here, return the dict directly
            # Ensure metadata is parsed if it exists
            if result.get('metadata'):
                 try:
                     result['metadata'] = json.loads(result['metadata'])
                 except json.JSONDecodeError:
                     logger.warning(f"Could not decode metadata JSON for model {result['model_id']}")
                     result['metadata'] = {} # Set to empty dict on error
            else:
                result['metadata'] = {}

            # Ensure metrics is parsed if it exists
            if result.get('metrics'):
                 try:
                     result['metrics'] = json.loads(result['metrics'])
                 except json.JSONDecodeError:
                     logger.warning(f"Could not decode metrics JSON for model {result['model_id']} in training_results")
                     result['metrics'] = {} # Set to empty dict on error
            else:
                result['metrics'] = {}
                
            # Remove the extracted metric_value helper column
            result.pop('metric_value', None) 
            return result
            
        return None
        
    except sqlite3.Error as e:
        logger.error(f"Database error in get_best_model_by_metric: {e}", exc_info=True)
        return None
    finally:
        if conn_created and conn:
            conn.close() # Correct indentation

def get_recent_models(limit: int = 5) -> List[Tuple[str, str, str, str, Optional[str]]]:
    """
    Retrieves the most recent model records, ordered by creation date.
    Returns a list of tuples: (model_id, job_id, model_path, created_at, metadata_json).
    """
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row # Use Row factory for easier access if needed later, but returning tuples for now
    cursor = conn.cursor()
    models = []
    try:
        cursor.execute(
            """
            SELECT model_id, job_id, model_path, created_at, metadata
            FROM models
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        )
        # Fetchall returns a list of Row objects if row_factory is set, or tuples otherwise.
        # We convert Rows to tuples explicitly for type hinting consistency.
        rows = cursor.fetchall()
        models = [(row[0], row[1], row[2], row[3], row[4]) for row in rows]

    except sqlite3.Error as e:
        print(f"Database error in get_recent_models: {e}")
    finally:
        conn.close()
    return models

def delete_model_record_and_file(model_id: str) -> bool:
    """
    Deletes a model record from the database and its associated file from the filesystem.
    Returns True if successful, False otherwise.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    model_path = None
    success = False
    
    try:
        # Get the model path before deleting the record
        cursor.execute("SELECT model_path FROM models WHERE model_id = ?", (model_id,))
        result = cursor.fetchone()
        if result:
            model_path = result[0]
        else:
            print(f"Model record not found for deletion: {model_id}")
            return False # Record doesn't exist

        # Delete the database record
        cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
        conn.commit()
        print(f"Deleted model record: {model_id}")

        # Delete the associated file if path exists
        if model_path and os.path.exists(model_path):
            try:
                os.remove(model_path)
                print(f"Deleted model file: {model_path}")
                success = True
            except OSError as e:
                print(f"Error deleting model file {model_path}: {e}")
                # Record deleted, but file deletion failed. Log and return False.
                success = False
        elif model_path:
            print(f"Model file not found for deletion (already deleted?): {model_path}")
            success = True # Record deleted, file was already gone. Consider this success.
        else:
             print(f"Model path was not stored for model_id {model_id}, cannot delete file.")
             success = True # Record deleted, no path to delete.

    except sqlite3.Error as e:
        print(f"Database error in delete_model_record_and_file for model_id {model_id}: {e}")
        conn.rollback()
        success = False
    finally:
        conn.close()
        
    return success

def create_training_result(
    job_id: str, 
    model_id: str, 
    parameter_set_id: str,
    metrics: Dict[str, Any], 
    parameters: Dict[str, Any],
    duration: Optional[int],
    connection: sqlite3.Connection = None # Allow passing connection for transaction
) -> str:
    """Creates a record for a completed training job and handles auto-activation."""
    result_id = str(uuid.uuid4())
    metrics_json = json.dumps(metrics)
    parameters_json = json.dumps(parameters) # Parameters used for this specific training run
    
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True

        cursor = conn.cursor()

        # Insert the training result
        query = """
        INSERT INTO training_results 
        (result_id, job_id, model_id, parameter_set_id, metrics, parameters, duration)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            result_id, job_id, model_id, parameter_set_id, metrics_json, parameters_json, duration
        )
        cursor.execute(query, params)
        logger.info(f"Created training result: {result_id} for job: {job_id}")

        # --- Auto-activation logic ---
        from deployment.app.config import settings # Import settings here

        # Auto-activate best parameter set
        if settings.auto_select_best_params:
            best_param_set = get_best_parameter_set_by_metric(
                metric_name=settings.default_metric,
                higher_is_better=settings.default_metric_higher_is_better,
                connection=conn # Use the same connection
            )
            if best_param_set and best_param_set.get("parameter_set_id"):
                logger.info(f"Auto-activating best parameter set: {best_param_set['parameter_set_id']}")
                set_parameter_set_active(best_param_set["parameter_set_id"], connection=conn)
            else:
                logger.warning(f"Auto-select params enabled, but couldn't find best parameter set by metric '{settings.default_metric}'")


        # Auto-activate best model
        if settings.auto_select_best_model and model_id: # Only if a model was actually created
             best_model = get_best_model_by_metric(
                 metric_name=settings.default_metric,
                 higher_is_better=settings.default_metric_higher_is_better,
                 connection=conn # Use the same connection
             )
             if best_model and best_model.get("model_id"):
                 logger.info(f"Auto-activating best model: {best_model['model_id']}")
                 set_model_active(best_model["model_id"], connection=conn)
             else:
                  logger.warning(f"Auto-select model enabled, but couldn't find best model by metric '{settings.default_metric}'")


        if conn_created: # Commit only if we created the connection here
             conn.commit()

    except sqlite3.Error as e:
        logger.error(f"Database error creating training result or auto-activating for job {job_id}: {e}", exc_info=True)
        if conn_created and conn:
            conn.rollback() # Rollback on error if we created the connection
        raise DatabaseError(f"Failed to create training result: {e}", original_error=e)
    finally:
        # Close only if we created the connection here
        if conn_created and conn:
            conn.close()
        
    return result_id

def create_prediction_result(
    job_id: str, 
    model_id: str, 
    output_path: str, 
    summary_metrics: Optional[Dict[str, Any]],
    connection: sqlite3.Connection = None
) -> str:
    """
    Create a prediction result record
    
    Args:
        job_id: Associated job ID
        model_id: ID of the model used for prediction
        output_path: Path to prediction output file
        summary_metrics: Dictionary of prediction metrics
        connection: Optional existing database connection to use
        
    Returns:
        Generated result ID
    """
    result_id = generate_id()
    
    query = """
    INSERT INTO prediction_results (result_id, job_id, model_id, output_path, summary_metrics)
    VALUES (?, ?, ?, ?, ?)
    """
    
    params = (
        result_id,
        job_id,
        model_id,
        output_path,
        json.dumps(summary_metrics) if summary_metrics else None
    )
    
    try:
        execute_query(query, params, connection=connection)
        return result_id
    except DatabaseError:
        logger.error(f"Failed to create prediction result for job {job_id}")
        raise

def create_report_result(job_id: str, report_type: str, parameters: Dict[str, Any], output_path: str, connection: sqlite3.Connection = None) -> str:
    """
    Create a report result record
    
    Args:
        job_id: Associated job ID
        report_type: Type of report
        parameters: Dictionary of report parameters
        output_path: Path to generated report
        connection: Optional existing database connection to use
        
    Returns:
        Generated result ID
    """
    result_id = generate_id()
    
    query = """
    INSERT INTO report_results (result_id, job_id, report_type, parameters, output_path)
    VALUES (?, ?, ?, ?, ?)
    """
    
    params = (
        result_id,
        job_id,
        report_type,
        json.dumps(parameters),
        output_path
    )
    
    try:
        execute_query(query, params, connection=connection)
        return result_id
    except DatabaseError:
        logger.error(f"Failed to create report result for job {job_id}")
        raise

def get_data_upload_result(result_id: str, connection: sqlite3.Connection = None) -> Dict:
    """Get data upload result by ID"""
    query = "SELECT * FROM data_upload_results WHERE result_id = ?"
    return execute_query(query, (result_id,), connection=connection)

def get_training_result(result_id: str, connection: sqlite3.Connection = None) -> Dict:
    """Get training result by ID"""
    query = "SELECT * FROM training_results WHERE result_id = ?"
    return execute_query(query, (result_id,), connection=connection)

def get_prediction_result(result_id: str, connection: sqlite3.Connection = None) -> Dict:
    """Get prediction result by ID"""
    query = "SELECT * FROM prediction_results WHERE result_id = ?"
    return execute_query(query, (result_id,), connection=connection)

def get_report_result(result_id: str, connection: sqlite3.Connection = None) -> Dict:
    """Get report result by ID"""
    query = "SELECT * FROM report_results WHERE result_id = ?"
    return execute_query(query, (result_id,), connection=connection)

# Processing runs management

def create_processing_run(start_time: datetime, status: str, 
                         cutoff_date: str, source_files: str,
                         end_time: datetime = None,
                         connection: sqlite3.Connection = None) -> int:
    """
    Create a processing run record
    
    Args:
        start_time: Start time of processing
        status: Status of the run
        cutoff_date: Date cutoff for data processing
        source_files: Comma-separated list of source files
        end_time: Optional end time of processing
        connection: Optional existing database connection to use
        
    Returns:
        Generated run ID
    """
    query = """
    INSERT INTO processing_runs (start_time, status, cutoff_date, source_files, end_time)
    VALUES (?, ?, ?, ?, ?)
    """
    
    params = (
        start_time.isoformat(),
        status,
        cutoff_date,
        source_files,
        end_time.isoformat() if end_time else None
    )
    
    try:
        execute_query(query, params, connection=connection)
        
        # Get the last inserted ID
        result = execute_query("SELECT last_insert_rowid() as run_id", connection=connection)
        return result["run_id"]
    except DatabaseError:
        logger.error("Failed to create processing run")
        raise

def update_processing_run(run_id: int, status: str, end_time: datetime = None,
                         connection: sqlite3.Connection = None) -> None:
    """
    Update a processing run
    
    Args:
        run_id: ID of the run to update
        status: New status
        end_time: Optional end time
        connection: Optional existing database connection to use
    """
    query = "UPDATE processing_runs SET status = ?"
    params = [status]
    
    if end_time:
        query += ", end_time = ?"
        params.append(end_time.isoformat())
    
    query += " WHERE run_id = ?"
    params.append(run_id)
    
    try:
        execute_query(query, tuple(params), connection=connection)
    except DatabaseError:
        logger.error(f"Failed to update processing run {run_id}")
        raise

# MultiIndex mapping functions

def get_or_create_multiindex_id(barcode: str, artist: str, album: str, 
                                cover_type: str, price_category: str,
                                release_type: str, recording_decade: str,
                                release_decade: str, style: str, 
                                record_year: int,
                                connection: sqlite3.Connection = None) -> int:
    """
    Get or create a multiindex mapping entry
    
    Args:
        barcode: Barcode value
        artist: Artist name
        album: Album name
        cover_type: Cover type
        price_category: Price category
        release_type: Release type
        recording_decade: Recording decade
        release_decade: Release decade
        style: Music style
        record_year: Record year
        connection: Optional existing database connection to use
        
    Returns:
        Multiindex ID
    """
    # Check if mapping already exists
    query = """
    SELECT multiindex_id FROM dim_multiindex_mapping
    WHERE barcode = ? AND artist = ? AND album = ? AND cover_type = ? AND
          price_category = ? AND release_type = ? AND recording_decade = ? AND
          release_decade = ? AND style = ? AND record_year = ?
    """
    
    params = (
        barcode, artist, album, cover_type, price_category,
        release_type, recording_decade, release_decade, style, record_year
    )
    
    try:
        existing = execute_query(query, params, connection=connection)
        
        if existing:
            return existing["multiindex_id"]
        
        # Create new mapping
        insert_query = """
        INSERT INTO dim_multiindex_mapping (
            barcode, artist, album, cover_type, price_category,
            release_type, recording_decade, release_decade, style, record_year
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        execute_query(insert_query, params, connection=connection)
        
        # Get the new ID
        result = execute_query("SELECT last_insert_rowid() as multiindex_id", connection=connection)
        return result["multiindex_id"]
        
    except DatabaseError:
        logger.error("Failed to get or create multiindex mapping")
        raise 

def get_parameter_sets(limit: int = 5, connection: sqlite3.Connection = None) -> List[Dict[str, Any]]:
    """
    Retrieves a list of parameter sets ordered by creation date.
    
    Args:
        limit: Maximum number of parameter sets to return
        connection: Optional existing database connection to use
        
    Returns:
        List of parameter sets with their details
    """
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT parameter_set_id, parameters, created_at, is_active, 
                   default_metric_name, default_metric_value
            FROM parameter_sets
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        )
        
        results = cursor.fetchall()
        for result in results:
            if 'parameters' in result and result['parameters']:
                result['parameters'] = json.loads(result['parameters'])
                
        return results
        
    except sqlite3.Error as e:
        logger.error(f"Database error in get_parameter_sets: {e}")
        return []
    finally:
        if conn_created and conn:
            conn.close()

def delete_parameter_sets_by_ids(parameter_set_ids: List[str], connection: sqlite3.Connection = None) -> Dict[str, Any]:
    """
    Deletes multiple parameter sets by their IDs.
    
    Args:
        parameter_set_ids: List of parameter set IDs to delete
        connection: Optional existing database connection to use
        
    Returns:
        Dict with results: {"successful": count, "failed": count, "errors": [list of errors]}
    """
    if not parameter_set_ids:
        return {"successful": 0, "failed": 0, "errors": []}
        
    conn_created = False
    result = {"successful": 0, "failed": 0, "errors": []}
    
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        cursor = conn.cursor()
        
        # Check for active parameter sets that would be deleted
        placeholders = ','.join(['?'] * len(parameter_set_ids))
        cursor.execute(
            f"""
            SELECT parameter_set_id FROM parameter_sets 
            WHERE is_active = 1 AND parameter_set_id IN ({placeholders})
            """, 
            parameter_set_ids
        )
        
        active_sets = cursor.fetchall()
        if active_sets:
            active_ids = [row[0] for row in active_sets]
            result["errors"].append(f"Cannot delete active parameter sets: {', '.join(active_ids)}")
            result["failed"] += len(active_ids)
            
            # Remove active sets from deletion list
            parameter_set_ids = [ps_id for ps_id in parameter_set_ids if ps_id not in active_ids]
            
        if parameter_set_ids:
            # Delete non-active parameter sets
            placeholders = ','.join(['?'] * len(parameter_set_ids))
            cursor.execute(
                f"DELETE FROM parameter_sets WHERE parameter_set_id IN ({placeholders})",
                parameter_set_ids
            )
            
            result["successful"] = cursor.rowcount
            conn.commit()
            
        return result
        
    except sqlite3.Error as e:
        if conn_created and conn:
            conn.rollback()
        error_msg = f"Error deleting parameter sets: {str(e)}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
        result["failed"] += len(parameter_set_ids)
        return result
    finally:
        if conn_created and conn:
            conn.close()

def get_all_models(limit: int = 100, include_active_status: bool = True, connection: sqlite3.Connection = None) -> List[Dict[str, Any]]:
    """
    Retrieves a list of all models with their details.
    
    Args:
        limit: Maximum number of models to return
        include_active_status: Whether to include the active status in the results
        connection: Optional existing database connection to use
        
    Returns:
        List of models with their details
    """
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT model_id, job_id, model_path, created_at, metadata, is_active
            FROM models
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        )
        
        results = cursor.fetchall()
        for result in results:
            if 'metadata' in result and result['metadata']:
                result['metadata'] = json.loads(result['metadata'])
                
        return results
        
    except sqlite3.Error as e:
        logger.error(f"Database error in get_all_models: {e}")
        return []
    finally:
        if conn_created and conn:
            conn.close()

def delete_models_by_ids(model_ids: List[str], connection: sqlite3.Connection = None) -> Dict[str, Any]:
    """
    Deletes multiple models by their IDs and their associated files.
    
    Args:
        model_ids: List of model IDs to delete
        connection: Optional existing database connection to use
        
    Returns:
        Dict with results: {"successful": count, "failed": count, "errors": [list of errors]}
    """
    if not model_ids:
        return {"successful": 0, "failed": 0, "errors": []}
        
    conn_created = False
    result = {"successful": 0, "failed": 0, "errors": []}
    
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
            
        cursor = conn.cursor()
        
        # Check for active models that would be deleted
        placeholders = ','.join(['?'] * len(model_ids))
        cursor.execute(
            f"""
            SELECT model_id FROM models 
            WHERE is_active = 1 AND model_id IN ({placeholders})
            """, 
            model_ids
        )
        
        active_models = cursor.fetchall()
        if active_models:
            active_ids = [row[0] for row in active_models]
            result["errors"].append(f"Cannot delete active models: {', '.join(active_ids)}")
            result["failed"] += len(active_ids)
            
            # Remove active models from deletion list
            model_ids = [m_id for m_id in model_ids if m_id not in active_ids]
            
        # Get model paths for non-active models to be deleted
        if model_ids:
            placeholders = ','.join(['?'] * len(model_ids))
            cursor.execute(
                f"""
                SELECT model_id, model_path FROM models 
                WHERE model_id IN ({placeholders})
                """, 
                model_ids
            )
            
            models_to_delete = cursor.fetchall()
            for model in models_to_delete:
                model_id = model[0]
                model_path = model[1]
                
                try:
                    # Delete file if it exists
                    if model_path and os.path.exists(model_path):
                        os.remove(model_path)
                        
                    # Delete database record
                    cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
                    
                    result["successful"] += 1
                except Exception as e:
                    error_msg = f"Error deleting model {model_id}: {str(e)}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
                    result["failed"] += 1
            
            conn.commit()
            
        return result
        
    except sqlite3.Error as e:
        if conn_created and conn:
            conn.rollback()
        error_msg = f"Error deleting models: {str(e)}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
        result["failed"] += len(model_ids)
        return result
    finally:
        if conn_created and conn:
            conn.close() 