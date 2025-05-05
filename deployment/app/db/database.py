import sqlite3
import json
import uuid
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from contextlib import contextmanager

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

def create_training_result(job_id: str, model_id: str, metrics: Dict[str, Any], 
                         parameters: Dict[str, Any], duration: int,
                         connection: sqlite3.Connection = None) -> str:
    """
    Create a training result record
    
    Args:
        job_id: Associated job ID
        model_id: ID of the trained model
        metrics: Dictionary of training metrics
        parameters: Dictionary of training parameters
        duration: Training duration in seconds
        connection: Optional existing database connection to use
        
    Returns:
        Generated result ID
    """
    result_id = generate_id()
    
    query = """
    INSERT INTO training_results (result_id, job_id, model_id, metrics, parameters, duration)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    
    params = (
        result_id,
        job_id,
        model_id,
        json.dumps(metrics),
        json.dumps(parameters),
        duration
    )
    
    try:
        execute_query(query, params, connection=connection)
        return result_id
    except DatabaseError:
        logger.error(f"Failed to create training result for job {job_id}")
        raise

def create_prediction_result(job_id: str, model_id: str, prediction_date: datetime,
                           output_path: str, summary_metrics: Dict[str, Any],
                           connection: sqlite3.Connection = None) -> str:
    """
    Create a prediction result record
    
    Args:
        job_id: Associated job ID
        model_id: ID of the model used for prediction
        prediction_date: Date of prediction
        output_path: Path to prediction output file
        summary_metrics: Dictionary of prediction metrics
        connection: Optional existing database connection to use
        
    Returns:
        Generated result ID
    """
    result_id = generate_id()
    
    query = """
    INSERT INTO prediction_results (result_id, job_id, model_id, prediction_date, 
                                   output_path, summary_metrics)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    
    params = (
        result_id,
        job_id,
        model_id,
        prediction_date.isoformat(),
        output_path,
        json.dumps(summary_metrics)
    )
    
    try:
        execute_query(query, params, connection=connection)
        return result_id
    except DatabaseError:
        logger.error(f"Failed to create prediction result for job {job_id}")
        raise

def create_report_result(job_id: str, report_type: str, 
                        parameters: Dict[str, Any], output_path: str,
                        connection: sqlite3.Connection = None) -> str:
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