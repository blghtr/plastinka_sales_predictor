import sqlite3
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

DB_PATH = "deployment/data/plastinka.db"

def get_db_connection():
    """Get a connection to the SQLite database"""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def dict_factory(cursor, row):
    """Convert row to dictionary"""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

def execute_query(query: str, params: tuple = (), fetchall: bool = False) -> Union[List[Dict], Dict, None]:
    """Execute a query and optionally return results"""
    conn = get_db_connection()
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    
    try:
        cursor.execute(query, params)
        
        if query.strip().upper().startswith(("SELECT", "PRAGMA")):
            if fetchall:
                result = cursor.fetchall()
            else:
                result = cursor.fetchone()
        else:
            conn.commit()
            result = None
    finally:
        conn.close()
        
    return result

def execute_many(query: str, params_list: List[tuple]) -> None:
    """Execute a query with multiple parameter sets"""
    if not params_list:
        return
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.executemany(query, params_list)
        conn.commit()
    finally:
        conn.close()

# Job-related database functions

def generate_id() -> str:
    """Generate a unique ID for jobs or results"""
    return str(uuid.uuid4())

def create_job(job_type: str, parameters: Dict[str, Any] = None) -> str:
    """Create a new job record and return the job ID"""
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
    
    execute_query(query, params)
    return job_id

def update_job_status(job_id: str, status: str, progress: float = None, 
                      result_id: str = None, error_message: str = None) -> None:
    """Update job status and related fields"""
    now = datetime.now().isoformat()
    
    # Build query dynamically based on provided parameters
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
    
    execute_query(query, tuple(params))

def get_job(job_id: str) -> Dict:
    """Get job details by ID"""
    query = "SELECT * FROM jobs WHERE job_id = ?"
    return execute_query(query, (job_id,))

def list_jobs(job_type: str = None, status: str = None, limit: int = 100) -> List[Dict]:
    """List jobs with optional filtering"""
    query = "SELECT * FROM jobs"
    params = []
    
    # Add filters
    if job_type or status:
        query += " WHERE"
        
        if job_type:
            query += " job_type = ?"
            params.append(job_type)
            
        if job_type and status:
            query += " AND"
            
        if status:
            query += " status = ?"
            params.append(status)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    return execute_query(query, tuple(params), fetchall=True)

# Result-related functions

def create_data_upload_result(job_id: str, records_processed: int, 
                              features_generated: List[str], 
                              processing_run_id: int) -> str:
    """Create a data upload result and return the result ID"""
    result_id = generate_id()
    
    query = """
    INSERT INTO data_upload_results 
    (result_id, job_id, records_processed, features_generated, processing_run_id)
    VALUES (?, ?, ?, ?, ?)
    """
    
    params = (
        result_id,
        job_id,
        records_processed,
        json.dumps(features_generated),
        processing_run_id
    )
    
    execute_query(query, params)
    return result_id

def create_training_result(job_id: str, model_id: str, metrics: Dict[str, Any], 
                         parameters: Dict[str, Any], duration: int) -> str:
    """Create a training result and return the result ID"""
    result_id = generate_id()
    
    query = """
    INSERT INTO training_results 
    (result_id, job_id, model_id, metrics, parameters, duration)
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
    
    execute_query(query, params)
    return result_id

def create_prediction_result(job_id: str, model_id: str, prediction_date: datetime,
                           output_path: str, summary_metrics: Dict[str, Any]) -> str:
    """Create a prediction result and return the result ID"""
    result_id = generate_id()
    
    query = """
    INSERT INTO prediction_results 
    (result_id, job_id, model_id, prediction_date, output_path, summary_metrics)
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
    
    execute_query(query, params)
    return result_id

def create_report_result(job_id: str, report_type: str, 
                        parameters: Dict[str, Any], output_path: str) -> str:
    """Create a report result and return the result ID"""
    result_id = generate_id()
    
    query = """
    INSERT INTO report_results 
    (result_id, job_id, report_type, parameters, output_path)
    VALUES (?, ?, ?, ?, ?)
    """
    
    params = (
        result_id,
        job_id,
        report_type,
        json.dumps(parameters),
        output_path
    )
    
    execute_query(query, params)
    return result_id

# Result retrieval functions

def get_data_upload_result(result_id: str) -> Dict:
    """Get data upload result details"""
    query = "SELECT * FROM data_upload_results WHERE result_id = ?"
    return execute_query(query, (result_id,))

def get_training_result(result_id: str) -> Dict:
    """Get training result details"""
    query = "SELECT * FROM training_results WHERE result_id = ?"
    return execute_query(query, (result_id,))

def get_prediction_result(result_id: str) -> Dict:
    """Get prediction result details"""
    query = "SELECT * FROM prediction_results WHERE result_id = ?"
    return execute_query(query, (result_id,))

def get_report_result(result_id: str) -> Dict:
    """Get report result details"""
    query = "SELECT * FROM report_results WHERE result_id = ?"
    return execute_query(query, (result_id,))

# Processing runs management

def create_processing_run(start_time: datetime, status: str, 
                         cutoff_date: str, source_files: str,
                         end_time: datetime = None) -> int:
    """Create a processing run record and return its ID"""
    query = """
    INSERT INTO processing_runs (start_time, end_time, status, cutoff_date, source_files)
    VALUES (?, ?, ?, ?, ?)
    """
    
    params = (
        start_time.isoformat(),
        end_time.isoformat() if end_time else None,
        status,
        cutoff_date,
        source_files
    )
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query, params)
        conn.commit()
        # Get the ID of the last inserted row
        run_id = cursor.lastrowid
    finally:
        conn.close()
        
    return run_id

def update_processing_run(run_id: int, status: str, end_time: datetime = None) -> None:
    """Update processing run status and end time"""
    query = "UPDATE processing_runs SET status = ?"
    params = [status]
    
    if end_time:
        query += ", end_time = ?"
        params.append(end_time.isoformat())
    
    query += " WHERE run_id = ?"
    params.append(run_id)
    
    execute_query(query, tuple(params))

# MultiIndex mapping functions
def get_or_create_multiindex_id(barcode: str, artist: str, album: str, 
                                cover_type: str, price_category: str,
                                release_type: str, recording_decade: str,
                                release_decade: str, style: str, 
                                record_year: int) -> int:
    """Get existing multiindex ID or create a new one"""
    # First, try to find existing record
    query = """
    SELECT multiindex_id FROM dim_multiindex_mapping
    WHERE barcode = ? AND artist = ? AND album = ? AND cover_type = ?
    AND price_category = ? AND release_type = ? AND recording_decade = ?
    AND release_decade = ? AND style = ? AND record_year = ?
    """
    
    params = (
        barcode, artist, album, cover_type, price_category, release_type,
        recording_decade, release_decade, style, record_year
    )
    
    result = execute_query(query, params)
    
    if result:
        return result["multiindex_id"]
    
    # Create new record if not found
    insert_query = """
    INSERT INTO dim_multiindex_mapping (
        barcode, artist, album, cover_type, price_category, release_type,
        recording_decade, release_decade, style, record_year
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(insert_query, params)
        conn.commit()
        # Get the ID of the last inserted row
        multiindex_id = cursor.lastrowid
    finally:
        conn.close()
        
    return multiindex_id 