"""Database queries for job management."""

import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from typing import Any

import asyncpg

from deployment.app.config import get_settings
from deployment.app.db.queries.core import execute_query
from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.utils import generate_id, json_default_serializer

logger = logging.getLogger("plastinka.database")


def _compute_param_hash(parameters: dict | None) -> str:
    """
    Compute a stable hash for job parameters to segment locks by job_type+parameters.
    """
    try:
        payload = json.dumps(parameters or {}, sort_keys=True, default=json_default_serializer)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        # Fallback: hash of string repr
        return hashlib.sha256(str(parameters).encode("utf-8")).hexdigest()


def _get_refractory_seconds(job_type: str) -> int:
    """Fetch refractory seconds from settings (per-type override or default)."""
    try:
        settings = get_settings()
        return settings.get_job_refractory_seconds(job_type)
    except Exception:
        return 300


async def create_job(
    job_type: str,
    parameters: dict[str, Any] = None,
    connection: asyncpg.Connection = None,
    status: str = "pending",
) -> str:
    """
    Create a new job record and return the job ID.

    Args:
        job_type: Type of job (from JobType enum)
        parameters: Dictionary of job parameters
        connection: Required database connection to use
        status: Initial job status (default: 'pending')

    Returns:
        Generated job ID
    """
    job_id = generate_id()
    now = datetime.now()

    sql_query = """
    INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, parameters, progress)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    """

    params_tuple = (
        job_id,
        job_type,
        status,
        now,
        now,
        json.dumps(parameters, default=json_default_serializer) if parameters else None,
        0.0,
    )

    await execute_query(sql_query, connection=connection, params=params_tuple)
    logger.info(f"Created new job: {job_id} of type {job_type}")
    return job_id


async def try_acquire_job_submission_lock(
    job_type: str,
    parameters: dict | None,
    connection: asyncpg.Connection,
) -> tuple[bool, int]:
    """
    Attempt to acquire submission lock for a (job_type, parameters) pair.

    Returns (acquired, retry_after_seconds).
    If acquired is False, retry_after_seconds indicates how long to wait before retrying.
    """
    now_iso = datetime.now()
    refractory_seconds = _get_refractory_seconds(job_type)
    param_hash = _compute_param_hash(parameters)

    # 1) Try to update existing lock if expired
    await execute_query(
        query=(
            "UPDATE job_submission_locks "
            "SET lock_until = $1 "
            "WHERE job_type = $2 AND param_hash = $3 AND lock_until <= $4"
        ),
        connection=connection,
        params=(
            datetime.now(),
            job_type,
            param_hash,
            now_iso,
        ),
    )

    # Select to see if lock was updated
    row = await execute_query(
        query="SELECT lock_until FROM job_submission_locks WHERE job_type = $1 AND param_hash = $2",
        connection=connection,
        params=(job_type, param_hash),
    )

    now_dt = datetime.now()

    if not row:
        # insert new lock row
        new_until = now_dt + timedelta(seconds=refractory_seconds)
        try:
            await execute_query(
                query=(
                    "INSERT INTO job_submission_locks (job_type, param_hash, lock_until) VALUES ($1, $2, $3)"
                ),
                connection=connection,
                params=(job_type, param_hash, new_until),
            )
            return True, 0
        except DatabaseError:
            # Could be raced by another inserter; fall through to check
            row = await execute_query(
                query="SELECT lock_until FROM job_submission_locks WHERE job_type = $1 AND param_hash = $2",
                connection=connection,
                params=(job_type, param_hash),
            )

    if row:
        lock_until = row.get("lock_until")
        try:
            lock_until_dt = datetime.fromisoformat(lock_until)
        except Exception:
            # If stored without full ISO format, try best effort
            lock_until_dt = datetime.fromisoformat(str(lock_until))

        if lock_until_dt <= now_dt:
            # Expired but our UPDATE didn't win; try one more UPDATE to extend
            new_until = now_dt + timedelta(seconds=refractory_seconds)
            await execute_query(
                query=(
                    "UPDATE job_submission_locks SET lock_until = $1 WHERE job_type = $2 AND param_hash = $3 AND lock_until <= $4"
                ),
                connection=connection,
                params=(new_until, job_type, param_hash, now_dt),
            )
            # Re-check state
            row2 = await execute_query(
                query="SELECT lock_until FROM job_submission_locks WHERE job_type = $1 AND param_hash = $2",
                connection=connection,
                params=(job_type, param_hash),
            )
            lock_until2_dt = datetime.fromisoformat(row2["lock_until"]) if row2 else now_dt
            if lock_until2_dt > now_dt:
                return True, 0
            return True, 0
        else:
            # Not expired: compute retry_after
            retry_after = max(0, int((lock_until_dt - now_dt).total_seconds()))
            return False, retry_after

    # Fallback acquire
    return True, 0


async def update_job_status(
    job_id: str,
    status: str,
    progress: float = None,
    result_id: str = None,
    error_message: str = None,
    status_message: str = None,
    connection: asyncpg.Connection = None,
) -> None:
    """
    Update job status and related fields

    Args:
        job_id: ID of the job to update
        status: New job status
        progress: Optional progress value (0-100)
        result_id: Optional result ID if job completed
        error_message: Optional error message if job failed
        status_message: Optional detailed status message (stored in job_status_history)
        connection: Required database connection to use
    """
    now = datetime.now()

    # First check if the job exists
    check_query = "SELECT 1 FROM jobs WHERE job_id = $1"
    result = await execute_query(check_query, connection=connection, params=(job_id,))
    if not result:
        logger.warning(
            f"Job with ID {job_id} not found while trying to update status to {status}"
        )
        return  # Exit early without raising an error

    # Update job status
    query = """
        UPDATE jobs
        SET
            status = $1,
            updated_at = $2,
            progress = COALESCE($3, progress),
            result_id = COALESCE($4, result_id),
            error_message = COALESCE($5, error_message)
        WHERE job_id = $6
    """
    params = (status, now, progress, result_id, error_message, job_id)
    await execute_query(query, connection=connection, params=params)

    # Always log status change to job_status_history table
    # If no status_message is provided, use the status itself
    history_message = (
        status_message if status_message else f"Status changed to: {status}"
    )
    history_query = """
        INSERT INTO job_status_history
        (job_id, status, status_message, progress, updated_at)
        VALUES ($1, $2, $3, $4, $5)
    """
    history_params = (job_id, status, history_message, progress, now)
    await execute_query(history_query, connection=connection, params=history_params)

    logger.info(
        f"Updated job {job_id}: status={status}, progress={progress}, message={status_message}"
    )


async def get_job(job_id: str, connection: asyncpg.Connection) -> dict | None:
    """
    Get job details by ID

    Args:
        job_id: Job ID to retrieve
        connection: Required database connection to use

    Returns:
        Job details dictionary or None if not found
    """
    query = "SELECT * FROM jobs WHERE job_id = $1"
    try:
        result = await execute_query(query, connection=connection, params=(job_id,))
        # Parse JSONB parameters field (defensive parsing)
        # Note: asyncpg documentation states JSONB should auto-deserialize to dict when Record is converted via dict(row).
        # However, production code shows defensive parsing is needed in multiple functions, suggesting:
        # - asyncpg version differences may affect behavior
        # - Connection configuration may affect type decoding
        # - SQL queries casting JSONB to TEXT return strings
        # This defensive pattern matches get_training_results, get_best_model_by_metric, get_configs
        if result and result.get("parameters"):
            if isinstance(result["parameters"], dict):
                pass  # Already parsed by asyncpg
            elif isinstance(result["parameters"], str):
                try:
                    result["parameters"] = json.loads(result["parameters"])
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in job parameters for job {job_id}")
                    result["parameters"] = {}
            else:
                result["parameters"] = {}
        return result
    except DatabaseError as e:
        logger.error(f"Failed to get job {job_id}: {str(e)}")
        raise


async def list_jobs(
    job_type: str = None,
    status: str = None,
    limit: int = 100,
    connection: asyncpg.Connection = None,
) -> list[dict]:
    """
    List jobs with optional filters

    Args:
        job_type: Optional job type filter
        status: Optional status filter
        limit: Maximum number of jobs to return
        connection: Required database connection to use

    Returns:
        List of job dictionaries
    """
    query_parts = ["SELECT * FROM jobs WHERE 1=1"]
    params = []
    param_num = 1

    if job_type:
        query_parts.append(f" AND job_type = ${param_num}")
        params.append(job_type)
        param_num += 1

    if status:
        query_parts.append(f" AND status = ${param_num}")
        params.append(status)
        param_num += 1

    query_parts.append(f" ORDER BY created_at DESC LIMIT ${param_num}")
    params.append(limit)

    query = "".join(query_parts)

    try:
        results = await execute_query(
            query, params=tuple(params), fetchall=True, connection=connection
        )
        # Parse JSONB parameters field for each job (defensive parsing)
        # See note in get_job() about why defensive parsing is needed despite asyncpg docs
        if results:
            for job in results:
                if job.get("parameters"):
                    if isinstance(job["parameters"], dict):
                        pass  # Already parsed by asyncpg
                    elif isinstance(job["parameters"], str):
                        try:
                            job["parameters"] = json.loads(job["parameters"])
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in job parameters for job {job.get('job_id', 'unknown')}")
                            job["parameters"] = {}
                    else:
                        job["parameters"] = {}
        return results or []
    except DatabaseError as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise


async def get_job_params(
    job_id: str,
    connection: asyncpg.Connection,
    param_name: str = None
) -> dict[str, Any] | Any:
    """
    Retrieves and parses job parameters from the database.
    
    Args:
        job_id: The job ID to retrieve parameters for
        connection: Required database connection to use
        param_name: Optional specific parameter name to extract (e.g., 'prediction_month')
        
    Returns:
        Dictionary of job parameters if param_name is None, or the specific parameter value
    """
    try:
        job_details = await get_job(job_id, connection=connection)
        if not job_details or not job_details.get("parameters"):
            logger.warning(f"No parameters found for job {job_id}")
            return {} if param_name is None else None
            
        params_str = job_details["parameters"]
        
        # Parse JSON string to dictionary
        if isinstance(params_str, str):
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in job parameters for job {job_id}: {params_str}, error: {e}")
                return {} if param_name is None else None
        else:
            params = params_str
            
        if not isinstance(params, dict):
            logger.warning(f"Job parameters for job {job_id} is not a dictionary: {type(params)}")
            return {} if param_name is None else None
            
        # Return specific parameter if requested
        if param_name is not None:
            return params.get(param_name)
            
        return params
        
    except Exception as e:
        logger.warning(f"Could not retrieve parameters for job {job_id}: {e}")
        return {} if param_name is None else None


async def get_job_prediction_month(job_id: str, connection: asyncpg.Connection) -> date:
    """
    Retrieves and parses the prediction_month parameter from job parameters.
    
    Args:
        job_id: The job ID to retrieve prediction_month for
        connection: Required database connection to use
        
    Returns:
        date object representing the prediction month, or today's date as fallback
    """
    prediction_month = await get_job_params(job_id, connection=connection, param_name="prediction_month")
    
    if prediction_month is None:
        logger.warning(f"No prediction_month found for job {job_id}, using today's date as fallback")
        return date.today()
    
    # Convert string to date object if needed
    if isinstance(prediction_month, str):
        try:
            return date.fromisoformat(prediction_month)
        except ValueError as e:
            logger.warning(f"Invalid prediction_month format for job {job_id}: {prediction_month}, error: {e}")
            return date.today()
    elif isinstance(prediction_month, date):
        return prediction_month
    else:
        logger.warning(f"Unexpected prediction_month type for job {job_id}: {type(prediction_month)}")
        return date.today()

