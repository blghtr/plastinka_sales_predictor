import hashlib
import json
import logging
import os
import sqlite3
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from deployment.app.config import get_settings
from deployment.app.models.api_models import TrainingConfig
from deployment.app.utils.retry import retry_with_backoff

logger = logging.getLogger("plastinka.database")

# SQLite default limit is 999, use 900 for safety
SQLITE_MAX_VARIABLES = 900


def json_default_serializer(obj):
    """
    JSON serializer for objects not serializable by default json code
    """
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# Define allowed metric names for dynamic queries
ALLOWED_METRICS = [
    # Standardized metric names from training phases (with prefixes)
    "val_MIC",
    "val_MIWS",
    "val_MIWS_MIC_Ratio",
    "val_loss",
    "train_loss",
    "train_MIC",
    "train_MIWS",
    "train_MIWS_MIC_Ratio",
    "training_duration_seconds",
    # FIXED: Add actual metric names with double prefixes from database
    "val_val_loss",
    "val_val_MIWS",
    "val_val_MIC",
    "val_val_MIWS_MIC_Ratio",
    "train_train_loss",
    "train_train_MIWS",
    "train_train_MIC",
    "train_train_MIWS_MIC_Ratio",
]

EXPECTED_REPORT_FEATURES = [
    "availability",
    "confidence",
    "masked_mean_sales_items",
    "masked_mean_sales_rub",
    "lost_sales",
]

EXPECTED_REPORT_FEATURES_SET = set(EXPECTED_REPORT_FEATURES)

# Use database path from settings
# DB_PATH = settings.database_path


class DatabaseError(Exception):
    """Exception raised for database errors."""

    def __init__(
        self,
        message: str,
        query: str = None,
        params: tuple = None,
        original_error: Exception = None,
    ):
        self.message = message
        self.query = query
        self.params = params
        self.original_error = original_error
        super().__init__(self.message)


def _is_path_safe(base_dir: str | Path, path_to_check: str | Path) -> bool:
    """
    Check if path_to_check is inside base_dir to prevent path traversal attacks.

    Args:
        base_dir: The base directory that should contain the path
        path_to_check: The path to validate

    Returns:
        bool: True if the path is safe (inside base_dir), False otherwise
    """
    try:
        # Resolve both paths to their absolute form
        resolved_base = Path(base_dir).resolve()
        resolved_path = Path(path_to_check).resolve()
        # Check if the resolved path is a subpath of the base directory
        return resolved_path.is_relative_to(resolved_base)
    except Exception:
        return False


def get_db_connection(
    db_path_override: str | Path | None = None,
    existing_connection: sqlite3.Connection | None = None,
):
    """
    Get a connection to the SQLite database.
    This function is primarily intended for internal use by DataAccessLayer.__init__.
    If existing_connection is provided, it uses that connection.
    Otherwise, if db_path_override is provided, it uses that path; otherwise, uses settings.database_path.
    """
    try:
        if existing_connection:
            return existing_connection

        if db_path_override:
            current_db_path = Path(db_path_override)
        else:
            # Always get the most up-to-date path from settings
            current_db_path = Path(get_settings().database_path)

        current_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(current_db_path), check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = dict_factory
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}", exc_info=True)
        raise DatabaseError(
            f"Database connection failed: {str(e)}", original_error=e
        ) from e





def dict_factory(cursor, row):
    """Convert row to dictionary"""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


@retry_with_backoff(max_tries=3, base_delay=1.0, max_delay=10.0, component="database_query")
def execute_query(
    query: str,
    connection: sqlite3.Connection,
    params: tuple = (),
    fetchall: bool = False,
) -> list[dict] | dict | None:
    """
    Execute a query and optionally return results.
    This function *always* requires an explicit `connection` parameter and
    *never* creates or closes its own connections. It operates purely within
    the transaction context provided by the caller.

    Args:
        query: SQL query with placeholders (?, :name)
        params: Parameters for the query
        fetchall: Whether to fetch all results or just one
        connection: The database connection to use. This function will NOT commit or rollback.

    Returns:
        Query results as dict or list of dicts, or None for operations

    Raises:
        DatabaseError: If database operation fails
    """
    conn = connection
    cursor = None

    try:
        _ = conn.isolation_level

        conn.row_factory = dict_factory
        cursor = conn.cursor()

        cursor.execute(query, params)

        # Проверяем, является ли это SELECT запросом (включая CTE)
        query_upper = query.strip().upper()
        is_select = (
            query_upper.startswith("SELECT") or
            query_upper.startswith("PRAGMA") or
            "SELECT" in query_upper[:50] or
            query_upper.startswith("WITH")
        )
        
        if is_select:
            if fetchall:
                result = cursor.fetchall()
            else:
                result = cursor.fetchone()
        else:
            result = None

        return result if result is not None else ([] if fetchall else None)

    except sqlite3.Error as e:
        safe_params = "..." if params else "()"
        logger.error(
            f"Database error in query: {query[:100]} with params: {safe_params}: {str(e)}",
            exc_info=True,
        )

        raise DatabaseError(
            message=f"Database operation failed: {str(e)}",
            query=query,
            params=params,
            original_error=e,
        ) from e
    finally:
        pass


@retry_with_backoff(max_tries=3, base_delay=1.0, max_delay=10.0, component="database_batch")
def execute_many(
    query: str,
    params_list: list[tuple],
    connection: sqlite3.Connection,
) -> None:
    """
    Execute a query with multiple parameter sets.
    This function *always* requires an explicit `connection` parameter and
    *never* creates or closes its own connections. It operates purely within
    the transaction context provided by the caller.

    Args:
        query: SQL query with placeholders
        params_list: List of parameter tuples
        connection: The database connection to use. This function will NOT commit or rollback.

    Raises:
        DatabaseError: If database operation fails
    """
    if not params_list:
        return

    conn = connection
    cursor = None

    try:
        cursor = conn.cursor()

        cursor.executemany(query, params_list)

    except sqlite3.Error as e:
        logger.error(
            f"Database error in executemany: {query[:100]}, params count: {len(params_list)}: {str(e)}",
            exc_info=True,
        )

        raise DatabaseError(
            message=f"Batch database operation failed: {str(e)}",
            query=query,
            original_error=e,
        ) from e
    finally:
        if cursor:
            cursor.close()


# Job-related database functions


def generate_id() -> str:
    """Generate a unique ID for jobs or results"""
    return str(uuid.uuid4())


def create_job(
    job_type: str,
    parameters: dict[str, Any] = None,
    connection: sqlite3.Connection = None,
    status: str = "pending",
) -> str:
    """
    Create a new job record and return the job ID.
    If an external 'connection' is provided, this function operates within that transaction.
    Otherwise, it creates a new transaction using the 'db_transaction' context manager.

    Args:
        job_type: Type of job (from JobType enum)
        parameters: Dictionary of job parameters
        connection: Optional existing database connection to use
        status: Initial job status (default: 'pending')

    Returns:
        Generated job ID
    """
    job_id = generate_id()
    now = datetime.now().isoformat()

    sql_query = """
    INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, parameters, progress)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    params_tuple = (
        job_id,
        job_type,
        status,
        now,
        now,
        json.dumps(parameters, default=json_default_serializer) if parameters else None,
        0,
    )

    def _db_operation(conn_to_use: sqlite3.Connection):
        # execute_query will use the provided conn_to_use and will NOT commit/rollback itself.
        execute_query(sql_query, params=params_tuple, connection=conn_to_use)
        logger.info(f"Created new job: {job_id} of type {job_type}")

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    _db_operation(connection)
    return job_id


def update_job_status(
    job_id: str,
    status: str,
    progress: float = None,
    result_id: str = None,
    error_message: str = None,
    status_message: str = None,
    connection: sqlite3.Connection = None,
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

        connection: Optional existing database connection to use
    """
    now = datetime.now().isoformat()

    def _update_operation(conn_to_use: sqlite3.Connection):
        # First check if the job exists
        check_query = "SELECT 1 FROM jobs WHERE job_id = ?"
        result = execute_query(check_query, params=(job_id,), connection=conn_to_use)
        if not result:
            logger.warning(
                f"Job with ID {job_id} not found while trying to update status to {status}"
            )
            return  # Exit early without raising an error

        # Update job status
        query = """
            UPDATE jobs
            SET
                status = ?,
                updated_at = ?,
                progress = COALESCE(?, progress),
                result_id = COALESCE(?, result_id),
                error_message = COALESCE(?, error_message)
            WHERE job_id = ?
        """
        params = (status, now, progress, result_id, error_message, job_id)
        execute_query(query, params=params, connection=conn_to_use)

        # Always log status change to job_status_history table
        # If no status_message is provided, use the status itself
        history_message = (
            status_message if status_message else f"Status changed to: {status}"
        )
        history_query = """
            INSERT INTO job_status_history
            (job_id, status, status_message, progress, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """
        history_params = (job_id, status, history_message, progress, now)
        execute_query(history_query, params=history_params, connection=conn_to_use)

        logger.info(
            f"Updated job {job_id}: status={status}, progress={progress}, message={status_message}"
        )

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    _update_operation(connection)


def get_job(job_id: str, connection: sqlite3.Connection = None) -> dict:
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
        result = execute_query(query, params=(job_id,), connection=connection)
        return result
    except DatabaseError as e:
        logger.error(f"Failed to get job {job_id}: {str(e)}")
        raise


def list_jobs(
    job_type: str = None,
    status: str = None,
    limit: int = 100,
    connection: sqlite3.Connection = None,
) -> list[dict]:
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
        results = execute_query(
            query, params=tuple(params), fetchall=True, connection=connection
        )
        return results or []
    except DatabaseError as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise


# Result-related functions


def create_data_upload_result(
    job_id: str,
    records_processed: int,
    features_generated: list[str],
    processing_run_id: int,
    connection: sqlite3.Connection = None,
) -> str:
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
        json.dumps(features_generated, default=json_default_serializer),
        processing_run_id,
    )

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    execute_query(query, params=params, connection=connection)
    return result_id


def create_or_get_config(
    config_dict: dict[str, Any],
    is_active: bool = False,
    source: str | None = None,
    connection: sqlite3.Connection = None,
) -> str:
    """
    Creates a config record if it doesn't exist, based on a hash of the config.
    If `is_active` is True, this config will be marked active, deactivating others.
    Returns the config_id.

    Args:
        config_dict: Dictionary of config
        is_active: Whether to explicitly set this config as active
        source: Optional source for the config
        connection: Optional existing database connection to use

    Returns:
        Config ID (hash)
    """
    try:
        _ = TrainingConfig(**config_dict)
    except (ValidationError, ValueError) as e:
        logger.error(f"Configuration is invalid: {e}", exc_info=True)
        raise ValueError(f"Invalid configuration provided: {e}") from e

    def _activate_config(connection: sqlite3.Connection, config_id: str) -> None:
        execute_query(query="UPDATE configs SET is_active = 0 WHERE config_id != ?", connection=connection, params=(config_id,))
        execute_query(query="UPDATE configs SET is_active = 1 WHERE config_id = ?", connection=connection, params=(config_id,))

    def _config_operation(conn_to_use: sqlite3.Connection) -> str:
        config_json = json.dumps(
            config_dict, sort_keys=True, default=json_default_serializer
        )
        config_id = hashlib.md5(config_json.encode()).hexdigest()

        result = execute_query(
            query="SELECT config_id FROM configs WHERE config_id = ?", connection=conn_to_use, params=(config_id,)
        )

        if not result:
            now = datetime.now().isoformat()

            execute_query(
                query="""
                INSERT INTO configs (config_id, config, is_active, created_at, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                connection=conn_to_use,
                params=(config_id, config_json, 1 if is_active else 0, now, source),
            )

            logger.info(
                f"Created config: {config_id}, active: {is_active}, source: {source}"
            )
        else:
            if source is not None:
                execute_query(query="UPDATE configs SET source = ? WHERE config_id = ?", connection=conn_to_use, params=(source, config_id))

        if is_active:
            _activate_config(conn_to_use, config_id)
            logger.info(f"Set config {config_id} as active")

        return config_id

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    return _config_operation(connection)


def get_active_config(connection: sqlite3.Connection = None) -> dict[str, Any] | None:
    """
    Returns the currently active config or None if none is active.
    Returns:
        Dictionary with config_id and config fields if an active config exists,
        otherwise None.
    """
    def _get_active_config_operation(conn_to_use: sqlite3.Connection) -> dict[str, Any] | None:
        result = execute_query(
            query="""
            SELECT config_id, config
            FROM configs
            WHERE is_active = 1
            LIMIT 1
            """,
            connection=conn_to_use
        )
        if not result:
            return None

        config_id = result["config_id"]
        config_json = result["config"]
        try:
            config = json.loads(config_json) if config_json else {}
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Error parsing config JSON: {config_json}")
            config = {}
        return {"config_id": config_id, "config": config}

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    return _get_active_config_operation(connection)


def set_config_active(
    config_id: str,
    deactivate_others: bool = True,
    connection: sqlite3.Connection = None,
) -> bool:
    """
    Sets a config as active and optionally deactivates others.

    Args:
        config_id: The config ID to activate
        deactivate_others: Whether to deactivate all other configs
        connection: Optional existing database connection to use

    Returns:
        True if successful, False otherwise
    """
    def _set_config_active_operation(conn_to_use: sqlite3.Connection) -> bool:
        # First check if config exists
        result = execute_query(query="SELECT 1 FROM configs WHERE config_id = ?", connection=conn_to_use, params=(config_id,))

        if not result:
            logger.error(f"Config {config_id} not found")
            return False

        if deactivate_others:
            execute_query(query="UPDATE configs SET is_active = 0", connection=conn_to_use)

        execute_query(
            query="UPDATE configs SET is_active = 1 WHERE config_id = ?", connection=conn_to_use, params=(config_id,)
        )

        logger.info(f"Set config {config_id} as active")
        return True

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    return _set_config_active_operation(connection)


def get_best_config_by_metric(
    metric_name: str,
    higher_is_better: bool = True,
    metric_source: str = "train",
    connection: sqlite3.Connection = None,
) -> dict[str, Any] | None:
    """
    Returns the config with the best metric value by searching across both
    training_results and tuning_results.

    Args:
        metric_name: The name of the metric to use for evaluation.
        higher_is_better: True if higher values of the metric are better.
        metric_source: The source of the metric, either 'train' or 'tune'.
        connection: Optional existing database connection.

    Returns:
        A dictionary with config_id, config, and metrics, or None if no config is found.
    """
    # Use the generalized get_top_configs function to find the single best config
    # Don't pass source parameter to get_top_configs - it should search both tables
    top_configs = get_top_configs(
        connection=connection,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        limit=1,
    )

    if not top_configs:
        return None

    # get_top_configs returns a list, so we take the first element
    best_config_result = top_configs[0]

    # The result from get_top_configs already includes `config_id`, `config`, and `metrics`
    # in the desired format, so we can return it directly.
    return best_config_result


def create_model_record(
    model_id: str,
    job_id: str,
    model_path: str,
    created_at: datetime,
    metadata: dict[str, Any] | None = None,
    is_active: bool = False,
    connection: sqlite3.Connection = None,
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
        connection: Optional existing database connection to use
    """
    def _create_model_record_operation(conn_to_use: sqlite3.Connection):
        execute_query(
            query="""
            INSERT INTO models (
                model_id, job_id, model_path, created_at, metadata, is_active
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            connection=conn_to_use,
            params=(
                model_id,
                job_id,
                model_path,
                created_at.isoformat()
                if isinstance(created_at, datetime)
                else created_at,
                json.dumps(metadata, default=json_default_serializer)
                if metadata
                else None,
                1 if is_active else 0,
            ),
        )

        if is_active:
            execute_query(
                query="""
                UPDATE models
                SET is_active = 0
                WHERE model_id != ?
                """,
                connection=conn_to_use,
                params=(model_id,),
            )

        logger.info(f"Created model record: {model_id}, active: {is_active}")

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    _create_model_record_operation(connection)


def get_active_model(connection: sqlite3.Connection = None) -> dict[str, Any] | None:
    """
    Returns the currently active model or None if none is active.

    Args:
        connection: Optional existing database connection to use

    Returns:
        Dictionary with model information if an active model exists,
        otherwise None.
    """
    def _get_active_model_operation(conn_to_use: sqlite3.Connection) -> dict[str, Any] | None:
        result = execute_query(
            query="""
            SELECT model_id, model_path, metadata
            FROM models
            WHERE is_active = 1
            LIMIT 1
            """,
            connection=conn_to_use
        )
        if not result:
            return None

        model_id = result["model_id"]
        model_path = result["model_path"]
        metadata_str = result["metadata"] if "metadata" in result else None

        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except json.JSONDecodeError:
            logger.warning(f"Could not decode metadata JSON for model {model_id}")
            metadata = {}
        return {"model_id": model_id, "model_path": model_path, "metadata": metadata}

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    return _get_active_model_operation(connection)


def get_active_model_primary_metric(connection: sqlite3.Connection = None) -> float | None:
    """
    Retrieves the primary metric of the currently active model.

    Args:
        connection: Optional existing database connection.

    Returns:
        The value of the primary metric as a float, or None if not found.
    """
    settings = get_settings()
    primary_metric_name = settings.default_metric

    if primary_metric_name not in ALLOWED_METRICS:
        logger.error(f"Primary metric '{primary_metric_name}' is not in ALLOWED_METRICS.")
        return None

    query = f"""
        SELECT
            JSON_EXTRACT(tr.metrics, '$.{primary_metric_name}') as metric_value
        FROM models m
        JOIN training_results tr ON m.model_id = tr.model_id
        WHERE m.is_active = 1
        ORDER BY m.created_at DESC
        LIMIT 1;
    """

    try:
        result = execute_query(query, connection=connection)
        if result and result["metric_value"] is not None:
            return float(result["metric_value"])
        return None
    except (DatabaseError, ValueError, TypeError) as e:
        logger.error(f"Could not retrieve or cast active model's primary metric: {e}", exc_info=True)
        return None


def set_model_active(
    model_id: str, deactivate_others: bool = True, connection: sqlite3.Connection = None
) -> bool:
    """
    Sets a model as active and optionally deactivates others.

    Args:
        model_id: The model ID to activate
        deactivate_others: Whether to deactivate all other models
        connection: Optional existing database connection to use

    Returns:
        True if successful, False otherwise
    """
    def _set_model_active_operation(conn_to_use: sqlite3.Connection) -> bool:
        # First check if model exists
        result = execute_query(query="SELECT 1 FROM models WHERE model_id = ?", connection=conn_to_use, params=(model_id,))

        if not result:
            logger.error(f"Model {model_id} not found")
            return False

        if deactivate_others:
            execute_query(query="UPDATE models SET is_active = 0", connection=conn_to_use)

        execute_query(
            query="UPDATE models SET is_active = 1 WHERE model_id = ?", connection=conn_to_use, params=(model_id,)
        )

        logger.info(f"Set model {model_id} as active")
        return True

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    return _set_model_active_operation(connection)


def get_best_model_by_metric(
    metric_name: str,
    higher_is_better: bool = True,
    connection: sqlite3.Connection = None,
) -> dict[str, Any] | None:
    """
    Returns the model with the best metric value based on training_results.

    Args:
        metric_name: The name of the metric to use for evaluation
        higher_is_better: True if higher values of the metric are better, False otherwise
        connection: Optional existing database connection to use

    Returns:
        Dictionary with model information if a best model exists, otherwise None.
    """
    if metric_name not in ALLOWED_METRICS:
        logger.error(
            f"Invalid metric_name '{metric_name}' provided to get_best_model_by_metric."
        )
        raise ValueError(
            f"Invalid metric_name: {metric_name}. Allowed metrics are: {ALLOWED_METRICS}"
        )

    def _get_best_model_by_metric_operation(conn_to_use: sqlite3.Connection) -> dict[str, Any] | None:
        order_direction = "DESC" if higher_is_better else "ASC"

        json_path = f"'$.{metric_name}'"

        query = f"""
            SELECT
                m.model_id,
                m.model_path,
                m.metadata,
                tr.metrics,
                JSON_EXTRACT(tr.metrics, {json_path}) as metric_value
            FROM training_results tr
            JOIN models m ON tr.model_id = m.model_id
            WHERE tr.model_id IS NOT NULL AND JSON_VALID(tr.metrics) = 1 AND JSON_EXTRACT(tr.metrics, {json_path}) IS NOT NULL
            ORDER BY metric_value {order_direction}
            LIMIT 1
        """

        result = execute_query(query=query, connection=conn_to_use)

        if result:
            if result.get("metadata"):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode metadata JSON for model {result['model_id']}"
                    )
                    result["metadata"] = {}
            else:
                result["metadata"] = {}

            if result.get("metrics"):
                try:
                    result["metrics"] = json.loads(result["metrics"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode metrics JSON for model {result['model_id']} in training_results"
                    )
                    result["metrics"] = {}
            else:
                result["metrics"] = {}

            result.pop("metric_value", None)
            return result

        return None

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    return _get_best_model_by_metric_operation(connection)


def get_recent_models(
    limit: int = 5, connection: sqlite3.Connection = None
) -> list[dict[str, Any]]:
    """
    Get the most recent models from the database.
    """
    query = """
        SELECT model_id, job_id, model_path, created_at, metadata, is_active
        FROM models
        ORDER BY created_at DESC
        LIMIT ?
    """
    return execute_query(query=query, connection=connection, params=(limit,), fetchall=True)


def delete_model_record_and_file(
    model_id: str, connection: sqlite3.Connection = None
) -> bool:
    """
    Deletes a model record from the database and its associated file from the filesystem.
    Returns True if successful, False otherwise.
    """
    def _delete_operation(conn_to_use: sqlite3.Connection) -> bool:
        # Get the model path before deleting the record
        result = execute_query(
            query="SELECT model_path FROM models WHERE model_id = ?", 
            connection=conn_to_use, 
            params=(model_id,)
        )
        if not result:
            logger.warning(f"Model with ID {model_id} not found for deletion.")
            return False

        model_path = result["model_path"]
        models_base_dir = get_settings().models_dir

        # 1. Delete the model file safely (with path traversal protection)
        if model_path and os.path.exists(model_path):
            if not _is_path_safe(models_base_dir, model_path):
                logger.error(
                    f"Path traversal attempt detected for model {model_id}. Path '{model_path}' is outside of designated models directory '{models_base_dir}'. File will not be deleted."
                )
            else:
                try:
                    os.remove(model_path)
                    logger.info(f"Deleted model file: {model_path}")
                except OSError as e:
                    logger.error(
                        f"Error removing model file {model_path}: {e}", exc_info=True
                    )

        # 2. Delete dependent records from training_results
        execute_query(
            query="DELETE FROM training_results WHERE model_id = ?", connection=conn_to_use, params=(model_id,)
        )

        # 3. Delete the model record
        execute_query(query="DELETE FROM models WHERE model_id = ?", connection=conn_to_use, params=(model_id,))
        return True

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    return _delete_operation(connection)


def create_training_result(
    job_id: str,
    model_id: str,
    config_id: str,
    metrics: dict[str, Any],
    duration: int | None,
    connection: sqlite3.Connection = None,  # Allow passing connection for transaction
) -> str:
    """Creates a record for a completed training job and handles auto-activation."""
    result_id = str(uuid.uuid4())
    metrics_json = json.dumps(metrics, default=json_default_serializer)

    def _create_training_result_operation(conn_to_use: sqlite3.Connection) -> str:
        # Insert the training result
        query = """
        INSERT INTO training_results
        (result_id, job_id, model_id, config_id, metrics, duration)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (result_id, job_id, model_id, config_id, metrics_json, duration)
        execute_query(query=query, connection=conn_to_use, params=params)
        logger.info(f"Created training result: {result_id} for job: {job_id}")

        # Update the job with the result_id
        update_query = "UPDATE jobs SET result_id = ? WHERE job_id = ?"
        execute_query(query=update_query, connection=conn_to_use, params=(result_id, job_id))
        logger.info(f"Updated job {job_id} with result_id: {result_id}")

        # Auto-activate best config
        auto_activate_best_config_if_enabled(connection=conn_to_use)

        # Auto-activate best model
        if model_id:
            auto_activate_best_model_if_enabled(connection=conn_to_use)
        return result_id

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    return _create_training_result_operation(connection)


def create_tuning_result(
    job_id: str,
    config_id: str,
    metrics: dict[str, Any],
    duration: int | None,
    connection: sqlite3.Connection = None,
) -> str:
    """Creates a record for a completed tuning job."""
    result_id = str(uuid.uuid4())
    metrics_json = json.dumps(metrics, default=json_default_serializer)

    def _create_tuning_result_operation(conn_to_use: sqlite3.Connection) -> str:
        query = """
        INSERT INTO tuning_results
        (result_id, job_id, config_id, metrics, duration)
        VALUES (?, ?, ?, ?, ?)
        """
        params = (result_id, job_id, config_id, metrics_json, duration)
        execute_query(query, params=params, connection=conn_to_use)
        logger.info(f"Created tuning result: {result_id} for job: {job_id}")
        return result_id

    return _create_tuning_result_operation(connection)


def create_prediction_result(
    job_id: str,
    prediction_month: str,
    model_id: str | None = None,
    output_path: str | None = None,
    summary_metrics: dict[str, Any] | None = None,
    connection: sqlite3.Connection = None,
) -> str:
    """
    Create or update a prediction result record.
    If a record for the job_id and prediction_month exists, it's updated.
    Otherwise, a new one is created.
    """
    result_id = generate_id()

    # Check if a result already exists for this job_id and month
    check_query = "SELECT result_id FROM prediction_results WHERE job_id = ? AND prediction_month = ?"
    existing_result = execute_query(
        check_query, params=(job_id, prediction_month), connection=connection
    )

    if existing_result:
        # Update existing record
        result_id = existing_result["result_id"]
        query = """
            UPDATE prediction_results
            SET model_id = COALESCE(?, model_id),
                output_path = COALESCE(?, output_path),
                summary_metrics = COALESCE(?, summary_metrics)
            WHERE result_id = ?
        """
        params = (
            model_id,
            output_path,
            json.dumps(summary_metrics, default=json_default_serializer)
            if summary_metrics
            else None,
            result_id,
        )
        logger.info(f"Updating prediction result for job {job_id} and month {prediction_month}")
    else:
        # Insert new record
        query = """
        INSERT INTO prediction_results (result_id, job_id, model_id, output_path, summary_metrics, prediction_month)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            result_id,
            job_id,
            model_id,
            output_path,
            json.dumps(summary_metrics, default=json_default_serializer)
            if summary_metrics
            else None,
            prediction_month.isoformat() if isinstance(prediction_month, date) else prediction_month,
        )
        logger.info(f"Creating new prediction result for job {job_id} and month {prediction_month}")

    try:
        execute_query(query, params=params, connection=connection)

        # Update the job with the result_id
        update_query = "UPDATE jobs SET result_id = ? WHERE job_id = ?"
        execute_query(update_query, params=(result_id, job_id), connection=connection)
        logger.info(f"Updated job {job_id} with result_id: {result_id}")

        return result_id
    except DatabaseError:
        logger.error(f"Failed to create/update prediction result for job {job_id}")
        raise


def insert_predictions(
    result_id: str,
    model_id: str,
    prediction_month: date,
    df: pd.DataFrame,
    connection: sqlite3.Connection = None,
):
    def _insert_predictions_operation(conn_to_use: sqlite3.Connection):
        timestamp = datetime.now().isoformat()
        predictions_data = []

        for _, row in df.iterrows():
            multiindex_id = get_or_create_multiindex_id(
                barcode=row["barcode"],
                artist=row["artist"],
                album=row["album"],
                cover_type=row["cover_type"],
                price_category=row["price_category"],
                release_type=row["release_type"],
                recording_decade=row["recording_decade"],
                release_decade=row["release_decade"],
                style=row["style"],
                record_year=int(row["record_year"]),
                connection=conn_to_use,
            )

            prediction_row = (
                result_id,
                multiindex_id,
                prediction_month,
                model_id,
                row["0.05"],
                row["0.25"],
                row["0.5"],
                row["0.75"],
                row["0.95"],
                timestamp,
            )
            predictions_data.append(prediction_row)

        execute_many(
            """
            INSERT OR REPLACE INTO fact_predictions
            (result_id, multiindex_id, prediction_month, model_id, quantile_05, quantile_25, quantile_50, quantile_75, quantile_95, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            predictions_data,
            conn_to_use,
        )

        return {"result_id": result_id, "predictions_count": len(df)}

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    return _insert_predictions_operation(connection)


def get_next_prediction_month(connection: sqlite3.Connection = None) -> date:
    """
    Finds the last month with complete data in fact_sales and returns the next month.
    """
    query = """
        SELECT
            strftime('%Y-%m', data_date) as month,
            COUNT(DISTINCT strftime('%d', data_date)) as distinct_days,
            julianday(strftime('%Y-%m-01', data_date, '+1 month')) - julianday(strftime('%Y-%m-01', data_date)) as days_in_month
        FROM fact_sales
        GROUP BY month
        HAVING distinct_days = days_in_month
        ORDER BY month DESC
        LIMIT 1;
    """
    try:
        result = execute_query(query, connection=connection)
        if result and result["month"]:
            last_full_month = datetime.strptime(result["month"], "%Y-%m").date()
            # Return the next month
            return (last_full_month.replace(day=1) + timedelta(days=32)).replace(day=1)
        else:
            # If no full month is found, default to the month after the latest data point
            latest_data_query = "SELECT MAX(data_date) as max_date FROM fact_sales"
            latest_data_result = execute_query(latest_data_query, connection=connection)
            if latest_data_result and latest_data_result["max_date"]:
                max_date = date.fromisoformat(latest_data_result["max_date"])
                return (max_date.replace(day=1) + timedelta(days=32)).replace(day=1)
            # Fallback to the current month's next month if no data exists at all
            today = date.today()
            return (today.replace(day=1) + timedelta(days=32)).replace(day=1)

    except (DatabaseError, TypeError, ValueError) as e:
        logger.error(f"Failed to get next prediction month: {e}")
        # Fallback in case of any error
        today = date.today()
        return (today.replace(day=1) + timedelta(days=32)).replace(day=1)


def get_latest_prediction_month(connection: sqlite3.Connection = None) -> date | None:
    """
    Get the most recent prediction_month from the prediction_results table.
    """
    query = "SELECT MAX(prediction_month) as latest_month FROM prediction_results"
    try:
        result = execute_query(query, connection=connection)
        if result and result["latest_month"]:
            return date.fromisoformat(result["latest_month"])
        return None
    except (DatabaseError, TypeError, ValueError) as e:
        logger.error(f"Failed to get latest prediction month: {e}")
        return None


# SQLite default limit is 999, use 900 for safety
# This can be overridden by settings.sqlite_max_variables
SQLITE_MAX_VARIABLES = 900


def get_batch_size() -> int:
    """Get the configured batch size for SQLite queries."""
    try:
        from deployment.app.config import get_settings
        return get_settings().sqlite_max_variables
    except ImportError:
        # Fallback to default if settings not available
        return SQLITE_MAX_VARIABLES


@retry_with_backoff(max_tries=3, base_delay=1.0, max_delay=10.0, component="database_batching")
def execute_query_with_batching(
    query_template: str,
    ids: list[Any],
    batch_size: int | None = None,
    connection: sqlite3.Connection | None = None,
    fetchall: bool = True,
    placeholder_name: str = "placeholders"
) -> list[Any]:
    """
    Execute query with IN clause using batching to avoid SQLite variable limit.

    Args:
        query_template: SQL query with {placeholder_name} placeholder for IN clause
        ids: List of IDs to use in IN clause
        batch_size: Maximum number of IDs per batch (default: 900)
        connection: Database connection
        fetchall: Whether to fetch all results
        placeholder_name: Name of placeholder in query_template (default: "placeholders")

    Returns:
        Combined results from all batches

    Example:
        query = "SELECT * FROM table WHERE id IN ({placeholders})"
        results = execute_query_with_batching(query, [1, 2, 3, ...], connection=conn)
    """
    if not ids:
        return []

    # Use configured batch size if not specified
    if batch_size is None:
        batch_size = get_batch_size()

    all_results = []

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        placeholders = ", ".join("?" * len(batch_ids))

        # Replace the placeholder in the query template
        query = query_template.format(**{placeholder_name: placeholders})

        try:
            batch_results = execute_query(
                query, params=tuple(batch_ids), fetchall=fetchall, connection=connection
            )
            if batch_results:
                all_results.extend(batch_results)

        except Exception as e:
            logger.error(f"Error executing batch query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Batch size: {len(batch_ids)}")
            raise

    return all_results


@retry_with_backoff(max_tries=3, base_delay=1.0, max_delay=10.0, component="database_many_batching")
def execute_many_with_batching(
    query: str,
    params_list: list[tuple],
    batch_size: int | None = None,
    connection: sqlite3.Connection | None = None
) -> None:
    """
    Execute many queries using batching to avoid SQLite variable limit.

    Args:
        query: SQL query to execute multiple times
        params_list: List of parameter tuples
        batch_size: Maximum number of parameters per batch
        connection: Database connection
    """
    if not params_list:
        return

    # Use configured batch size if not specified
    if batch_size is None:
        batch_size = get_batch_size()

    for i in range(0, len(params_list), batch_size):
        batch_params = params_list[i:i + batch_size]

        try:
            execute_many(query, params_list=batch_params, connection=connection)
        except Exception as e:
            logger.error(f"Error executing batch insert/update: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Batch size: {len(batch_params)}")
            raise


def split_ids_for_batching(ids: list[Any], batch_size: int | None = None) -> list[list[Any]]:
    """
    Split a list of IDs into batches for processing.

    Args:
        ids: List of IDs to split
        batch_size: Maximum size of each batch

    Returns:
        List of batches (each batch is a list of IDs)
    """
    if not ids:
        return []

    # Use configured batch size if not specified
    if batch_size is None:
        batch_size = get_batch_size()

    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


def check_batch_size_needed(ids: list[Any], batch_size: int | None = None) -> bool:
    """
    Check if batching is needed for the given list of IDs.

    Args:
        ids: List of IDs to check
        batch_size: Maximum allowed size

    Returns:
        True if batching is needed, False otherwise
    """
    # Use configured batch size if not specified
    if batch_size is None:
        batch_size = get_batch_size()

    return len(ids) > batch_size


def get_data_upload_result(
    result_id: str, connection: sqlite3.Connection = None
) -> dict:
    """Get data upload result by ID"""
    query = "SELECT * FROM data_upload_results WHERE result_id = ?"
    return execute_query(query=query, connection=connection, params=(result_id,))


def get_training_results(
    result_id: str | None = None, limit: int = 100, connection: sqlite3.Connection = None
) -> dict | list[dict]:
    """
    Get training result(s) by ID or a list of recent results.
    If result_id is provided, returns a single dict.
    If result_id is None, returns a list of dicts, ordered by creation date and limited.
    """
    if result_id:
        query = "SELECT * FROM training_results WHERE result_id = ?"
        return execute_query(query, params=(result_id,), connection=connection)
    else:
        query = "SELECT * FROM training_results ORDER BY created_at DESC LIMIT ?"
        return execute_query(query, params=(limit,), fetchall=True, connection=connection)


def get_prediction_result(
    result_id: str, connection: sqlite3.Connection = None
) -> dict:
    """Get prediction result by ID"""
    query = "SELECT * FROM prediction_results WHERE result_id = ?"
    return execute_query(query=query, connection=connection, params=(result_id,))


def get_prediction_results_by_month(
    prediction_month: str, model_id: str | None = None, connection: sqlite3.Connection = None
) -> list[dict]:
    """Get all prediction results for a specific month, optionally filtered by model_id."""
    query = "SELECT * FROM prediction_results WHERE prediction_month = ?"
    params = [prediction_month]

    if model_id:
        query += " AND model_id = ?"
        params.append(model_id)

    query += " ORDER BY prediction_month DESC"
    return execute_query(query, tuple(params), fetchall=True, connection=connection)


def get_predictions(
    job_ids: list[str], 
    model_id: str | None = None, 
    prediction_month: date | None = None,
    connection: sqlite3.Connection = None
) -> list[dict]:
    """
    Extract prediction data for the given training jobs.

    Args:
        job_ids: List of training job IDs
        model_id: Optional model_id for filtering
        prediction_month: Optional prediction month for filtering
        connection: Optional existing database connection

    Returns:
        List of dictionaries with prediction data
    """
    if not job_ids:
        return []

    job_ids_placeholder = ",".join(["?" for _ in job_ids])

    # Get prediction results for these jobs
    base_query = f"""
        SELECT DISTINCT pr.result_id, pr.job_id, pr.model_id
        FROM prediction_results pr
        WHERE pr.job_id IN ({job_ids_placeholder})
    """

    params_list: list[Any] = job_ids.copy()

    if model_id:
        base_query += " AND pr.model_id = ?"
        params_list.append(model_id)

    if prediction_month:
        base_query += " AND pr.prediction_month = ?"
        params_list.append(prediction_month.isoformat() if isinstance(prediction_month, date) else prediction_month)

    prediction_results = execute_query(
        base_query, params=tuple(params_list), fetchall=True, connection=connection
    )

    if not prediction_results:
        logger.warning("No prediction results found for training jobs")
        return []

    result_ids = [pr["result_id"] for pr in prediction_results]
    result_ids_placeholder = ",".join(["?" for _ in result_ids])

    # Get actual prediction data
    predictions_query = f"""
        SELECT
            fp.result_id,
            dmm.multiindex_id,
            dmm.barcode,
            dmm.artist,
            dmm.album,
            dmm.cover_type,
            dmm.price_category,
            dmm.release_type,
            dmm.recording_decade,
            dmm.release_decade,
            dmm.style,
            dmm.record_year,
            fp.model_id,
            fp.prediction_month,
            fp.quantile_05,
            fp.quantile_25,
            fp.quantile_50,
            fp.quantile_75,
            fp.quantile_95
        FROM fact_predictions fp
        JOIN dim_multiindex_mapping dmm ON fp.multiindex_id = dmm.multiindex_id
        WHERE fp.result_id IN ({result_ids_placeholder})
    """

    query_params = result_ids.copy()
    
    if model_id:
        predictions_query += " AND fp.model_id = ?"
        query_params.append(model_id)

    if prediction_month:
        predictions_query += " AND fp.prediction_month = ?"
        query_params.append(prediction_month.isoformat() if isinstance(prediction_month, date) else prediction_month)

    predictions_query += " ORDER BY dmm.artist, dmm.album, fp.prediction_month"

    return execute_query(
        predictions_query, params=tuple(query_params), fetchall=True, connection=connection
    )

def get_report_result(result_id: str, connection: sqlite3.Connection = None) -> dict:
    """
    Get report result by ID"""
    query = "SELECT * FROM report_results WHERE result_id = ?"
    return execute_query(query=query, connection=connection, params=(result_id,))


# Processing runs management


def create_processing_run(
    start_time: datetime,
    status: str,
    source_files: str,
    end_time: datetime = None,
    connection: sqlite3.Connection = None,
) -> int:
    """
    Create a processing run record

    Args:
        start_time: Start time of processing
        status: Status of the run
        source_files: Comma-separated list of source files
        end_time: Optional end time of processing
        connection: Optional existing database connection to use

    Returns:
        Generated run ID
    """
    query = """
    INSERT INTO processing_runs (start_time, status, source_files, end_time)
    VALUES (?, ?, ?, ?)
    """

    params = (
        start_time.isoformat(),
        status,
        source_files,
        end_time.isoformat() if end_time else None,
    )

    try:
        execute_query(query, params=params, connection=connection)

        # Get the last inserted ID
        result = execute_query(
            "SELECT last_insert_rowid() as run_id", connection=connection
        )
        return result["run_id"]
    except DatabaseError:
        logger.error("Failed to create processing run")
        raise


def update_processing_run(
    run_id: int,
    status: str,
    end_time: datetime = None,
    connection: sqlite3.Connection = None,
) -> None:
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
        execute_query(query, params=tuple(params), connection=connection)
    except DatabaseError:
        logger.error(f"Failed to update processing run {run_id}")
        raise


# MultiIndex mapping functions


def get_or_create_multiindex_ids_batch(
    tuples_to_process: list[tuple],
    connection: sqlite3.Connection
) -> dict[tuple, int]:
    """
    Efficiently gets or creates multiple multi-index IDs in a single batch.

    Args:
        tuples_to_process: A list of unique tuples, each representing a multi-index.
        connection: An active sqlite3.Connection object.

    Returns:
        A dictionary mapping each input tuple to its integer multiindex_id.
    """
    if not tuples_to_process:
        return {}

    def _batch_operation(conn_to_use: sqlite3.Connection) -> dict[tuple, int]:
        execute_query("CREATE TEMP TABLE _tuples_to_find (barcode TEXT, artist TEXT, album TEXT, cover_type TEXT, price_category TEXT, release_type TEXT, recording_decade TEXT, release_decade TEXT, style TEXT, record_year INTEGER)", conn_to_use)

        try:
            execute_many("INSERT INTO _tuples_to_find VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuples_to_process, conn_to_use)

            query_existing = """
                SELECT t.*, m.multiindex_id
                FROM _tuples_to_find t
                JOIN dim_multiindex_mapping m ON
                    m.barcode = t.barcode AND
                    m.artist = t.artist AND
                    m.album = t.album AND
                    m.cover_type = t.cover_type AND
                    m.price_category = t.price_category AND
                    m.release_type = t.release_type AND
                    m.recording_decade = t.recording_decade AND
                    m.release_decade = t.release_decade AND
                    m.style = t.style AND
                    m.record_year = t.record_year
            """
            existing_rows = execute_query(query_existing, conn_to_use, fetchall=True) or []

            existing_tuples = set()
            id_map = {}
            for row in existing_rows:
                existing_tuple = tuple(row[col] for col in row.keys() if col != 'multiindex_id')
                existing_tuples.add(existing_tuple)
                id_map[existing_tuple] = row['multiindex_id']

            new_tuples = [t for t in tuples_to_process if tuple(t) not in existing_tuples]

            if new_tuples:
                insert_query = """
                INSERT OR IGNORE INTO dim_multiindex_mapping (
                    barcode, artist, album, cover_type, price_category,
                    release_type, recording_decade, release_decade, style, record_year
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                execute_many(insert_query, new_tuples, conn_to_use)

                all_rows = execute_query(query_existing, conn_to_use, fetchall=True) or []
                for row in all_rows:
                    full_tuple = tuple(row[col] for col in row.keys() if col != 'multiindex_id')
                    id_map[full_tuple] = row['multiindex_id']

        finally:
            execute_query("DROP TABLE _tuples_to_find", conn_to_use)
        return id_map

    try:
        return _batch_operation(connection)
    except Exception as e:
        logger.error(f"Error in get_or_create_multiindex_ids_batch: {e}", exc_info=True)
        raise


def get_or_create_multiindex_id(
    barcode: str,
    artist: str,
    album: str,
    cover_type: str,
    price_category: str,
    release_type: str,
    recording_decade: str,
    release_decade: str,
    style: str,
    record_year: int,
    connection: sqlite3.Connection = None,
) -> int:
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
        barcode,
        artist,
        album,
        cover_type,
        price_category,
        release_type,
        recording_decade,
        release_decade,
        style,
        record_year,
    )

    try:
        existing = existing = execute_query(query, params=params, connection=connection)

        if existing:
            return existing["multiindex_id"]

        # Create new mapping
        insert_query = """
        INSERT INTO dim_multiindex_mapping (
            barcode, artist, album, cover_type, price_category,
            release_type, recording_decade, release_decade, style, record_year
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        execute_query(insert_query, params=params, connection=connection)

        # Get the new ID
        result = execute_query(
            "SELECT last_insert_rowid() as multiindex_id", connection=connection
        )
        return result["multiindex_id"]

    except DatabaseError:
        logger.error("Failed to get or create multiindex mapping")
        raise


def get_configs(
    limit: int = 5, connection: sqlite3.Connection = None
) -> list[dict[str, Any]]:
    """
    Retrieves a list of configs ordered by creation date.

    Args:
        limit: Maximum number of configs to return
        connection: Optional existing database connection to use

    Returns:
        List of configs with their details
    """
    def _get_configs_operation(conn_to_use: sqlite3.Connection) -> list[dict[str, Any]]:
        results = execute_query(
            """
            SELECT config_id, config, created_at, is_active
            FROM configs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            conn_to_use,
            (limit,),
            fetchall=True
        ) or []

        for result in results:
            if "config" in result and result["config"]:
                result["config"] = json.loads(result["config"])

        return results

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    return _get_configs_operation(connection)


def insert_report_features(
    features_data: list[tuple], connection: sqlite3.Connection = None
) -> None:
    """
    Inserts a batch of report features into the report_features table.
    """
    def _insert_report_features_operation(conn_to_use: sqlite3.Connection):
        execute_many(
            """
            INSERT OR REPLACE INTO report_features (
                data_date, multiindex_id, availability, confidence, masked_mean_sales_items, masked_mean_sales_rub, lost_sales, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            features_data,
            connection=conn_to_use,
        )

    # This function now *always* expects an external connection.
    # The caller (DataAccessLayer) is responsible for transaction management.
    _insert_report_features_operation(connection)


def get_report_features(
    multiidx_ids: list[int] | None = None,
    prediction_month: date | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    feature_subset: list[str] | None = None,
    connection: sqlite3.Connection = None,
) -> list[dict]:
    """
    Retrieves a specific subset of report features and associated product
    attributes, filtered by a specific month or a date range.

    If `multiidx_ids` is provided, it filters for those specific IDs. Otherwise,
    it fetches features for all available items within the date filter.

    Args:
        multiidx_ids: Optional list of multiindex_ids to retrieve features for.
        prediction_month: The specific month for which to retrieve features (YYYY-MM-DD).
        start_date: The start of the date range (inclusive).
        end_date: The end of the date range (inclusive).
        feature_subset: Optional list of specific features to retrieve from report_features
                        (e.g., ["avg_sales_items", "lost_sales_rub"]). If None, all are fetched.
        connection: An active database connection.

    Returns:
        A list of dictionaries, where each dictionary contains product attributes
        and its requested features. Returns an empty list if no matching data is found.

    Raises:
        ValueError: If no date filter (prediction_month or start/end_date) is provided.
    """
    if prediction_month:
        start_date = prediction_month.replace(day=1) 
        end_date =(prediction_month.replace(day=1) + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date()
    elif not (start_date and end_date):
        raise ValueError("A date filter must be provided. Use 'prediction_month' or both 'start_date' and 'end_date'.")

    # Build the SELECT clause
    select_parts = ["dmm.*", "rf.multiindex_id as rf_multiindex_id"]  # Alias to avoid collision
    if feature_subset:
        selected_features = list(
            EXPECTED_REPORT_FEATURES_SET.intersection(feature_subset)
        )
    else:
        selected_features = EXPECTED_REPORT_FEATURES
    
    for feature in selected_features:
        select_parts.append(f"rf.{feature}")

    select_clause = "SELECT " + ", ".join(select_parts)
    from_clause = """
    FROM report_features rf
    JOIN dim_multiindex_mapping dmm ON rf.multiindex_id = dmm.multiindex_id
    """

    # Date filtering logic
    where_clauses = []
    params = []
    if start_date:
        where_clauses.append("rf.data_date >= ?")
        params.append(start_date.strftime("%Y-%m-%d"))
    if end_date:
        where_clauses.append("rf.data_date <= ?")
        params.append(end_date.strftime("%Y-%m-%d"))
    # Handle multiidx_ids with batching if provided

    if multiidx_ids:
        # Use existing batching function for multiidx_ids filtering
        where_clauses.append("rf.multiindex_id IN ({placeholders})")
        query_template = f"{select_clause} {from_clause} WHERE {' AND '.join(where_clauses)}"

        # Execute with batching using the existing function
        return execute_query_with_batching(
            query_template=query_template,
            ids=multiidx_ids,
            batch_size=get_batch_size(),
            connection=connection,
            fetchall=True,
            placeholder_name="placeholders"
        )
    else:
        # No multiidx_ids, single query for all items in the date range
        if where_clauses:
            query_template = f"{select_clause} {from_clause} WHERE {' AND '.join(where_clauses)}"
        else:
            query_template = f"{select_clause} {from_clause}"

        try:
            results = execute_query(query=query_template, params=tuple(params), fetchall=True, connection=connection)
            return results or []
        except DatabaseError as e:
            logger.error(f"Failed to get report features: {e}")
            raise


def delete_configs_by_ids(
    config_ids: list[str], connection: sqlite3.Connection = None
) -> dict[str, Any]:
    """
    Deletes multiple config records by their IDs, skipping active configs.

    Args:
        config_ids: List of config IDs to delete.
        connection: Optional existing database connection.

    Returns:
        A dictionary with a deletion summary.
    """
    if not config_ids:
        return {"deleted_count": 0, "skipped_count": 0, "skipped_configs": []}

    summary = {"deleted_count": 0, "skipped_count": 0, "skipped_models": []}
    conn = connection # Use the provided connection directly

    # Use batching to avoid SQLite variable limit
    try:
         # Find which configs are active
        active_query = "SELECT config_id FROM configs WHERE config_id IN ({placeholders}) AND is_active = 1"
        active_results = execute_query_with_batching(
            active_query, config_ids, connection=conn
        )
        active_configs = {row["config_id"] for row in active_results}

        summary["skipped_configs"] = list(active_configs)
        summary["skipped_count"] = len(active_configs)

        configs_to_delete = [cid for cid in config_ids if cid not in active_configs]

        if not configs_to_delete:
            return summary

        # Find jobs using these configs
        jobs_query = "SELECT config_id FROM jobs WHERE config_id IN ({placeholders})"
        jobs_results = execute_query_with_batching(
            jobs_query, configs_to_delete, connection=conn
        )
        used_configs_in_jobs = {row["config_id"] for row in jobs_results}

        # Also check training_results
        training_query = "SELECT config_id FROM training_results WHERE config_id IN ({placeholders})"
        training_results = execute_query_with_batching(
            training_query, configs_to_delete, connection=conn
        )
        used_configs_in_results = {row["config_id"] for row in training_results}

        used_configs = used_configs_in_jobs.union(used_configs_in_results)

        # Add used configs to the skipped list
        newly_skipped = used_configs.difference(set(summary["skipped_configs"]))
        if newly_skipped:
            summary["skipped_configs"].extend(list(newly_skipped))
            summary["skipped_count"] += len(newly_skipped)

        # Recalculate configs to delete
        final_configs_to_delete = [
            cid for cid in configs_to_delete if cid not in used_configs
        ]

        if not final_configs_to_delete:
            return summary

        # Delete the unused, non-active configs using batching

        deleted_count = 0
        for batch in split_ids_for_batching(final_configs_to_delete):
            batch_placeholders = ",".join("?" for _ in batch)
            execute_query(
                execute_query(
            query=f"DELETE FROM configs WHERE config_id IN ({batch_placeholders})",
            connection=conn,
            params=batch,
        )
            )
            deleted_count += len(batch)  # Since execute_query doesn't return rowcount, we use batch length

        summary["deleted_count"] = deleted_count

    except Exception as e:
        logger.error(f"Error deleting configs by IDs: {e}", exc_info=True)
    finally:
        pass # Removed conn_created and conn.close() as DAL manages connection

    return summary


def get_all_models(
    limit: int = 100,
    include_active_status: bool = True,
    connection: sqlite3.Connection = None,
) -> list[dict[str, Any]]:
    """
    Retrieves a list of all models with their details.

    Args:
        limit: Maximum number of models to return
        include_active_status: Whether to include the active status in the results
        connection: Required existing database connection to use

    Returns:
        List of models with their details
    """
    if not connection:
        raise ValueError("connection parameter is required")

    def _get_all_models_operation(conn_to_use: sqlite3.Connection) -> list[dict[str, Any]]:
        results = execute_query(
            query="""
            SELECT model_id, job_id, model_path, created_at, metadata, is_active
            FROM models
            ORDER BY created_at DESC
            LIMIT ?
            """,
            connection=conn_to_use,
            params=(limit,),
            fetchall=True
        ) or []

        for result in results:
            if "metadata" in result and result["metadata"]:
                result["metadata"] = json.loads(result["metadata"])

        return results

    return _get_all_models_operation(connection)


def delete_models_by_ids(
    model_ids: list[str], connection: sqlite3.Connection
) -> dict[str, Any]:
    """
    Deletes multiple model records by their IDs, skipping active models.

    Args:
        model_ids: List of model IDs to delete.
        connection: An existing database connection.

    Returns:
        A dictionary with a deletion summary.
    """
    if not model_ids:
        return {"deleted_count": 0, "skipped_count": 0, "skipped_models": []}

    summary = {"deleted_count": 0, "skipped_count": 0, "skipped_models": []}

    # Use batching to avoid SQLite variable limit

    # Find which models are active
    active_models_query = "SELECT model_id FROM models WHERE is_active = 1 AND model_id IN ({placeholders})"
    active_models = execute_query_with_batching(
        active_models_query, model_ids, connection=connection
    )
    active_model_ids = {m["model_id"] for m in active_models}

    models_to_delete = []
    for model_id in model_ids:
        if model_id in active_model_ids:
            summary["skipped_count"] += 1
            summary["skipped_models"].append(model_id)
            logger.warning(f"Skipping deletion of active model: {model_id}")
        else:
            models_to_delete.append(model_id)

    if models_to_delete:
        # Delete dependent records from training_results first
        delete_training_results_query = "DELETE FROM training_results WHERE model_id IN ({placeholders})"
        execute_query_with_batching(
            delete_training_results_query, models_to_delete, connection=connection
        )

        # Delete the model records
        delete_models_query = "DELETE FROM models WHERE model_id IN ({placeholders})"
        execute_query_with_batching(
            delete_models_query, models_to_delete, connection=connection
        )
        summary["deleted_count"] = len(models_to_delete)

        # Delete associated model files
        for model_id in models_to_delete:
            delete_model_record_and_file(model_id, connection=connection)

    return summary


def auto_activate_best_config_if_enabled(connection: sqlite3.Connection = None) -> bool:
    """
    Automatically activates the best config by metric if auto_select_best_configs is enabled.

    Args:
        connection: Optional existing database connection to use

    Returns:
        bool: True if activation was performed, False if disabled or no config found

    Raises:
        DatabaseError: If database operations fail
    """
    settings = get_settings()

    if not settings.auto_select_best_configs:
        return False

    try:
        best_config = get_best_config_by_metric(
            metric_name=settings.default_metric,
            higher_is_better=settings.default_metric_higher_is_better,
            connection=connection,
        )

        if best_config and best_config.get("config_id"):
            config_id = best_config["config_id"]
            logger.info(f"Auto-activating best config by {settings.default_metric}: {config_id}")

            success = set_config_active(config_id, connection=connection)
            if success:
                logger.info(f"Successfully auto-activated config: {config_id}")
                return True
            else:
                logger.warning(f"Failed to auto-activate config: {config_id}")
                return False
        else:
            logger.warning(
                f"Auto-activation enabled, but no best config found by metric '{settings.default_metric}'"
            )
            return False

    except Exception as e:
        logger.error(f"Error during auto-activation of best config: {e}", exc_info=True)
        # Don't re-raise - auto-activation should not break the main flow
        return False


def auto_activate_best_model_if_enabled(connection: sqlite3.Connection = None) -> bool:
    """
    Automatically activates the best model by metric if auto_select_best_model is enabled.

    Args:
        connection: Optional existing database connection to use

    Returns:
        bool: True if activation was performed, False if disabled or no model found

    Raises:
        DatabaseError: If database operations fail
    """
    settings = get_settings()

    if not settings.auto_select_best_model:
        return False

    try:
        best_model = get_best_model_by_metric(
            metric_name=settings.default_metric,
            higher_is_better=settings.default_metric_higher_is_better,
            connection=connection,
        )

        if best_model and best_model.get("model_id"):
            model_id = best_model["model_id"]
            logger.info(f"Auto-activating best model by {settings.default_metric}: {model_id}")

            success = set_model_active(model_id, connection=connection)
            if success:
                logger.info(f"Successfully auto-activated model: {model_id}")
                return True
            else:
                logger.warning(f"Failed to auto-activate model: {model_id}")
                return False
        else:
            logger.warning(
                f"Auto-activation enabled, but no best model found by metric '{settings.default_metric}'"
            )
            return False

    except Exception as e:
        logger.error(f"Error during auto-activation of best model: {e}", exc_info=True)
        # Don't re-raise - auto-activation should not break the main flow
        return False


def get_effective_config(settings, connection: sqlite3.Connection, logger=None):
    """
    Determines the effective configuration by first trying to load the active config
    from the database. If no active config is found or an error occurs, it falls back
    to the provided settings.

    Args:
        settings: The current application settings object.
        logger: Optional logger instance.
        connection: An existing database connection.

    Returns:
        A dictionary representing the effective configuration.
    """
    active_config_data = get_active_config(connection=connection)
    if active_config_data:
        if logger:
            logger.info(f"Found active config: {active_config_data['config_id']}")
        return active_config_data
    # Fallback: best by metric
    metric_name = getattr(settings, "default_metric", None)
    higher_is_better = getattr(settings, "default_metric_higher_is_better", True)
    best_config_data = get_best_config_by_metric(
        metric_name, higher_is_better, connection=connection
    )
    if best_config_data:
        if logger:
            logger.info(
                f"Using best config by {metric_name}: {best_config_data['config_id']}"
            )
        return best_config_data
    error_msg = "No active config and no best config by metric available"
    if logger:
        logger.error(error_msg)
    raise ValueError(error_msg)


def get_tuning_results(
    connection: sqlite3.Connection,
    result_id: str | None = None,
    metric_name: str | None = None,
    higher_is_better: bool | None = None,
    limit: int = 100,
) -> dict | list[dict]:
    """
    Get tuning result(s).
    - If result_id is provided, fetches a single result by its ID.
    - If result_id is None, fetches a list of results, sorted by a specified metric
      or by creation date if no metric is provided.

    Args:
        result_id: The specific result ID to fetch.
        metric_name: The metric to sort by (e.g., 'val_loss').
        higher_is_better: Direction for sorting the metric. Required if metric_name is set.
        limit: The maximum number of results to return in list view.
        connection: Optional existing database connection.

    Returns:
        A single dictionary if result_id is provided, otherwise a list of dictionaries.
    """
    # Fetch a single result by ID
    if result_id:
        query = "SELECT * FROM tuning_results WHERE result_id = ?"
        return execute_query(query, params=(result_id,), fetchall=False, connection=connection)

    # Fetch a list of results, with optional sorting
    params = [limit]

    if metric_name:
        if higher_is_better is None:
            raise ValueError("`higher_is_better` must be specified when `metric_name` is provided.")
        if metric_name not in ALLOWED_METRICS:
            raise ValueError(f"Invalid metric name: {metric_name}")

        order_direction = "DESC" if higher_is_better else "ASC"
        json_path = f"'$.{metric_name}'"

        query = f"""
            SELECT *, json_extract(metrics, {json_path}) as metric_value
            FROM tuning_results
            WHERE json_valid(metrics) = 1 AND json_extract(metrics, {json_path}) IS NOT NULL
            ORDER BY metric_value {order_direction}
            LIMIT ?
        """
    else:
        # Default sorting by creation date if no metric is specified
        query = "SELECT * FROM tuning_results ORDER BY created_at DESC LIMIT ?"

    try:
        results = execute_query(query, params=tuple(params), fetchall=True, connection=connection)
        return results if results is not None else []
    except DatabaseError as e:
        logger.error(f"Failed to get tuning results: {e}")
        raise


def get_top_configs(
    connection: sqlite3.Connection,
    limit: int = 5,
    metric_name: str | None = None,
    higher_is_better: bool = True,
    include_active: bool = True,
    source: str | None = None,  # 'manual' | 'tuning' | NULL
) -> list[dict[str, Any]]:
    """Return best historical configs for seeding tuning.

    Selection rules:
    1. Optionally include the currently active config first.
    2. Then order configs by metric from both *training_results* and *tuning_results* tables.
       Rows where metric is NULL are ignored.
    3. Falls back to most recently created configs if no metrics.
    """
    metric_name = metric_name or "val_MIC"
    if metric_name not in ALLOWED_METRICS:
        logger.error("get_top_configs: Invalid metric name provided. Raising error.")
        raise ValueError("Invalid metric_name")
    order = "DESC" if higher_is_better else "ASC"

    json_path = f"'$.{metric_name}'"

    sql = f"""
        WITH config_metrics AS (
            SELECT
                c.config_id,
                c.config,
                c.created_at,
                c.is_active,
                c.source,
                (SELECT JSON_EXTRACT(tr.metrics, {json_path})
                 FROM training_results tr
                 WHERE tr.config_id = c.config_id
                 AND JSON_VALID(tr.metrics) = 1 AND JSON_EXTRACT(tr.metrics, {json_path}) IS NOT NULL
                 ORDER BY JSON_EXTRACT(tr.metrics, {json_path}) {order} LIMIT 1) as training_metric,
                (SELECT JSON_EXTRACT(tu.metrics, {json_path})
                 FROM tuning_results tu
                 WHERE tu.config_id = c.config_id
                 AND JSON_VALID(tu.metrics) = 1 AND JSON_EXTRACT(tu.metrics, {json_path}) IS NOT NULL
                 ORDER BY JSON_EXTRACT(tu.metrics, {json_path}) {order} LIMIT 1) as tuning_metric
            FROM configs c
        ),
        final_metrics AS (
            SELECT
                config_id,
                config,
                created_at,
                is_active,
                source,
                CASE
                    WHEN training_metric IS NOT NULL AND tuning_metric IS NOT NULL THEN
                        CASE WHEN {higher_is_better} THEN
                            MAX(training_metric, tuning_metric)
                        ELSE
                            MIN(training_metric, tuning_metric)
                        END
                    WHEN training_metric IS NOT NULL THEN training_metric
                    WHEN tuning_metric IS NOT NULL THEN tuning_metric
                    ELSE NULL
                END as best_metric,
                (SELECT metrics FROM training_results WHERE config_id = config_metrics.config_id ORDER BY JSON_EXTRACT(metrics, {json_path}) {order} LIMIT 1) as training_metrics_json,
                (SELECT metrics FROM tuning_results WHERE config_id = config_metrics.config_id ORDER BY JSON_EXTRACT(metrics, {json_path}) {order} LIMIT 1) as tuning_metrics_json
            FROM config_metrics
        )
        SELECT config_id, config, created_at, is_active, source, best_metric, training_metrics_json, tuning_metrics_json
        FROM final_metrics
    """

    where_conditions = []

    if not include_active:
        where_conditions.append("is_active = 0")

    if source:
        where_conditions.append(f"source = '{source}'")

    if where_conditions:
        sql += " WHERE " + " AND ".join(where_conditions)

    sql += f" ORDER BY is_active DESC, best_metric {order} NULLS LAST, created_at DESC LIMIT ?"

    rows = execute_query(sql, connection, (limit,), fetchall=True) or []

    top_cfgs = []
    for r in rows:
        try:
            new_row = {
                "config_id": r["config_id"],
                "config": json.loads(r["config"]),
                "created_at": r["created_at"],
                "is_active": bool(r["is_active"]),
                "source": r["source"],
                "metrics": {},  # Initialize metrics as empty dict
            }

            training_metrics = json.loads(r['training_metrics_json']) if r.get('training_metrics_json') else None
            tuning_metrics = json.loads(r['tuning_metrics_json']) if r.get('tuning_metrics_json') else None


            if training_metrics:
                new_row["metrics"] = training_metrics
            elif tuning_metrics:
                new_row["metrics"] = tuning_metrics
            else:
                new_row["metrics"] = {}

            top_cfgs.append(new_row)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse config or metrics JSON for {r.get('config_id')}: {e}")
            continue
    return top_cfgs


def insert_retry_event(event: dict[str, Any], connection: sqlite3.Connection) -> None:
    """Insert a single retry event into retry_events table.

    Args:
        event: Dict with keys matching retry_events columns.
        connection: Optional existing DB connection.
    """
    query = (
        "INSERT INTO retry_events (timestamp, component, operation, attempt, max_attempts, "
        "successful, duration_ms, exception_type, exception_message) "
        "VALUES (:timestamp, :component, :operation, :attempt, :max_attempts, :successful, "
        ":duration_ms, :exception_type, :exception_message)"
    )

    event = event.copy()
    event["successful"] = 1 if event.get("successful") else 0

    try:
        execute_query(query, params=event, connection=connection)
    except Exception as e:
        logger.error(f"[insert_retry_event] Error during insert for event {event.get('operation')}: {e}", exc_info=True)
        raise


def fetch_recent_retry_events(limit: int = 1000, connection: sqlite3.Connection = None) -> list[dict[str, Any]]:
    """Fetch most recent retry events ordered oldest->newest up to *limit*."""

    query = "SELECT * FROM retry_events ORDER BY id DESC LIMIT ?"
    rows = execute_query(query, (limit,), fetchall=True, connection=connection) or []
    rows.reverse()
    return rows


def delete_features_by_table(table: str, connection: sqlite3.Connection = None) -> None:
    """Delete all records from a feature table."""
    def _delete_operation(conn_to_use: sqlite3.Connection) -> None:
        query = f"DELETE FROM {table}"
        execute_query(query, connection=conn_to_use)

    _delete_operation(connection)


def insert_features_batch(table: str, params_list: list[tuple], connection: sqlite3.Connection = None) -> None:
    """Insert a batch of feature records into the specified table."""
    def _insert_operation(conn_to_use: sqlite3.Connection) -> None:
        query = f"INSERT OR REPLACE INTO {table} (multiindex_id, data_date, value) VALUES (?, ?, ?)"
        execute_many_with_batching(query, params_list, batch_size=SQLITE_MAX_VARIABLES, connection=conn_to_use)

    _insert_operation(connection)


def get_features_by_date_range(
    table: str, start_date: str | None, end_date: str | None, connection: sqlite3.Connection = None
) -> list[dict]:
    """Generic function to get features from a table by date range."""

    def _get_operation(conn_to_use: sqlite3.Connection) -> list[dict]:
        # Determine the correct date column based on the table
        date_column = "data_date"

        query = f"SELECT * FROM {table}"
        params = []
        where_clauses = []

        if start_date:
            where_clauses.append(f"{date_column} >= ?")
            params.append(start_date)
        if end_date:
            where_clauses.append(f"{date_column} <= ?")
            params.append(end_date)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += f" ORDER BY {date_column}"

        try:
            return execute_query(query=query, params=tuple(params), connection=conn_to_use, fetchall=True) or []
        except DatabaseError as e:
            logger.error(f"Failed to get features from {table}: {e}")
            raise

    return _get_operation(connection)


def adjust_dataset_boundaries(
    start_date: date | None = None,
    end_date: date | None = None,
    connection: sqlite3.Connection = None,
) -> date | None:
    """
    Adjusts the end date for the training dataset.

    1. Validates the date range.
    2. Finds the latest available date in fact_sales within the specified range.
    3. Checks if the month of that date is complete.
    4. If the month is complete, returns the original end_date.
    5. If the month is incomplete, returns the end of the previous complete month.

    Returns:
        The adjusted end date (date) or None if no data is found.

    Raises:
        ValueError: If the date range is invalid.
    """
    # Base query to find the last date
    query = "SELECT MAX(data_date) as last_date FROM fact_sales"
    params = []

    # Add date range conditions if provided
    if start_date and end_date:
        query += " WHERE data_date BETWEEN ? AND ?"
        params.extend([start_date.isoformat(), end_date.isoformat()])
    elif start_date:
        query += " WHERE data_date >= ?"
        params.append(start_date.isoformat())
    elif end_date:
        query += " WHERE data_date <= ?"
        params.append(end_date.isoformat())

    try:
        result = execute_query(query, params=tuple(params), connection=connection)

        if not result or not result.get("last_date"):
            # No data found — return the original end_date
            logger.warning("No sales data found for date range, returning original end_date.")
            return end_date

        last_date_in_data = date.fromisoformat(result["last_date"])

        # Check if the month is complete
        next_day = last_date_in_data + timedelta(days=1)
        is_month_complete = last_date_in_data.month != next_day.month

        if is_month_complete:
            # Month is complete — return the original end_date
            return end_date
        else:
            # Month is incomplete — return the end of the previous month
            adjusted_end_date = last_date_in_data.replace(day=1) - timedelta(days=1)
            return adjusted_end_date

    except (DatabaseError, TypeError, ValueError) as e:
        logger.error(f"Failed to adjust dataset boundaries: {e}", exc_info=True)
        return end_date


def get_multiindex_mapping_by_ids(
    multiindex_ids: list[int],
    connection: sqlite3.Connection
) -> list[dict]:
    """
    Get multiindex mapping data by IDs.

    Args:
        multiindex_ids: List of multiindex IDs to get mapping for
        connection: Database connection

    Returns:
        List of dictionaries with mapping data
    """
    if not multiindex_ids:
        return []

    # Use batching to avoid SQLite variable limit

    query_template = """
    SELECT multiindex_id, barcode, artist, album, cover_type, price_category,
           release_type, recording_decade, release_decade, style, record_year
    FROM dim_multiindex_mapping
    WHERE multiindex_id IN ({placeholders})
    """

    return execute_query_with_batching(query_template, multiindex_ids, connection=connection)


def get_job_params(
    job_id: str, 
    connection: sqlite3.Connection = None, 
    param_name: str = None
) -> dict[str, Any]:
    """
    Retrieves and parses job parameters from the database.
    
    Args:
        job_id: The job ID to retrieve parameters for
        connection: Optional existing database connection to use
        param_name: Optional specific parameter name to extract (e.g., 'prediction_month')
        
    Returns:
        Dictionary of job parameters if param_name is None, or the specific parameter value
    """
    try:
        job_details = get_job(job_id, connection=connection)
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


def get_job_prediction_month(job_id: str, connection: sqlite3.Connection = None) -> date:
    """
    Retrieves and parses the prediction_month parameter from job parameters.
    
    Args:
        job_id: The job ID to retrieve prediction_month for
        connection: Optional existing database connection to use
        
    Returns:
        date object representing the prediction month, or today's date as fallback
    """
    prediction_month = get_job_params(job_id, connection=connection, param_name="prediction_month")
    
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
