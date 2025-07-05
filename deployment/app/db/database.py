import hashlib
import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from deployment.app.config import get_settings
from deployment.app.utils.retry import retry_with_backoff

logger = logging.getLogger("plastinka.database")


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
    "mae",
    "mse",
    "rmse",
    "r2",
    "mape",
    "smape",
    "median_absolute_error",
    "mean_squared_log_error",
    "explained_variance_score",
    "max_error",
    # Standardized metric names from training phases (with prefixes)
    "val_MIC",
    "val_accuracy",
    "train_accuracy",
    "train_mae",
    "train_mse",
    "train_rmse",
    "train_r2",
    "train_mape",
    "train_smape",
    "train_median_absolute_error",
    "train_mean_squared_log_error",
    "train_explained_variance_score",
    "train_max_error",
    "training_duration_seconds",
    # Metrics from Ray Tune results
    "val_MIWS",
    "val_MIWS_MIC_Ratio",
    # Add other allowed metric names as needed, especially for tuning results
    # (e.g., 'tuning_loss', 'tuning_accuracy' if Ray metrics are flattened this way)
]

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


def get_db_connection(
    db_path_override: str | Path | None = None,
    existing_connection: sqlite3.Connection | None = None,
):
    """
    Get a connection to the SQLite database.
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
        conn = sqlite3.connect(str(current_db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = dict_factory
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}", exc_info=True)
        raise DatabaseError(
            f"Database connection failed: {str(e)}", original_error=e
        ) from e


@contextmanager
def db_transaction(db_path_or_conn: str | Path | sqlite3.Connection = None):
    """
    Context manager for database transactions.

    Args:
        db_path_or_conn: Path to the database file or an existing sqlite3.Connection.
                         If None, uses the default DB_PATH from settings.

    Yields:
        sqlite3.Connection: The database connection.

    Raises:
        DatabaseError: If connection or transaction handling fails.
    """
    conn: sqlite3.Connection | None = None
    conn_created_internally = False

    try:
        if isinstance(db_path_or_conn, str | Path):
            conn = get_db_connection(
                db_path_or_conn
            )  # Assuming get_db_connection can take a path
            conn_created_internally = True
        elif isinstance(db_path_or_conn, sqlite3.Connection):
            conn = db_path_or_conn
            # Ensure foreign keys are on for provided connections if not already set by get_db_connection
            # However, get_db_connection already does this. If conn is external, we assume it's configured.
            # For safety, we could execute it, but it might interfere if the user has specific PRAGMA settings.
            # Let's assume external connections are ready or get_db_connection handles it.
        else:  # db_path_or_conn is None, use default
            conn = get_db_connection()  # Uses DB_PATH by default
            conn_created_internally = True

        if conn is None:  # Should not happen if get_db_connection raises on failure
            raise DatabaseError(
                "Failed to establish a database connection for transaction."
            )

        # Ensure PRAGMA foreign_keys = ON; is set for connections it establishes.
        # get_db_connection already handles this.

        # SQLite's default isolation_level is DEFERRED, which means a transaction
        # doesn't actually start until the first DML statement.
        # To ensure BEGIN is issued, we can set isolation_level to None
        # and manage BEGIN/COMMIT/ROLLBACK manually, or rely on Python's
        # context manager behavior with sqlite3.
        # For explicit control, setting isolation_level to None is often preferred
        # when using a context manager like this.
        # However, sqlite3's connection object itself can be used as a context manager
        # which handles this. Let's stick to explicit commit/rollback.
        # The default behavior of conn.commit() and conn.rollback() is fine.

        yield conn
        conn.commit()

    except Exception as e:
        if conn:
            logger.error(
                f"db_transaction: Exception occurred, rolling back transaction for connection id: {id(conn)}. Error: {e}",
                exc_info=True,
            )
            conn.rollback()
        # Re-raise the exception to be handled by the caller
        # If it's already a DatabaseError, re-raise it directly. Otherwise, wrap it.
        if isinstance(e, DatabaseError):
            raise
        raise DatabaseError(f"Transaction failed: {str(e)}", original_error=e) from e
    finally:
        if conn and conn_created_internally:
            conn.close()


def dict_factory(cursor, row):
    """Convert row to dictionary"""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def execute_query(
    query: str,
    params: tuple = (),
    fetchall: bool = False,
    connection: sqlite3.Connection = None,
) -> list[dict] | dict | None:
    """
    Execute a query and optionally return results

    Args:
        query: SQL query with placeholders (?, :name)
        params: Parameters for the query
        fetchall: Whether to fetch all results or just one
        connection: Optional existing database connection. If provided, this function
                    will operate within the transaction context of the caller and will
                    NOT commit or rollback. The caller is responsible for transaction
                    management. If not provided, a new connection is created and
                    the operation (if not a SELECT/PRAGMA) is committed.

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

            try:
                # Validate connection state
                _ = conn.isolation_level
            except Exception as e:
                # Throw detailed exception instead of continuing with invalid connection
                raise DatabaseError(
                    message=f"Connection validation failed: {str(e)}",
                    query=query,
                    params=params,
                    original_error=e,
                ) from e
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
            # Only commit if this function created the connection
            if conn_created:
                conn.commit()
            result = None

        return result

    except sqlite3.Error as e:
        if conn and conn_created:
            conn.rollback()

        # Log with limited parameter data for security
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
        # Close the connection if it was created here
        if conn_created and conn:
            try:
                conn.close()
            except Exception:
                pass  # Ignore close errors


@retry_with_backoff(max_tries=3, base_delay=1.0, max_delay=10.0, component="database_batch")
def execute_many(
    query: str, params_list: list[tuple], connection: sqlite3.Connection = None
) -> None:
    """
    Execute a query with multiple parameter sets

    Args:
        query: SQL query with placeholders
        params_list: List of parameter tuples
        connection: Optional existing database connection. If provided, this function
                    will operate within the transaction context of the caller and will
                    NOT commit or rollback. The caller is responsible for transaction
                    management. If not provided, a new connection is created and
                    the batch operation is committed.

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

        # Only commit if we created this connection
        if conn_created:
            conn.commit()

    except sqlite3.Error as e:
        if (
            conn and conn_created
        ):  # Only rollback if this function created the connection
            conn.rollback()

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
        if conn and conn_created:
            conn.close()


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
        execute_query(sql_query, params_tuple, connection=conn_to_use)
        logger.info(f"Created new job: {job_id} of type {job_type}")

    try:
        if connection:
            # Operate within the caller's transaction using the provided connection
            _db_operation(connection)
        else:
            # Create and manage its own transaction
            with (
                db_transaction() as new_conn
            ):  # db_transaction handles commit/rollback/close
                _db_operation(new_conn)
        return job_id
    except (
        DatabaseError
    ):  # db_transaction would have logged and rolled back if it was used
        logger.error(
            f"Failed to create job {job_id} of type {job_type}. Error re-raised.",
            exc_info=True,
        )
        raise  # Re-raise the original DatabaseError which includes details


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

    # Create a database connection if one wasn't provided
    conn_created = False
    if not connection:
        connection = get_db_connection()
        conn_created = True
    else:
        # Проверка состояния соединения
        try:
            _ = connection.isolation_level
        except Exception:
            pass  # Connection state check failed

    try:
        # First check if the job exists
        check_query = "SELECT 1 FROM jobs WHERE job_id = ?"
        result = execute_query(check_query, (job_id,), connection=connection)
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
        execute_query(query, params, connection=connection)

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
        execute_query(history_query, history_params, connection=connection)

        # Explicitly commit if we created our own connection
        if conn_created:
            connection.commit()

        logger.info(
            f"Updated job {job_id}: status={status}, progress={progress}, message={status_message}"
        )
    finally:
        if conn_created and connection:
            connection.close()


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
        result = execute_query(query, (job_id,), connection=connection)
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
            query, tuple(params), fetchall=True, connection=connection
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

    try:
        execute_query(query, params, connection=connection)
        return result_id
    except DatabaseError:
        logger.error(f"Failed to create data upload result for job {job_id}")
        raise


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
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True

        cursor = conn.cursor()

        # Hash the JSON representation of the config for a stable ID
        config_json = json.dumps(
            config_dict, sort_keys=True, default=json_default_serializer
        )
        config_id = hashlib.md5(config_json.encode()).hexdigest()

        # Check if this config already exists
        cursor.execute(
            "SELECT config_id FROM configs WHERE config_id = ?", (config_id,)
        )

        if not cursor.fetchone():
            # Create the config
            now = datetime.now().isoformat()

            cursor.execute(
                """
                INSERT INTO configs (config_id, config, is_active, created_at, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (config_id, config_json, 1 if is_active else 0, now, source),
            )

            if is_active or source is not None:
                # If this config should be active, deactivate all others
                cursor.execute(
                    """
                    UPDATE configs
                    SET is_active = 0
                    WHERE config_id != ?
                    """,
                    (config_id,),
                )

            if conn_created:  # Only commit if this function created the connection
                conn.commit()
            logger.info(
                f"Created config: {config_id}, active: {is_active}, source: {source}"
            )
        elif is_active or source is not None:
            # Handle active flag
            if is_active:
                cursor.execute("UPDATE configs SET is_active = 0 WHERE config_id != ?", (config_id,))
                cursor.execute("UPDATE configs SET is_active = 1 WHERE config_id = ?", (config_id,))

            # Optionally update source if provided
            if source is not None:
                cursor.execute("UPDATE configs SET source = ? WHERE config_id = ?", (source, config_id))

            if conn_created:  # Only commit if this function created the connection
                conn.commit()
            logger.info(f"Set existing config {config_id} as active")

        return config_id

    except Exception as e:
        logger.error(f"Error with config: {e}")
        if conn_created and "conn" in locals():
            try:
                conn.rollback()
            except Exception:
                pass  # Already closed or other issue
        raise
    finally:
        if conn_created and "conn" in locals():
            try:
                conn.close()
            except Exception:
                pass  # Already closed or other issue


def get_active_config(connection: sqlite3.Connection = None) -> dict[str, Any] | None:
    """
    Returns the currently active config or None if none is active.
    Returns:
        Dictionary with config_id and config fields if an active config exists,
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
            SELECT config_id, config
            FROM configs
            WHERE is_active = 1
            LIMIT 1
            """
        )
        result = cursor.fetchone()
        if not result:
            return None
        # ... existing code ...
        if hasattr(result, "keys"):  # sqlite.Row
            config_id = result["config_id"]
            config_json = result["config"]
        else:  # tuple
            config_id = result[0]
            config_json = result[1]
        try:
            config = json.loads(config_json) if config_json else {}
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Error parsing config JSON: {config_json}")
            config = {}
        return {"config_id": config_id, "config": config}
    except Exception as e:
        logger.error(f"Error getting active config: {e}")
        return None
    finally:
        if conn_created and "conn" in locals():
            try:
                conn.close()
            except Exception:
                pass  # Already closed or other issue


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
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True

        cursor = conn.cursor()

        # First check if config exists
        cursor.execute("SELECT 1 FROM configs WHERE config_id = ?", (config_id,))

        if not cursor.fetchone():
            logger.error(f"Config {config_id} not found")
            return False

        if deactivate_others:
            cursor.execute("UPDATE configs SET is_active = 0")

        cursor.execute(
            "UPDATE configs SET is_active = 1 WHERE config_id = ?", (config_id,)
        )

        if conn_created:
            conn.commit()
        logger.info(f"Set config {config_id} as active")
        return True

    except sqlite3.Error as e:
        if conn_created and conn:
            conn.rollback()
        logger.error(f"Error setting config {config_id} as active: {e}")
        return False
    finally:
        if conn_created and conn:
            conn.close()


def get_best_config_by_metric(
    metric_name: str,
    higher_is_better: bool = True,
    connection: sqlite3.Connection = None,
) -> dict[str, Any] | None:
    """
    Returns the config with the best metric value based on training_results.

    Args:
        metric_name: The name of the metric to use for evaluation
        higher_is_better: True if higher values of the metric are better, False otherwise
        connection: Optional existing database connection to use

    Returns:
        Dictionary with config_id, config, and metrics fields if a best config exists,
        otherwise None.
    """
    if metric_name not in ALLOWED_METRICS:
        logger.error(
            f"Invalid metric_name '{metric_name}' provided to get_best_config_by_metric."
        )
        raise ValueError(
            f"Invalid metric_name: {metric_name}. Allowed metrics are: {ALLOWED_METRICS}"
        )

    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True

        conn.row_factory = dict_factory  # Use dict_factory
        cursor = conn.cursor()

        order_direction = "DESC" if higher_is_better else "ASC"

        # Construct the JSON path safely
        json_path = f"'$.{metric_name}'"  # The metric_name is now validated

        # Join configs and training_results to find the best metrics
        # Ensure config_id is not NULL in training_results
        # The metric_name for JSON_EXTRACT and order_direction are now safely constructed.
        query = f"""
            SELECT
                c.config_id,
                c.config,
                tr.metrics,
                JSON_EXTRACT(tr.metrics, {json_path}) as metric_value
            FROM training_results tr
            JOIN configs c ON tr.config_id = c.config_id
            WHERE tr.config_id IS NOT NULL AND JSON_VALID(tr.metrics) = 1 AND JSON_EXTRACT(tr.metrics, {json_path}) IS NOT NULL
            ORDER BY metric_value {order_direction}
            LIMIT 1
        """

        # No parameters needed for this query as dynamic parts are validated and inlined
        cursor.execute(query)
        result = cursor.fetchone()

        if result:
            # Parse JSON fields
            if result.get("config"):
                try:
                    result["config"] = json.loads(result["config"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode config JSON for config {result['config_id']}"
                    )
                    result["config"] = {}  # Set to empty dict on error
            else:
                result["config"] = {}

            if result.get("metrics"):
                try:
                    result["metrics"] = json.loads(result["metrics"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode metrics JSON for config {result['config_id']} in training_results"
                    )
                    result["metrics"] = {}  # Set to empty dict on error
            else:
                result["metrics"] = {}

            # Remove the extracted metric_value helper column
            result.pop("metric_value", None)
            return result

        return None

    except sqlite3.Error as e:
        logger.error(
            f"Database error in get_best_config_by_metric for metric '{metric_name}': {e}",
            exc_info=True,
        )
        # It might be better to re-raise a custom error or let DatabaseError propagate if execute_query was used
        return None  # Or raise DatabaseError(f"Failed to get best config: {e}", original_error=e)
    except ValueError as ve:  # Catch the ValueError from metric_name validation
        logger.error(f"ValueError in get_best_config_by_metric: {ve}", exc_info=True)
        raise  # Re-raise the ValueError to be handled by the caller
    finally:
        if conn_created and conn:
            conn.close()  # Correct indentation


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
            INSERT INTO models (
                model_id, job_id, model_path, created_at, metadata, is_active
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
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
            # If this model should be active, deactivate all others
            cursor.execute(
                """
                UPDATE models
                SET is_active = 0
                WHERE model_id != ?
                """,
                (model_id,),
            )

        conn.commit()
        logger.info(f"Created model record: {model_id}, active: {is_active}")

    except Exception as e:
        logger.error(f"Error creating model record: {e}")
        if conn_created and "conn" in locals():
            try:
                conn.rollback()
            except Exception:
                pass  # Already closed or other issue
        raise
    finally:
        if conn_created and "conn" in locals():
            try:
                conn.close()
            except Exception:
                pass  # Already closed or other issue


def get_active_model(connection: sqlite3.Connection = None) -> dict[str, Any] | None:
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
        if not result:
            return None

        # Check how to access the result (could be dict or sqlite.Row)
        if hasattr(result, "keys"):  # sqlite.Row or dict
            model_id = result["model_id"]
            model_path = result["model_path"]
            metadata_str = result["metadata"] if "metadata" in result else None
        else:  # tuple
            model_id = result[0]
            model_path = result[1]
            metadata_str = result[2] if len(result) > 2 else None

        # Parse metadata JSON if it exists
        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except json.JSONDecodeError:
            logger.warning(f"Could not decode metadata JSON for model {model_id}")
            metadata = {}  # Set to empty dict on error

        return {"model_id": model_id, "model_path": model_path, "metadata": metadata}

    except sqlite3.Error as e:
        logger.error(f"Database error in get_active_model: {e}")
        return None
    finally:
        if conn_created and conn:
            conn.close()


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
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True

        cursor = conn.cursor()

        # First check if model exists
        cursor.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,))

        if not cursor.fetchone():
            logger.error(f"Model {model_id} not found")
            return False

        if deactivate_others:
            cursor.execute("UPDATE models SET is_active = 0")

        cursor.execute(
            "UPDATE models SET is_active = 1 WHERE model_id = ?", (model_id,)
        )

        if conn_created:
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

    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True

        conn.row_factory = dict_factory  # Use dict_factory for easier access
        cursor = conn.cursor()

        order_direction = "DESC" if higher_is_better else "ASC"

        # Construct the JSON path safely
        json_path = f"'$.{metric_name}'"  # The metric_name is now validated

        # Join models and training_results to find the best metrics
        # Ensure model_id is not NULL in training_results
        # The metric_name for JSON_EXTRACT and order_direction are now safely constructed.
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

        # No parameters needed for this query as dynamic parts are validated and inlined
        cursor.execute(query)
        result = cursor.fetchone()

        if result:
            # No need to parse JSON here, return the dict directly
            # Ensure metadata is parsed if it exists
            if result.get("metadata"):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode metadata JSON for model {result['model_id']}"
                    )
                    result["metadata"] = {}  # Set to empty dict on error
            else:
                result["metadata"] = {}

            # Ensure metrics is parsed if it exists
            if result.get("metrics"):
                try:
                    result["metrics"] = json.loads(result["metrics"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode metrics JSON for model {result['model_id']} in training_results"
                    )
                    result["metrics"] = {}  # Set to empty dict on error
            else:
                result["metrics"] = {}

            # Remove the extracted metric_value helper column
            result.pop("metric_value", None)
            return result

        return None

    except sqlite3.Error as e:
        logger.error(
            f"Database error in get_best_model_by_metric for metric '{metric_name}': {e}",
            exc_info=True,
        )
        # It might be better to re-raise a custom error or let DatabaseError propagate if execute_query was used
        return None  # Or raise DatabaseError(f"Failed to get best model: {e}", original_error=e)
    except ValueError as ve:  # Catch the ValueError from metric_name validation
        logger.error(f"ValueError in get_best_model_by_metric: {ve}", exc_info=True)
        raise  # Re-raise the ValueError to be handled by the caller
    finally:
        if conn_created and conn:
            conn.close()  # Correct indentation


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
    return execute_query(query, (limit,), fetchall=True, connection=connection)


def delete_model_record_and_file(
    model_id: str, connection: sqlite3.Connection = None
) -> bool:
    """
    Deletes a model record from the database and its associated file from the filesystem.
    Returns True if successful, False otherwise.
    """
    conn_created = False
    actual_conn = None
    try:
        if connection:
            actual_conn = connection
        else:
            actual_conn = get_db_connection()
            conn_created = True

        with actual_conn as conn:  # Use connection as a context manager for transaction
            cursor = conn.cursor()

            # Get the model path before deleting the record
            cursor.execute(
                "SELECT model_path FROM models WHERE model_id = ?", (model_id,)
            )
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Model with ID {model_id} not found for deletion.")
                return False

            model_path = result["model_path"]

            # 1. Delete dependent records from training_results
            cursor.execute(
                "DELETE FROM training_results WHERE model_id = ?", (model_id,)
            )

            # 2. Delete the model record
            cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))

            # 3. Delete the model file
            if model_path and os.path.exists(model_path):
                try:
                    os.remove(model_path)
                except OSError as e:
                    logger.error(
                        f"Error removing model file {model_path}: {e}", exc_info=True
                    )
                    # Depending on policy, we might still consider the DB part a success
                    # For now, we'll let it be a success if the DB records are gone.

        return True

    except sqlite3.Error as e:
        logger.error(
            f"Database error in delete_model_record_and_file for model_id {model_id}: {e}",
            exc_info=True,
        )
        return False
    finally:
        if conn_created and actual_conn:
            actual_conn.close()


def create_training_result(
    job_id: str,
    model_id: str,
    config_id: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    duration: int | None,
    connection: sqlite3.Connection = None,  # Allow passing connection for transaction
) -> str:
    """Creates a record for a completed training job and handles auto-activation."""
    result_id = str(uuid.uuid4())
    metrics_json = json.dumps(metrics, default=json_default_serializer)
    # config_json = json.dumps(config) # This line is removed as config_json is no longer used for direct DB insertion
    settings = get_settings()  # Get latest settings

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
        (result_id, job_id, model_id, config_id, metrics, duration)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (result_id, job_id, model_id, config_id, metrics_json, duration)
        cursor.execute(query, params)
        logger.info(f"Created training result: {result_id} for job: {job_id}")

        # --- Auto-activation logic ---
        # from deployment.app.config import settings  # Import settings here

        # Auto-activate best config
        if settings.auto_select_best_configs:
            best_config = get_best_config_by_metric(
                metric_name=settings.default_metric,
                higher_is_better=settings.default_metric_higher_is_better,
                connection=conn,  # Use the same connection
            )
            if best_config and best_config.get("config_id"):
                logger.info(f"Auto-activating best config: {best_config['config_id']}")
                set_config_active(best_config["config_id"], connection=conn)
            else:
                logger.warning(
                    f"Auto-select configs enabled, but couldn't find best config by metric '{settings.default_metric}'"
                )

        # Auto-activate best model
        if (
            settings.auto_select_best_model and model_id
        ):  # Only if a model was actually created
            best_model = get_best_model_by_metric(
                metric_name=settings.default_metric,
                higher_is_better=settings.default_metric_higher_is_better,
                connection=conn,  # Use the same connection
            )
            if best_model and best_model.get("model_id"):
                logger.info(f"Auto-activating best model: {best_model['model_id']}")
                set_model_active(best_model["model_id"], connection=conn)
            else:
                logger.warning(
                    f"Auto-select model enabled, but couldn't find best model by metric '{settings.default_metric}'"
                )

        if conn_created:  # Commit only if we created the connection here
            conn.commit()

    except sqlite3.Error as e:
        logger.error(
            f"Database error creating training result or auto-activating for job {job_id}: {e}",
            exc_info=True,
        )
        if conn_created and conn:
            conn.rollback()  # Rollback on error if we created the connection
        raise DatabaseError(
            f"Failed to create training result: {e}", original_error=e
        ) from e
    finally:
        # Close only if we created the connection here
        if conn_created and conn:
            conn.close()

    return result_id


def create_prediction_result(
    job_id: str,
    model_id: str,
    output_path: str,
    summary_metrics: dict[str, Any] | None,
    prediction_month: date | None = None,  # New parameter
    connection: sqlite3.Connection = None,
) -> str:
    """
    Create a prediction result record

    Args:
        job_id: Associated job ID
        model_id: ID of the model used for prediction
        output_path: Path to prediction output file
        summary_metrics: Dictionary of prediction metrics
        prediction_month: Month for which predictions were made
        connection: Optional existing database connection to use

    Returns:
        Generated result ID
    """
    result_id = generate_id()

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
        prediction_month.isoformat() if prediction_month else None,
    )

    try:
        execute_query(query, params, connection=connection)
        return result_id
    except DatabaseError:
        logger.error(f"Failed to create prediction result for job {job_id}")
        raise


def create_report_result(
    job_id: str,
    report_type: str,
    parameters: dict[str, Any],
    output_path: str,
    connection: sqlite3.Connection = None,
) -> str:
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
        json.dumps(parameters, default=json_default_serializer),
        output_path,
    )

    try:
        execute_query(query, params, connection=connection)
        return result_id
    except DatabaseError:
        logger.error(f"Failed to create report result for job {job_id}")
        raise


def get_data_upload_result(
    result_id: str, connection: sqlite3.Connection = None
) -> dict:
    """Get data upload result by ID"""
    query = "SELECT * FROM data_upload_results WHERE result_id = ?"
    return execute_query(query, (result_id,), connection=connection)


def get_training_result(result_id: str, connection: sqlite3.Connection = None) -> dict:
    """Get training result by ID"""
    query = "SELECT * FROM training_results WHERE result_id = ?"
    return execute_query(query, (result_id,), connection=connection)


def get_prediction_result(
    result_id: str, connection: sqlite3.Connection = None
) -> dict:
    """Get prediction result by ID"""
    query = "SELECT * FROM prediction_results WHERE result_id = ?"
    return execute_query(query, (result_id,), connection=connection)


def get_report_result(result_id: str, connection: sqlite3.Connection = None) -> dict:
    """Get report result by ID"""
    query = "SELECT * FROM report_results WHERE result_id = ?"
    return execute_query(query, (result_id,), connection=connection)


# Processing runs management


def create_processing_run(
    start_time: datetime,
    status: str,
    cutoff_date: str,
    source_files: str,
    end_time: datetime = None,
    connection: sqlite3.Connection = None,
) -> int:
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
        end_time.isoformat() if end_time else None,
    )

    try:
        execute_query(query, params, connection=connection)

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
        execute_query(query, tuple(params), connection=connection)
    except DatabaseError:
        logger.error(f"Failed to update processing run {run_id}")
        raise


# MultiIndex mapping functions


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
            SELECT config_id, config, created_at, is_active
            FROM configs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

        results = cursor.fetchall()
        for result in results:
            if "config" in result and result["config"]:
                result["config"] = json.loads(result["config"])

        return results

    except sqlite3.Error as e:
        logger.error(f"Database error in get_configs: {e}")
        return []
    finally:
        if conn_created and conn:
            conn.close()


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

    summary = {"deleted_count": 0, "skipped_count": 0, "skipped_configs": []}
    conn_created = False
    try:
        if not connection:
            conn = get_db_connection()
            conn_created = True
        else:
            conn = connection

        with conn:  # Use connection as a context manager for transaction
            cursor = conn.cursor()

            placeholders = ",".join("?" for _ in config_ids)

            # Find which configs are active
            cursor.execute(
                f"SELECT config_id FROM configs WHERE config_id IN ({placeholders}) AND is_active = 1",
                config_ids,
            )
            active_configs = {row["config_id"] for row in cursor.fetchall()}

            summary["skipped_configs"] = list(active_configs)
            summary["skipped_count"] = len(active_configs)

            configs_to_delete = [cid for cid in config_ids if cid not in active_configs]

            if not configs_to_delete:
                return summary

            delete_placeholders = ",".join("?" for _ in configs_to_delete)

            # Find jobs using these configs
            cursor.execute(
                f"SELECT config_id FROM jobs WHERE config_id IN ({delete_placeholders})"
            )
            used_configs_in_jobs = {row["config_id"] for row in cursor.fetchall()}

            # Also check training_results
            cursor.execute(
                f"SELECT config_id FROM training_results WHERE config_id IN ({delete_placeholders})"
            )
            used_configs_in_results = {row["config_id"] for row in cursor.fetchall()}

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

            final_placeholders = ",".join("?" for _ in final_configs_to_delete)

            # Now delete the unused, non-active configs
            cursor.execute(
                f"DELETE FROM configs WHERE config_id IN ({final_placeholders})",
                final_configs_to_delete,
            )

            summary["deleted_count"] = cursor.rowcount

    except Exception as e:
        logger.error(f"Error deleting configs by IDs: {e}", exc_info=True)
    finally:
        if conn_created and "conn" in locals() and conn:
            conn.close()

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
            (limit,),
        )

        results = cursor.fetchall()
        for result in results:
            if "metadata" in result and result["metadata"]:
                result["metadata"] = json.loads(result["metadata"])

        return results

    except sqlite3.Error as e:
        logger.error(f"Database error in get_all_models: {e}")
        return []
    finally:
        if conn_created and conn:
            conn.close()


def delete_models_by_ids(
    model_ids: list[str], connection: sqlite3.Connection = None
) -> dict[str, Any]:
    """
    Deletes multiple model records by their IDs and their associated files, skipping active models.

    Args:
        model_ids: List of model IDs to delete.
        connection: Optional existing database connection.

    Returns:
        A dictionary with deletion summary.
    """
    if not model_ids:
        return {
            "deleted_count": 0,
            "skipped_count": 0,
            "skipped_models": [],
            "failed_deletions": [],
        }

    summary = {
        "deleted_count": 0,
        "skipped_count": 0,
        "skipped_models": [],
        "failed_deletions": [],
    }
    conn_created = False
    try:
        if not connection:
            conn = get_db_connection()
            conn_created = True
        else:
            conn = connection

        with conn:  # Use connection as a context manager for transaction
            cursor = conn.cursor()

            placeholders = ",".join("?" for _ in model_ids)

            # Find which models are active
            cursor.execute(
                f"SELECT model_id FROM models WHERE model_id IN ({placeholders}) AND is_active = 1",
                model_ids,
            )
            active_models = {row["model_id"] for row in cursor.fetchall()}

            summary["skipped_models"] = list(active_models)
            summary["skipped_count"] = len(active_models)

            models_to_delete_ids = [
                mid for mid in model_ids if mid not in active_models
            ]

            if not models_to_delete_ids:
                return summary

            delete_placeholders = ",".join("?" for _ in models_to_delete_ids)

            # Get model paths before deleting records from DB
            cursor.execute(
                f"SELECT model_id, model_path FROM models WHERE model_id IN ({delete_placeholders})",
                models_to_delete_ids,
            )
            models_to_delete_with_paths = cursor.fetchall()

            # Delete associated training results first
            cursor.execute(
                f"DELETE FROM training_results WHERE model_id IN ({delete_placeholders})",
                models_to_delete_ids,
            )

            # Now delete the models from DB
            cursor.execute(
                f"DELETE FROM models WHERE model_id IN ({delete_placeholders})",
                models_to_delete_ids,
            )

            summary["deleted_count"] = cursor.rowcount

            # Finally, delete the physical files
            for model_info in models_to_delete_with_paths:
                model_id = model_info["model_id"]
                model_path = model_info["model_path"]
                if model_path and os.path.exists(model_path):
                    try:
                        os.remove(model_path)
                        logger.info(f"Deleted model file: {model_path}")
                    except OSError as e:
                        logger.error(
                            f"Error removing model file {model_path} for model_id {model_id}: {e}",
                            exc_info=True,
                        )
                        summary["failed_deletions"].append(model_id)

    except Exception as e:
        logger.error(f"Error deleting models by IDs: {e}", exc_info=True)
        # Do not re-raise to avoid hiding the summary
    finally:
        if conn_created and "conn" in locals() and conn:
            conn.close()

    return summary


def get_effective_config(settings, logger=None, connection=None):
    """
    Returns the active config if it exists, otherwise the best by metric.
    Raises ValueError if neither exists.
    Args:
        settings: AppSettings instance (или объект с default_metric и default_metric_higher_is_better)
        logger: Optional logger for info/error messages
        connection: Optional sqlite3.Connection to use (для тестов с in-memory БД)
    Returns:
        dict: config data (как возвращает get_active_config или get_best_config_by_metric)
    Raises:
        ValueError: если ни один сет не найден
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


def create_tuning_result(
    job_id: str,
    config_id: str,
    metrics: dict[str, Any] | None,
    duration: int | None,
    connection: sqlite3.Connection = None,
) -> str:
    """Insert a single tuning attempt produced by Ray tuning.

    Args:
        job_id: ID of the tuning job (from jobs table)
        config_id: ID of the configuration (hash, must already exist in configs)
        metrics: Dict with metric values returned by Ray
        duration: Wall-clock seconds spent on the trial
        connection: Optional existing sqlite3 connection

    Returns:
        Newly created result_id (UUID4 string)
    """
    result_id = generate_id()
    metrics_json = json.dumps(metrics or {}, default=json_default_serializer)
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
            INSERT INTO tuning_results (result_id, job_id, config_id, metrics, duration, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                job_id,
                config_id,
                metrics_json,
                duration,
                datetime.now().isoformat(),
            ),
        )

        if conn_created:
            conn.commit()
        return result_id
    except sqlite3.Error as e:
        logger.error(f"Database error creating tuning_result for job {job_id}: {e}")
        if conn_created and conn:
            conn.rollback()
        raise DatabaseError(
            f"Failed to create tuning result: {e}", original_error=e
        ) from e
    finally:
        if conn_created and conn:
            conn.close()


def get_top_tuning_results(
    metric_name: str,
    higher_is_better: bool = True,
    limit: int = 10,
    threshold: float | None = None,
    connection: sqlite3.Connection = None,
) -> list[dict[str, Any]]:
    """Return top tuning_results ordered by specific metric.

    Rows where metric is NULL are ignored.  Optionally filter by threshold.
    """
    if metric_name not in ALLOWED_METRICS:
        raise ValueError(
            f"Invalid metric_name: {metric_name}. Allowed: {ALLOWED_METRICS}"
        )

    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        order = "DESC" if higher_is_better else "ASC"
        json_path = f"'$.{metric_name}'"

        sql = f"""
            SELECT result_id, job_id, config_id, metrics,
                   JSON_EXTRACT(metrics, {json_path}) AS metric_value,
                   duration, created_at
            FROM tuning_results
            WHERE JSON_VALID(metrics)=1
              AND JSON_EXTRACT(metrics, {json_path}) IS NOT NULL
        """
        params: tuple = ()
        if threshold is not None:
            comp = ">=" if higher_is_better else "<="
            sql += f" AND JSON_EXTRACT(metrics, {json_path}) {comp} ?"
            params = (threshold,)
        sql += f" ORDER BY metric_value {order} LIMIT ?"
        params += (limit,)

        cursor.execute(sql, params)
        rows = cursor.fetchall() or []
        # parse metrics JSON for convenience
        for r in rows:
            try:
                r["metrics"] = json.loads(r["metrics"] or "{}")
            except json.JSONDecodeError:
                r["metrics"] = {}
            r.pop("metric_value", None)
        return rows
    finally:
        if conn_created and conn:
            conn.close()


def get_top_configs(
    limit: int = 5,
    metric_name: str | None = None,
    higher_is_better: bool = True,
    include_active: bool = True,
    connection: sqlite3.Connection = None,
) -> list[dict[str, Any]]:
    """Return best historical configs for seeding tuning.

    Selection rules:
    1. Optionally include the currently active config first.
    2. Then order configs by metric from *training_results* table.
       Rows where metric is NULL are ignored.
    3. Falls back to most recently created configs if no metrics.
    """
    metric_name = metric_name or "val_MIC"  # default if not provided
    if metric_name not in ALLOWED_METRICS:
        # allow dynamic metrics but warn
        logger.warning(
            "get_top_configs: metric %s not in ALLOWED_METRICS – proceeding anyway", metric_name
        )
    order = "DESC" if higher_is_better else "ASC"
    conn_created = False
    try:
        if connection:
            conn = connection
        else:
            conn = get_db_connection()
            conn_created = True
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        # Build base SQL to fetch configs with metric value
        json_path = f"'$.{metric_name}'"
        sql = f"""
            SELECT c.config_id, c.config, c.created_at, c.is_active,
                   JSON_EXTRACT(tr.metrics, {json_path}) AS metric_value
            FROM configs c
            LEFT JOIN training_results tr ON c.config_id = tr.config_id
            WHERE JSON_VALID(tr.metrics)=1 OR tr.metrics IS NULL
        """
        # Filter: if include_active false, remove active
        if not include_active:
            sql += " AND c.is_active = 0"
        sql += f" GROUP BY c.config_id ORDER BY c.is_active DESC, metric_value {order}, c.created_at DESC LIMIT ?"
        cursor.execute(sql, (limit,))
        rows = cursor.fetchall() or []
        # ensure config json parsed
        top_cfgs = []
        for r in rows:
            try:
                cfg = json.loads(r["config"])
            except Exception:
                continue
            top_cfgs.append(cfg)
        return top_cfgs
    finally:
        if conn_created and conn:
            conn.close()


# ---------------------------------------------------------------------------
# Retry events persistence helpers (used by RetryMonitor)
# ---------------------------------------------------------------------------


def insert_retry_event(event: dict[str, Any], connection: sqlite3.Connection = None) -> None:
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

    # Ensure boolean stored as int (SQLite lacks boolean)
    event = event.copy()
    event["successful"] = 1 if event.get("successful") else 0

    execute_query(query, event, connection=connection)
    # Commit if using external connection to ensure persistence
    if connection:
        connection.commit()


def fetch_recent_retry_events(limit: int = 1000, connection: sqlite3.Connection = None) -> list[dict[str, Any]]:
    """Fetch most recent retry events ordered oldest->newest up to *limit*."""

    query = "SELECT * FROM retry_events ORDER BY id DESC LIMIT ?"
    rows = execute_query(query, (limit,), fetchall=True, connection=connection) or []
    rows.reverse()
    return rows
