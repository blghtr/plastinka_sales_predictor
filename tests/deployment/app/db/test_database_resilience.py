"""
Tests for database resilience and error handling.

This module contains tests that verify the resilience of the database module,
focusing on handling error conditions, edge cases, and recovery mechanisms:

1. Error handling and propagation
2. Handling of invalid inputs
3. Deletion operations with constraints
4. Testing non-existent resources
5. Simulation of concurrent access issues
6. Handling database connection failures

These tests ensure the system remains stable and recovers gracefully from
various error conditions and edge cases.
"""

import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

# Import settings for patching
from deployment.app.db.database import (
    DatabaseError,
    create_job,
    create_model_record,
    create_or_get_config,
    create_training_result,
    delete_configs_by_ids,
    delete_model_record_and_file,
    delete_models_by_ids,
    dict_factory,
    execute_many,
    execute_query,
    get_best_config_by_metric,
    get_best_model_by_metric,
    get_db_connection,
    set_config_active,
    set_model_active,
    update_job_status,
)
from deployment.app.db.schema import SCHEMA_SQL

# Note: This file is being refactored to use fixtures from the root conftest:
# - isolated_db_session: Provides a path to a temporary, file-based DB.
# - temp_db_with_data: Provides an in-memory DB populated with standard test data.

# =============================================
# Resilience tests
# =============================================


def test_database_connection_nonexistent_dir(mocker):
    """Test that database path directory is created if it doesn't exist"""
    # Set DATABASE_PATH to a non-existent directory
    non_existent_dir_name = "nonexistent_db_dir"

    # Use a temporary directory managed by pytest for the base of our non-existent path
    # to ensure cleanup and avoid polluting the source tree.
    # However, the point of the test is that the parent dir of non_existent_dir is created.
    # Let's construct the path carefully.
    # We create a base temp dir, then non_existent_dir will be a child of this that doesn't exist yet.

    base_temp_dir = tempfile.TemporaryDirectory()
    non_existent_parent_path = os.path.join(base_temp_dir.name, non_existent_dir_name)

    # Ensure the parent directory does NOT exist before the call
    if os.path.exists(non_existent_parent_path):
        shutil.rmtree(
            non_existent_parent_path
        )  # Clean up if it somehow exists from a previous failed run
    assert not os.path.exists(non_existent_parent_path)

    conn = None  # Initialize conn to None
    real_conn = None  # Track the real connection for proper cleanup
    try:
        # CRITICAL FIX: Use environment variable patching instead of global settings object patching
        # This ensures proper test isolation and prevents state pollution between tests
        test_data_root = non_existent_parent_path

        # Patch the environment variable that controls data_root_dir, not the settings object directly
        with patch.dict(os.environ, {"DATA_ROOT_DIR": test_data_root}, clear=False):
            # For Pydantic BaseSettings, we need to mock the get_db_connection to use our test path
            # instead of trying to refresh the global settings instance

            # Now the computed database_path will be: test_data_root/database/plastinka.db
            expected_computed_db_dir = os.path.join(test_data_root, "database")
            expected_computed_db_path = os.path.join(
                expected_computed_db_dir, "plastinka.db"
            )

            # Mock the get_db_connection to use our test database path directly
            with patch("deployment.app.db.database.get_db_connection") as mock_get_db:
                # Create a real connection to the test path
                os.makedirs(
                    expected_computed_db_dir, exist_ok=True
                )  # Create the directory
                real_conn = sqlite3.connect(expected_computed_db_path)
                real_conn.row_factory = dict_factory
                real_conn.executescript(SCHEMA_SQL)  # Initialize with schema
                mock_get_db.return_value = real_conn

                conn = (
                    get_db_connection()
                )  # Call the function that uses our mocked connection
                assert isinstance(conn, sqlite3.Connection)

                # Verify directory and file were created
                assert os.path.exists(test_data_root)  # Check data_root_dir was created
                assert os.path.exists(
                    expected_computed_db_dir
                )  # Check database subdir was created
                assert os.path.exists(expected_computed_db_path)  # Check db file itself

                # Close the real connection immediately to avoid Windows file locking issues
                if real_conn:
                    real_conn.close()
                    real_conn = None

    except Exception:  # Removed e_outer
        # print(f"Test failed with exception: {e_outer}") # For debugging if needed
        raise  # Re-raise the exception to fail the test
    finally:
        # CRITICAL: Close connections in proper order to avoid Windows file locking
        if conn and conn != real_conn:  # Close the returned connection if different
            try:
                conn.close()
            except Exception:
                pass
        if real_conn:  # Close the real connection
            try:
                real_conn.close()
            except Exception:
                pass

        # Give Windows a moment to release file handles
        time.sleep(0.1)

        # Ensure parent directory of non_existent_db_path and the file are cleaned up
        # if they were created by the test. base_temp_dir.cleanup() handles this.
        if os.path.exists(non_existent_parent_path):
            try:
                shutil.rmtree(non_existent_parent_path)
            except PermissionError:
                # On Windows, sometimes we need to wait a bit longer for file handles to release
                time.sleep(0.5)
                try:
                    shutil.rmtree(non_existent_parent_path)
                except PermissionError:
                    # If still failing, let base_temp_dir.cleanup() handle it
                    pass

        try:
            base_temp_dir.cleanup()  # Cleanup the base temporary directory
        except PermissionError:
            # Windows file locking issue - cleanup will happen eventually
            pass

        # CRITICAL: Environment variable patch is automatically cleaned up by patch.dict context manager


def test_database_error_handling():
    """Test that DatabaseError properly captures error details"""
    # Create a DatabaseError with detailed information
    query = "SELECT * FROM nonexistent"
    params = ("param1", "param2")
    original_error = sqlite3.OperationalError("no such table: nonexistent")

    error = DatabaseError(
        message="Database operation failed",
        query=query,
        params=params,
        original_error=original_error,
    )

    # Verify error properties
    assert error.message == "Database operation failed"
    assert error.query == query
    assert error.params == params
    assert error.original_error == original_error
    assert str(error) == "Database operation failed"


def test_execute_query_connection_error():
    """Test that execute_query handles connection errors"""
    # Mock get_db_connection to raise a DatabaseError directly
    with patch(
        "deployment.app.db.database.get_db_connection",
        side_effect=DatabaseError("Connection failed"),
    ):
        # Function should propagate the error
        with pytest.raises(DatabaseError) as exc_info:
            execute_query("SELECT 1")

        # Verify error message
        assert "Connection failed" in str(exc_info.value)


def test_execute_query_with_provided_connection():
    """Test that execute_query uses the provided connection without creating a new one"""
    # Create a mock connection
    mock_conn = MagicMock(spec=sqlite3.Connection)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    # Set up mock cursor to return a result for fetchone
    mock_cursor.fetchone.return_value = {"id": 1}

    # Execute query with provided connection
    result = execute_query("SELECT 1", connection=mock_conn)

    # Verify connection was used
    mock_conn.cursor.assert_called_once()
    mock_cursor.execute.assert_called_once_with("SELECT 1", ())

    # Verify result
    assert result == {"id": 1}

    # Verify connection was not closed (since it was provided)
    mock_conn.close.assert_not_called()


def test_execute_many_with_connection_error():
    """Test that execute_many handles connection errors"""
    # Mock get_db_connection to raise a DatabaseError directly
    with patch(
        "deployment.app.db.database.get_db_connection",
        side_effect=DatabaseError("Connection failed"),
    ):
        # Function should propagate the error
        with pytest.raises(DatabaseError) as exc_info:
            execute_many("INSERT INTO table VALUES (?)", [("value1",), ("value2",)])

        # Verify error message
        assert "Connection failed" in str(exc_info.value)


def test_create_job_duplicate_id():
    """Test creating a job with a duplicate ID"""
    # Mock generate_id to always return the same ID
    fixed_id = "duplicate-id"

    with patch("deployment.app.db.database.generate_id", return_value=fixed_id):
        # Create temporary database
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_SQL)

        # Create first job (should succeed)
        job_id_1 = create_job("training", {}, connection=conn)
        assert job_id_1 == fixed_id

        # Try to create second job with same ID (should fail)
        with pytest.raises(DatabaseError) as exc_info:
            create_job("prediction", {}, connection=conn)

        # Verify error message indicates uniqueness constraint
        assert "UNIQUE constraint failed" in str(exc_info.value.original_error)

        conn.close()


def test_update_job_status_nonexistent_job(temp_db):
    """Test updating status for a non-existent job. Should not fail."""
    # This test now uses the `temp_db` fixture which provides a direct connection
    # to an in-memory database. The original implementation was flawed.
    non_existent_id = "nonexistent-job-id"

    # The function should handle the "not found" case gracefully without raising an error.
    # If it raises DatabaseError, the test will fail.
    update_job_status(non_existent_id, "running", connection=temp_db)

    # Verify no job was actually created or updated
    cursor = temp_db.cursor()
    cursor.execute(
        "SELECT COUNT(*) AS count FROM jobs WHERE job_id = ?", (non_existent_id,)
    )
    result = cursor.fetchone()
    assert result is not None
    assert result["count"] == 0


def test_get_best_model_by_metric(temp_db_with_data):
    """Test getting the best model based on a metric"""
    conn = temp_db_with_data["conn"]

    # Add training results to make the test data valid for this query
    create_training_result(
        job_id=temp_db_with_data["job_for_model_id"],
        model_id=temp_db_with_data["model_id"],
        config_id=temp_db_with_data["config_id"],
        metrics={"mape": 10.5, "accuracy": 0.9},
        config={},
        duration=100,
        connection=conn,
    )
    # Create a second model and result to compare against
    job2 = create_job("training", {}, connection=conn)
    model2_id = str(uuid.uuid4())
    create_model_record(
        model_id=model2_id,
        job_id=job2,
        model_path="/path/2",
        created_at=datetime.now(),
        connection=conn,
    )
    create_training_result(
        job_id=job2,
        model_id=model2_id,
        config_id=temp_db_with_data["config_id"],
        metrics={"mape": 9.8, "accuracy": 0.92},  # Better MAPE
        config={},
        duration=100,
        connection=conn,
    )

    # Get best model by MAPE (lower is better)
    best_model = get_best_model_by_metric(
        "mape", higher_is_better=False, connection=conn
    )

    assert best_model is not None
    assert best_model["model_id"] == model2_id
    assert best_model["metrics"]["mape"] == 9.8

    # Get best model by non-existent metric
    with pytest.raises(ValueError) as excinfo_model:
        get_best_model_by_metric("nonexistent_metric", connection=conn)
    assert "Invalid metric_name: nonexistent_metric" in str(excinfo_model.value)


def test_get_best_config_by_metric(temp_db_with_data):
    """Test retrieving the best config by a given metric"""
    conn = temp_db_with_data["conn"]
    config_id = temp_db_with_data["config_id"]

    # Ensure there are no active configs to force fallback to best by metric
    execute_query("UPDATE configs SET is_active = 0", connection=conn)

    # Create necessary data for the query to work
    job_id = temp_db_with_data["job_id"]
    model_id = temp_db_with_data["model_id"]
    create_training_result(
        job_id, model_id, config_id, {"accuracy": 0.88}, {}, 120, connection=conn
    )

    best_config = get_best_config_by_metric(
        "accuracy", higher_is_better=True, connection=conn
    )

    assert best_config is not None
    assert best_config["config_id"] == config_id


def test_delete_model_record_and_file(temp_db_with_data, fs):
    """Test deleting a model record and its associated file"""
    conn = temp_db_with_data["conn"]
    model_id = temp_db_with_data["model_id"]

    # Create a fake model file for the record to point to
    model_path = "/path/to/model/to_delete.pkl"
    fs.create_file(model_path, contents="fake model data")
    # Update the record to point to our fake file
    conn.execute(
        "UPDATE models SET model_path = ? WHERE model_id = ?", (model_path, model_id)
    )
    conn.commit()

    assert fs.exists(model_path)

    # Act
    deleted = delete_model_record_and_file(model_id, connection=conn)

    # Assert
    assert deleted is True
    assert not fs.exists(model_path)  # File should be gone
    res = conn.execute(
        "SELECT * FROM models WHERE model_id = ?", (model_id,)
    ).fetchone()
    assert res is None  # Record should be gone


def test_delete_model_nonexistent_file(temp_db_with_data):
    """Test that deleting a model record succeeds even if the file is already gone"""
    conn = temp_db_with_data["conn"]
    model_id = temp_db_with_data["model_id"]

    # Ensure the file does NOT exist
    model_path = "/path/to/nonexistent_model.pkl"
    conn.execute(
        "UPDATE models SET model_path = ? WHERE model_id = ?", (model_path, model_id)
    )
    conn.commit()

    # Act
    deleted = delete_model_record_and_file(model_id, connection=conn)

    # Assert
    assert deleted is True  # Should report success
    res = conn.execute(
        "SELECT * FROM models WHERE model_id = ?", (model_id,)
    ).fetchone()
    assert res is None  # Record should still be gone


def test_delete_models_by_ids_with_active_model(temp_db_with_data):
    """Test that an active model is not deleted by delete_models_by_ids"""
    conn = temp_db_with_data["conn"]
    model_id = temp_db_with_data["model_id"]
    set_model_active(model_id, connection=conn)  # Make it active

    # Try to delete it
    result = delete_models_by_ids([model_id], connection=conn)

    assert result["deleted_count"] == 0
    assert result["skipped_count"] == 1
    assert model_id in result["skipped_models"]

    res = conn.execute(
        "SELECT * FROM models WHERE model_id = ?", (model_id,)
    ).fetchone()
    assert res is not None  # Should still exist


def test_delete_configs_by_ids_with_active_set(temp_db_with_data):
    """Test that an active config is not deleted by delete_configs_by_ids"""
    conn = temp_db_with_data["conn"]
    config_id = temp_db_with_data["config_id"]
    set_config_active(config_id, connection=conn)  # Make it active

    # Try to delete it
    result = delete_configs_by_ids([config_id], connection=conn)

    assert result["deleted_count"] == 0
    assert result["skipped_count"] == 1
    assert config_id in result["skipped_configs"]

    res = conn.execute(
        "SELECT * FROM configs WHERE config_id = ?", (config_id,)
    ).fetchone()
    assert res is not None  # Should still exist


def test_set_model_active_nonexistent_id(temp_db):
    """Test setting a non-existent model ID as active fails gracefully"""
    assert set_model_active("non-existent-id", connection=temp_db) is False


def test_set_config_active_nonexistent_id(temp_db):
    """Test setting a non-existent config ID as active fails gracefully"""
    assert set_config_active("non-existent-id", connection=temp_db) is False


def test_create_or_get_config_idempotent(temp_db):
    """Test that create_or_get_config is idempotent"""
    params = {"a": 1, "b": 2}

    config_id1 = create_or_get_config(params, connection=temp_db)
    config_id2 = create_or_get_config(params, connection=temp_db)

    assert config_id1 == config_id2

    res = temp_db.execute("SELECT COUNT(*) AS count FROM configs").fetchone()
    assert res["count"] == 1


@patch("deployment.app.db.database.create_or_get_config")
def test_concurrency_simulation(
    mock_create_config, isolated_db_session, create_training_params_fn
):
    """Simulate concurrent access to database functions using a file-based DB."""
    db_path = isolated_db_session

    # Create a simple mock for create_or_get_config
    mock_create_config.return_value = "mocked-config-id"

    def worker(db_path_for_thread, job_name_suffix):
        # Each thread must create its own connection to the shared file DB
        with get_db_connection(db_path_for_thread) as conn:
            try:
                # Simulate some work - simplified to avoid using the complex TrainingConfig
                params = {"batch_size": 32, "test": True, "thread": job_name_suffix}
                create_job(f"training_{job_name_suffix}", params, connection=conn)
            except Exception as e:
                # Store exceptions to be checked in the main thread
                exceptions.append(e)

    exceptions = []
    threads = []
    num_threads = 5

    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(db_path, f"thread{i}"))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Assert that no exceptions were raised
    assert len(exceptions) == 0, f"Exceptions were raised: {exceptions}"

    # Assert that jobs were created
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs")
        result = cursor.fetchone()

        # Handle both dictionary and tuple result formats
        if isinstance(result, dict):
            job_count = result["COUNT(*)"]
        else:
            job_count = result[0]

        assert job_count == num_threads


def json_default_serializer(obj):
    """
    JSON serializer for objects not serializable by default json code
    """
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def test_transaction_rollback_on_error(isolated_db_session, create_training_params_fn):
    """Test that a transaction is rolled back if an error occurs."""
    db_path = isolated_db_session

    with get_db_connection(db_path) as conn:
        # Initialize database schema if it doesn't exist
        conn.executescript(SCHEMA_SQL)

        # Get initial count using cursor directly
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs")
        result = cursor.fetchone()

        # Handle both dictionary and tuple result formats
        if isinstance(result, dict):
            initial_count = result["COUNT(*)"]
        else:
            initial_count = result[0]

        with pytest.raises(sqlite3.OperationalError):
            with conn:  # Start a transaction
                # This one is fine
                create_job("job1", {"p": 1}, connection=conn)
                # This one will fail due to a non-existent table, triggering a rollback
                conn.execute("INSERT INTO non_existent_table VALUES (1)")

        # Verify that the first job was rolled back
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs")
        result = cursor.fetchone()

        # Handle both dictionary and tuple result formats
        if isinstance(result, dict):
            final_count = result["COUNT(*)"]
        else:
            final_count = result[0]

        assert final_count == initial_count


def test_database_error_propagation():
    """Test that DatabaseError properly propagates from lower-level functions"""
    # Mock sqlite3.connect to raise an error
    with patch(
        "sqlite3.connect", side_effect=sqlite3.OperationalError("Connection error")
    ):
        # Try to get a connection
        with pytest.raises(DatabaseError) as exc_info:
            get_db_connection()

        # Verify error details
        assert "Database connection failed" in str(exc_info.value)
        assert isinstance(exc_info.value.original_error, sqlite3.OperationalError)

        # Now try to execute a query (which calls get_db_connection)
        with pytest.raises(DatabaseError) as exc_info:
            execute_query("SELECT 1")

        # Verify error details
        assert "Database connection failed" in str(exc_info.value)
