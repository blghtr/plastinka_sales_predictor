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

import pytest
import sqlite3
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import threading
import tempfile
import shutil

from deployment.app.db.database import (
    get_db_connection,
    execute_query,
    execute_many,
    create_job,
    update_job_status,
    get_job,
    create_or_get_parameter_set,
    set_parameter_set_active,
    create_model_record,
    set_model_active,
    get_best_model_by_metric,
    get_best_parameter_set_by_metric,
    delete_model_record_and_file,
    delete_models_by_ids,
    delete_parameter_sets_by_ids,
    DatabaseError
)
from deployment.app.db.schema import SCHEMA_SQL

# Import settings for patching
from deployment.app.config import DatabaseSettings, settings


# Note: Using fixtures from parent conftest.py files:
# - temp_db: Creates a temporary SQLite database with schema
# - setup_db_with_data: Sets up the database with test data

# =============================================
# Resilience tests
# =============================================

def test_database_connection_nonexistent_dir(mocker):
    """Test that database path directory is created if it doesn't exist"""
    # Set DATABASE_PATH to a non-existent directory
    non_existent_dir_name = "nonexistent_db_dir"
    non_existent_file_name = "test_db.sqlite"
    
    # Use a temporary directory managed by pytest for the base of our non-existent path
    # to ensure cleanup and avoid polluting the source tree.
    # However, the point of the test is that the parent dir of non_existent_dir is created.
    # Let's construct the path carefully.
    # We create a base temp dir, then non_existent_dir will be a child of this that doesn't exist yet.
    
    base_temp_dir = tempfile.TemporaryDirectory()
    non_existent_parent_path = os.path.join(base_temp_dir.name, non_existent_dir_name)
    non_existent_db_path = os.path.join(non_existent_parent_path, non_existent_file_name)

    # Ensure the parent directory does NOT exist before the call
    if os.path.exists(non_existent_parent_path):
        shutil.rmtree(non_existent_parent_path) # Clean up if it somehow exists from a previous failed run
    assert not os.path.exists(non_existent_parent_path)

    conn = None  # Initialize conn to None
    try:
        # Patch DatabaseSettings.reload to be a no-op that returns the instance
        def reload_side_effect_for_class(instance_self):
            return instance_self # Mimics original reload returning self

        mocker.patch.object(DatabaseSettings, 'reload', side_effect=reload_side_effect_for_class, autospec=True)
        mocker.patch.object(settings.db, 'path', new=non_existent_db_path)
        
        conn = get_db_connection() # Call the function that uses the patched DB_PATH
        assert isinstance(conn, sqlite3.Connection)

        # Verify directory and file were created
        assert os.path.exists(non_existent_parent_path) # Check parent dir
        assert os.path.exists(non_existent_db_path)   # Check db file itself
            
    except Exception as e_outer: # Renamed to avoid conflict with inner 'e' if any
        # print(f"Test failed with exception: {e_outer}") # For debugging if needed
        raise # Re-raise the exception to fail the test
    finally:
        if conn: # Ensure conn was successfully created before trying to close
            conn.close()
        # Ensure parent directory of non_existent_db_path and the file are cleaned up 
        # if they were created by the test. base_temp_dir.cleanup() handles this.
        if os.path.exists(non_existent_parent_path):
             shutil.rmtree(non_existent_parent_path)
        base_temp_dir.cleanup() # Cleanup the base temporary directory

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
        original_error=original_error
    )
    
    # Verify error properties
    assert error.message == "Database operation failed"
    assert error.query == query
    assert error.params == params
    assert error.original_error == original_error
    assert str(error) == "Database operation failed"

def test_execute_query_connection_error():
    """Test that execute_query handles connection errors"""
    # Mock get_db_connection to raise an error
    with patch('deployment.app.db.database.get_db_connection', 
               side_effect=DatabaseError("Connection failed")):
        
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
    # Mock get_db_connection to raise an error
    with patch('deployment.app.db.database.get_db_connection', 
               side_effect=DatabaseError("Connection failed")):
        
        # Function should propagate the error
        with pytest.raises(DatabaseError) as exc_info:
            execute_many("INSERT INTO table VALUES (?)", [("value1",), ("value2",)])
        
        # Verify error message
        assert "Connection failed" in str(exc_info.value)

def test_create_job_duplicate_id():
    """Test creating a job with a duplicate ID"""
    # Mock generate_id to always return the same ID
    fixed_id = "duplicate-id"
    
    with patch('deployment.app.db.database.generate_id', return_value=fixed_id):
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
    """Test updating status for a non-existent job"""
    conn = temp_db["conn"]

    # Update non-existent job
    non_existent_id = "nonexistent-job-id"

    # Should not raise an error, just update nothing
    update_job_status(non_existent_id, "running", connection=conn)

    # Verify no job was updated
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS count FROM jobs WHERE job_id = ?", (non_existent_id,))
    result = cursor.fetchone()
    assert result is not None, "fetchone() should return a row for COUNT(*)"
    count = result['count']
    assert count == 0

def test_get_best_model_by_metric(setup_db_with_data):
    """Test getting the best model based on a metric"""
    conn = setup_db_with_data["conn"]
    
    # Get best model by MAPE (lower is better)
    best_model = get_best_model_by_metric("mape", higher_is_better=False, connection=conn)
    
    # Should return model-2 which has lower MAPE
    assert best_model is not None
    assert best_model["model_id"] == "model-2"
    assert best_model["metrics"]["mape"] == 9.8
    
    # Get best model by non-existent metric
    with pytest.raises(ValueError) as excinfo_model:
        get_best_model_by_metric("nonexistent_metric", connection=conn)
    assert "Invalid metric_name: nonexistent_metric" in str(excinfo_model.value)

def test_get_best_parameter_set_by_metric(setup_db_with_data):
    """Test getting the best parameter set based on a metric"""
    conn = setup_db_with_data["conn"]
    
    # Get best parameter set by MAPE (lower is better)
    best_params = get_best_parameter_set_by_metric("mape", higher_is_better=False, connection=conn)
    
    # Should return param-2 which has lower MAPE
    assert best_params is not None
    assert best_params["parameter_set_id"] == "param-2"
    assert best_params["metrics"]["mape"] == 9.8
    
    # Get best parameter set by non-existent metric
    with pytest.raises(ValueError) as excinfo_params:
        get_best_parameter_set_by_metric("nonexistent_metric", connection=conn)
    assert "Invalid metric_name: nonexistent_metric" in str(excinfo_params.value)

def test_delete_model_record_and_file(setup_db_with_data):
    """Test deleting a model record and its file"""
    conn = setup_db_with_data["conn"]
    model_path_1 = setup_db_with_data["model_path_1"]
    
    # Verify model file exists
    assert os.path.exists(model_path_1)
    
    # Mock get_db_connection to return our test connection
    with patch('deployment.app.db.database.get_db_connection', return_value=conn):
        # Delete the model, passing the connection explicitly
        result = delete_model_record_and_file("model-1", connection=conn)
        
        # Verify deletion was successful
        assert result is True
        
        # Verify record is gone from database
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM models WHERE model_id = ?", ("model-1",))
        result = cursor.fetchone()
        assert result is not None, "fetchone() should return a row for COUNT(*)"
        count = result['count']
        assert count == 0
        
        # Verify file is gone from filesystem
        assert not os.path.exists(model_path_1)


def test_delete_model_nonexistent_file(setup_db_with_data):
    """Test deleting a model record with a non-existent file"""
    conn = setup_db_with_data["conn"]
    
    # Update model-2 to have a non-existent file path
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE models SET model_path = ? WHERE model_id = ?",
        ("nonexistent/path.onnx", "model-2")
    )
    conn.commit()
    
    # Mock get_db_connection
    with patch('deployment.app.db.database.get_db_connection', return_value=conn):
        # Delete the model, passing the connection explicitly
        result = delete_model_record_and_file("model-2", connection=conn)
        
        # Should still return True (record deleted, file not found)
        assert result is True
        
        # Verify record is gone from database
        cursor.execute("SELECT COUNT(*) AS count FROM models WHERE model_id = ?", ("model-2",))
        result = cursor.fetchone()
        assert result is not None, "fetchone() should return a row for COUNT(*)"
        count = result['count']
        assert count == 0

def test_delete_models_by_ids_with_active_model(setup_db_with_data):
    """Test trying to delete models including an active model"""
    conn = setup_db_with_data["conn"]
    
    # Try to delete both models (one is active)
    with patch('deployment.app.db.database.get_db_connection', return_value=conn):
        result = delete_models_by_ids(["model-1", "model-2"], connection=conn)
        
        # Verify result
        assert result["successful"] == 1  # Only model-2 should be deleted
        assert result["failed"] == 1      # model-1 is active and can't be deleted
        assert len(result["errors"]) == 1
        assert "Cannot delete active models: model-1" in result["errors"][0]
        
        # Verify only model-2 was deleted
        cursor = conn.cursor()
        cursor.execute("SELECT model_id FROM models")
        remaining_models = [row['model_id'] for row in cursor.fetchall()]
        assert "model-1" in remaining_models # Active model should remain
        assert "model-2" not in remaining_models # Non-active should be gone
        
        # Verify files (model-2.onnx should be gone)
        assert not os.path.exists(setup_db_with_data["model_path_2"])

def test_delete_parameter_sets_by_ids_with_active_set(setup_db_with_data):
    """Test trying to delete parameter sets including an active set"""
    conn = setup_db_with_data["conn"]
    
    # Try to delete both parameter sets (one is active)
    with patch('deployment.app.db.database.get_db_connection', return_value=conn):
        result = delete_parameter_sets_by_ids(["param-1", "param-2"], connection=conn)
        
        # Verify result
        assert result["successful"] == 1  # Only param-2 should be deleted
        assert result["failed"] == 1      # param-1 is active and can't be deleted
        assert len(result["errors"]) == 1
        assert "Cannot delete active parameter sets: param-1" in result["errors"][0]
        
        # Verify only param-2 was deleted
        cursor = conn.cursor()
        cursor.execute("SELECT parameter_set_id FROM parameter_sets")
        remaining_params = [row['parameter_set_id'] for row in cursor.fetchall()]
        assert "param-1" in remaining_params # Active set should remain
        assert "param-2" not in remaining_params # Non-active should be gone

def test_set_model_active_nonexistent_id(setup_db_with_data):
    """Test setting a non-existent model as active"""
    conn = setup_db_with_data["conn"]
    
    # Try to set a non-existent model as active
    result = set_model_active("nonexistent-model-id", connection=conn)
    
    # Should return False
    assert result is False

def test_set_parameter_set_active_nonexistent_id(setup_db_with_data):
    """Test setting a non-existent parameter set as active"""
    conn = setup_db_with_data["conn"]
    
    # Try to set a non-existent parameter set as active
    result = set_parameter_set_active("nonexistent-param-id", connection=conn)
    
    # Should return False
    assert result is False

def test_create_or_get_parameter_set_idempotent(temp_db, create_training_params_fn):
    """Test that create_or_get_parameter_set is idempotent."""
    conn = temp_db["conn"]

    try:
        # Correctly use the injected fixture that returns a function
        params_creator_func = create_training_params_fn
        training_params_obj = params_creator_func({"batch_size": 16, "dropout": 0.1})
        params_to_save = training_params_obj.model_dump()

        # Create first time
        param_set_id_1 = create_or_get_parameter_set(params_to_save, connection=conn)
        param_set_id_2 = create_or_get_parameter_set(params_to_save, connection=conn)

        assert param_set_id_1 == param_set_id_2

        # Verify it's in the database
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM parameter_sets WHERE parameter_set_id = ?", (param_set_id_1,))
        result = cursor.fetchone()
        assert result is not None
        assert result['count'] == 1
    finally:
        conn.close()

@patch('deployment.app.db.database.create_or_get_parameter_set')
def test_concurrency_simulation(mock_create_param_set, temp_db, create_training_params_fn):
    """Simulate concurrent access to database functions."""
    # conn1 and conn2 from main thread will not be used by worker threads directly for DB operations.
    # They can be used for setup if needed, but worker threads will create their own connections.

    db_path = temp_db["db_path"] # Get path for threads to use

    params_creator_func = create_training_params_fn
    training_params_obj = params_creator_func({"learning_rate": 0.01, "batch_size": 16})
    params_for_job = training_params_obj.model_dump()

    mock_create_param_set.return_value = "param_set_123"

    results = []
    exceptions = [] # To store exceptions from threads

    def worker(db_path_for_thread, job_params, job_name_suffix):
        worker_conn = None
        try:
            # Each thread creates its own connection
            worker_conn = sqlite3.connect(db_path_for_thread)
            # It's crucial that functions called here (create_job, update_job_status)
            # correctly handle this passed 'worker_conn' or establish their own if not passed.
            # The database.py functions *should* use the passed connection if provided.
            
            job_id = create_job("training", job_params, connection=worker_conn)
            update_job_status(job_id, "completed", connection=worker_conn)
            results.append(True)
        except Exception as e:
            print(f"Error in worker {job_name_suffix}: {e}")
            exceptions.append(e)
            results.append(False)
        finally:
            if worker_conn:
                worker_conn.close()

    thread1 = threading.Thread(target=worker, args=(db_path, params_for_job, "job1"))
    thread2 = threading.Thread(target=worker, args=(db_path, params_for_job, "job2"))

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    
    # For debugging, print exceptions
    if exceptions:
        print("Exceptions from threads:")
        for exc in exceptions:
            print(exc)

    assert all(results), "All concurrent operations should succeed. Check printed exceptions."

def test_transaction_rollback_on_error(temp_db, create_training_params_fn):
    """Test that a transaction is rolled back if an error occurs."""
    conn = temp_db["conn"]

    try:
        params_creator_func = create_training_params_fn
        training_params_obj = params_creator_func()
        params_dict_for_set = training_params_obj.model_dump()

        # Start a transaction
        conn.execute("BEGIN TRANSACTION") # Use execute for non-query
        try:
            # Create a parameter set
            create_or_get_parameter_set(params_dict_for_set, connection=conn)

            # Simulate an error by trying to insert into a non-existent table
            conn.execute("INSERT INTO non_existent_table VALUES ('error')")
            conn.commit() # This should not be reached
        except sqlite3.OperationalError as e:
            assert "no such table: non_existent_table" in str(e)
            conn.rollback()
        except Exception as e_other:
            conn.rollback()
            pytest.fail(f"Unexpected error during transaction: {e_other}")
        else:
            conn.rollback()
            pytest.fail("sqlite3.OperationalError was not raised as expected.")

        # Verify parameter set was not created due to rollback
        cursor = conn.cursor()
        parameters_json_for_query = json.dumps(params_dict_for_set, sort_keys=True)
        cursor.execute("SELECT COUNT(*) AS count FROM parameter_sets WHERE parameters = ?", (parameters_json_for_query,))
        result = cursor.fetchone()
        assert result is not None, "COUNT(*) should always return a row."
        count = result['count']
        assert count == 0, "Parameter set should have been rolled back."
    finally:
        conn.close()

def test_database_error_propagation():
    """Test that DatabaseError properly propagates from lower-level functions"""
    # Mock sqlite3.connect to raise an error
    with patch('sqlite3.connect', side_effect=sqlite3.OperationalError("Connection error")):
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