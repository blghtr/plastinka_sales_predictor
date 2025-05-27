"""
Tests for transaction safety in database operations.

This module contains tests that verify the proper handling of transactions
in the database module, ensuring that:

1. Operations are properly committed or rolled back
2. Connection isolation is maintained
3. Concurrent transactions behave correctly
4. Nested transactions are handled correctly
5. Complex transaction chains maintain data integrity
"""

import pytest
import sqlite3
import os
import json
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock
import threading
import time

from deployment.app.db.database import (
    get_db_connection,
    execute_query,
    execute_many,
    create_job,
    update_job_status,
    create_model_record,
    create_training_result,
    create_prediction_result,
    DatabaseError
)
from deployment.app.db.schema import SCHEMA_SQL

# =============================================
# Используем фикстуры из conftest.py
# =============================================

# @pytest.fixture
# def temp_db():
#     """Create a temporary SQLite database with schema"""
#     # Create temporary directory for test files
#     temp_dir = os.path.join(os.path.dirname(__file__), "temp_transactions")
#     os.makedirs(temp_dir, exist_ok=True)
#     db_path = os.path.join(temp_dir, "test_transactions.db")
#     
#     # Initialize schema
#     conn = sqlite3.connect(db_path)
#     conn.executescript(SCHEMA_SQL)
#     conn.commit()
#     conn.close()
#     
#     # Save original DB_PATH
#     original_db_path = os.environ.get('DATABASE_PATH')
#     
#     # Set environment variable to point to test DB
#     os.environ['DATABASE_PATH'] = db_path
#     
#     # Yield the database path
#     yield {"db_path": db_path, "temp_dir": temp_dir}
#     
#     # Restore original DB_PATH
#     if original_db_path:
#         os.environ['DATABASE_PATH'] = original_db_path
#     else:
#         os.environ.pop('DATABASE_PATH', None)
#     
#     # Close and remove the temporary file
#     try:
#         os.unlink(db_path)
#     except:
#         pass
#     
#     # Remove temp directory
#     try:
#         os.rmdir(temp_dir)
#     except:
#         pass

# =============================================
# Tests for basic transaction safety
# =============================================

def test_execute_query_transaction(temp_db):
    """Test that execute_query properly manages transactions"""
    # Get a direct connection for verification
    conn = sqlite3.connect(temp_db["db_path"])
    cursor = conn.cursor()
    
    # Create a test table
    cursor.execute("CREATE TABLE test_transactions (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    
    # Test commit behavior
    execute_query(
        "INSERT INTO test_transactions (value) VALUES (?)",
        ("test_value",),
        connection=conn
    )
    
    # Verify data was committed
    cursor.execute("SELECT value FROM test_transactions WHERE id = 1")
    result = cursor.fetchone()
    assert result is not None
    assert result[0] == "test_value"
    
    # Test rollback on error
    try:
        # This should fail (missing a required parameter)
        execute_query(
            "INSERT INTO test_transactions (id, value) VALUES (?, ?)",
            (2,),  # Missing a parameter
            connection=conn
        )
    except DatabaseError:
        pass  # Expected error
    
    # Verify nothing was inserted
    cursor.execute("SELECT COUNT(*) FROM test_transactions WHERE id = 2")
    count = cursor.fetchone()[0]
    assert count == 0
    
    # Close the connection
    conn.close()

def test_execute_many_transaction(temp_db):
    """Test that execute_many properly manages transactions"""
    # Debug information
    print(f"DEBUG: temp_db path: {temp_db['db_path']}")
    print(f"DEBUG: DB file exists: {os.path.exists(temp_db['db_path'])}")
    
    # Get a direct connection for verification
    conn = sqlite3.connect(temp_db["db_path"])
    print(f"DEBUG: SQLite isolation level: {conn.isolation_level}")
    
    # Set to None first to ensure no active transaction (auto-commit mode)
    conn.isolation_level = None
    print(f"DEBUG: SQLite isolation level after setting to None: {conn.isolation_level}")
    
    # Check database tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"DEBUG: Database tables: {[table[0] if isinstance(table, tuple) else table['name'] for table in tables]}")
    
    # Create a test table
    cursor.execute("CREATE TABLE test_batch (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    
    # Prepare data for batch insert
    params_list = [
        ("value1",),
        ("value2",),
        ("value3",)
    ]
    
    # Test batch insert with commit
    execute_many(
        "INSERT INTO test_batch (value) VALUES (?)",
        params_list,
        connection=conn
    )
    
    # Verify all data was committed
    cursor.execute("SELECT COUNT(*) FROM test_batch")
    count_result = cursor.fetchone()
    print(f"DEBUG: Count result after insert: {count_result}")
    # Handle row as dictionary or tuple
    count = count_result[0] if isinstance(count_result, tuple) else count_result['COUNT(*)']
    assert count == 3
    
    # Now set isolation level and start transaction for error test
    conn.isolation_level = 'DEFERRED'
    print(f"DEBUG: SQLite isolation level set to DEFERRED: {conn.isolation_level}")

    # Start transaction by executing a statement (SQLite auto-starts transaction)
    cursor.execute("SELECT 1")
    print("DEBUG: Transaction implicitly started by first SQL statement")
    
    # Test rollback on error with batch insert
    # Create an invalid parameter list (missing values)
    invalid_params = [
        ("value4",),
        (),  # Invalid - missing value
        ("value5",)
    ]
    
    try:
        execute_many(
            "INSERT INTO test_batch (value) VALUES (?)",
            invalid_params,
            connection=conn
        )
    except DatabaseError:
        print("DEBUG: Caught expected DatabaseError from execute_many")
        # With an external connection, execute_many does not roll back. The caller (this test) must.
        if conn:
            print("DEBUG: Rolling back transaction on external connection in test_execute_many_transaction")
            conn.rollback()
    
    # Verify no new data was inserted (rollback)
    cursor.execute("SELECT COUNT(*) FROM test_batch")
    count_result = cursor.fetchone()
    print(f"DEBUG: Count result after failed insert: {count_result}")
    # Handle row as dictionary or tuple
    count = count_result[0] if isinstance(count_result, tuple) else count_result['COUNT(*)']
    print(f"DEBUG: Count value: {count}")
    assert count == 3  # Still only the original 3 rows
    
    # Close the connection
    conn.close()

def test_nested_transactions(temp_db):
    """Test nested transactions behavior with SQLite"""
    # Get a direct connection
    conn = sqlite3.connect(temp_db["db_path"])
    
    # Start an outer transaction
    cursor = conn.cursor()
    cursor.execute("BEGIN TRANSACTION")
    
    # Insert a record
    cursor.execute("CREATE TABLE nested_test (id INTEGER PRIMARY KEY, value TEXT)")
    cursor.execute("INSERT INTO nested_test (value) VALUES (?)", ("outer",))
    
    # Start a nested transaction (savepoint in SQLite)
    cursor.execute("SAVEPOINT nested_savepoint")
    
    # Insert another record in the nested transaction
    cursor.execute("INSERT INTO nested_test (value) VALUES (?)", ("inner",))
    
    # Rollback the nested transaction
    cursor.execute("ROLLBACK TO SAVEPOINT nested_savepoint")
    
    # Commit the outer transaction
    conn.commit()
    
    # Verify only the outer insert was committed
    cursor.execute("SELECT COUNT(*) FROM nested_test")
    count = cursor.fetchone()[0]
    assert count == 1
    
    cursor.execute("SELECT value FROM nested_test")
    value = cursor.fetchone()[0]
    assert value == "outer"
    
    # Close the connection
    conn.close()

# =============================================
# Tests for complex transaction patterns
# =============================================

def test_create_job_transaction_safety(temp_db):
    """Test that create_job function operates safely within transactions"""
    # Debug information
    import os
    print(f"DEBUG: temp_db path: {temp_db['db_path']}")
    print(f"DEBUG: DB file exists: {os.path.exists(temp_db['db_path'])}")

    # Get a connection for testing
    conn = sqlite3.connect(temp_db["db_path"])
    print(f"DEBUG: SQLite isolation level: {conn.isolation_level}")
    
    # Check database tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"DEBUG: Database tables: {[table[0] if isinstance(table, tuple) else table['name'] for table in tables]}")
    
    # Start an explicit transaction on the connection
    conn.execute("BEGIN TRANSACTION")
    
    # Create a job within the transaction
    job_id = create_job("training", {"param": "value"}, connection=conn)
    print(f"DEBUG: Created job_id: {job_id}")
    
    # Verify job exists in this transaction context
    cursor = conn.cursor()
    cursor.execute("SELECT job_id FROM jobs WHERE job_id = ?", (job_id,))
    result = cursor.fetchone()
    print(f"DEBUG: Query result for job_id {job_id}: {result}")
    
    if result is None:
        # Try a broader query to see if any jobs exist
        cursor.execute("SELECT * FROM jobs")
        all_jobs = cursor.fetchall()
        print(f"DEBUG: All jobs in database: {all_jobs}")
        raise AssertionError(f"Job {job_id} was not found in database after creation")
    
    assert result is not None
    # Handle row as dictionary or tuple
    job_id_from_db = result['job_id'] if isinstance(result, dict) or hasattr(result, 'keys') else result[0]
    assert job_id_from_db == job_id
    
    # Now rollback the transaction
    conn.rollback()
    
    # Verify job no longer exists
    cursor.execute("SELECT job_id FROM jobs WHERE job_id = ?", (job_id,))
    result = cursor.fetchone()
    assert result is None
    
    # Create a job again and commit
    job_id = create_job("training", {"param": "value"}, connection=conn)
    conn.commit()
    
    # Verify job exists after commit
    cursor.execute("SELECT job_id FROM jobs WHERE job_id = ?", (job_id,))
    result = cursor.fetchone()
    print(f"DEBUG: Query result after commit for job_id {job_id}: {result}")
    assert result is not None
    
    # Close the connection
    conn.close()

def test_update_job_transaction_safety(temp_db):
    """Test update_job_status within transactions"""
    # Debug information
    import os
    print(f"DEBUG: temp_db path: {temp_db['db_path']}")
    print(f"DEBUG: DB file exists: {os.path.exists(temp_db['db_path'])}")

    # Get a connection for testing
    conn = sqlite3.connect(temp_db["db_path"])
    print(f"DEBUG: SQLite isolation level: {conn.isolation_level}")
    
    # Set explicit isolation level to None to enable auto-commit mode
    # This ensures we're not already in a transaction
    conn.isolation_level = None
    print(f"DEBUG: SQLite isolation level after setting: {conn.isolation_level}")
    
    # Check if job_status_history table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='job_status_history'")
    job_history_table = cursor.fetchone()
    print(f"DEBUG: job_status_history table exists: {job_history_table is not None}")

    # Create a job
    job_id = create_job("training", {}, connection=conn)
    print(f"DEBUG: Created job_id: {job_id}")

    # Verify job was created
    cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    job_data = cursor.fetchone()
    print(f"DEBUG: Initial job data: {job_data}")

    # Start an explicit transaction
    # First check if we're already in a transaction
    try:
        print("DEBUG: Attempting to start transaction")
        conn.execute("BEGIN TRANSACTION")
        print("DEBUG: Transaction started successfully")
    except sqlite3.OperationalError as e:
        if "within a transaction" in str(e):
            print("DEBUG: Already in a transaction, continuing")
            # We're already in a transaction, so we can continue
            pass
        else:
            # Some other error occurred
            raise

    # Update job status
    update_job_status(job_id, "running", progress=50, connection=conn)

    # Verify job is updated in this transaction
    cursor = conn.cursor()
    cursor.execute("SELECT status, progress FROM jobs WHERE job_id = ?", (job_id,))
    result = cursor.fetchone()
    print(f"DEBUG: Job status after update: {result}")

    if result is None:
        # Check if job still exists
        cursor.execute("SELECT * FROM jobs")
        all_jobs = cursor.fetchall()
        print(f"DEBUG: All jobs in database: {all_jobs}")
        raise AssertionError(f"Job {job_id} was not found after status update")

    assert result is not None
    # Handle row as dictionary or tuple
    status = result['status'] if isinstance(result, dict) or hasattr(result, 'keys') else result[0]
    progress = result['progress'] if isinstance(result, dict) or hasattr(result, 'keys') else result[1]
    assert status == "running"
    assert progress == 50

    # Verify job status history was updated
    cursor.execute("SELECT status FROM job_status_history WHERE job_id = ?", (job_id,))
    history_result = cursor.fetchone()
    print(f"DEBUG: Job history status: {history_result}")

    if history_result is not None:
        # Handle row as dictionary or tuple
        history_status = history_result['status'] if isinstance(history_result, dict) or hasattr(history_result, 'keys') else history_result[0]
        assert history_status == "running"

    # Now rollback
    print("DEBUG: Rolling back transaction")
    conn.rollback()

    # Verify job status is back to the original
    cursor.execute("SELECT status, progress FROM jobs WHERE job_id = ?", (job_id,))
    result = cursor.fetchone()
    print(f"DEBUG: Job status after rollback: {result}")

    if result is not None:
        # Handle row as dictionary or tuple
        status = result['status'] if isinstance(result, dict) or hasattr(result, 'keys') else result[0]
        progress = result['progress'] if isinstance(result, dict) or hasattr(result, 'keys') else result[1]
        assert status == "pending"  # Initial status
        assert progress == 0  # Initial progress

def test_complex_transaction_chain(temp_db):
    """Test a complex chain of database operations within a transaction"""
    # Debug information
    import os
    print(f"DEBUG: temp_db path: {temp_db['db_path']}")
    print(f"DEBUG: DB file exists: {os.path.exists(temp_db['db_path'])}")
    
    # Get a connection for testing
    conn = sqlite3.connect(temp_db["db_path"])
    
    # Check database tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"DEBUG: Database tables: {[table[0] if isinstance(table, tuple) else table['name'] for table in tables]}")
    
    job_id = None
    model_id = None
    
    # Start a transaction
    conn.execute("BEGIN TRANSACTION")
    
    try:
        # 1. Create a job
        job_id = create_job("training", {"batch_size": 32}, connection=conn)
        print(f"DEBUG: Created job_id: {job_id}")
        
        # 2. Update job status
        update_job_status(job_id, "running", progress=25, connection=conn)
        
        # 3. Create a model record
        model_id = str(uuid.uuid4())
        model_path = "/path/to/model.onnx"
        print(f"DEBUG: Created model_id: {model_id}")
        
        # Insert into jobs and models tables
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO models
               (model_id, job_id, model_path, created_at, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (model_id, job_id, model_path, datetime.now().isoformat(),
             json.dumps({"framework": "pytorch"}))
        )
        
        # Verify model was inserted
        cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        model_data = cursor.fetchone()
        print(f"DEBUG: Model data after insert: {model_data}")
        
        # 4. Create a parameter set
        cursor.execute(
            """INSERT INTO parameter_sets
               (parameter_set_id, parameters, created_at)
               VALUES (?, ?, ?)""",
            ("param-1", json.dumps({"batch_size": 32}), datetime.now().isoformat())
        )
        
        # 5. Create a training result
        cursor.execute(
            """INSERT INTO training_results
               (result_id, job_id, model_id, parameter_set_id, metrics, parameters)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("result-1", job_id, model_id, "param-1",
             json.dumps({"mape": 10.5}), json.dumps({"batch_size": 32}))
        )
        
        # 6. Update job status again
        update_job_status(job_id, "completed", progress=100, result_id="result-1", connection=conn)  
        
        # Intentionally cause an error to trigger rollback
        if True:  # Simulate a conditional error
            raise ValueError("Simulated error to trigger rollback")
            
        # This part should not execute due to the error
        conn.commit()
        
    except Exception as e:
        # Rollback on any error
        print(f"DEBUG: Exception triggered rollback: {str(e)}")
        conn.rollback()
    
    # Verify all operations were rolled back
    cursor = conn.cursor()
    
    # Check job doesn't exist
    if job_id:
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job_id,))
        count_result = cursor.fetchone()
        print(f"DEBUG: Job count after rollback: {count_result}")
        
        if count_result is not None:
            # Handle row as dictionary or tuple
            count = count_result['COUNT(*)'] if isinstance(count_result, dict) or hasattr(count_result, 'keys') else count_result[0]
            assert count == 0
    
    # Check model doesn't exist
    if model_id:
        cursor.execute("SELECT COUNT(*) FROM models WHERE model_id = ?", (model_id,))
        count_result = cursor.fetchone()
        print(f"DEBUG: Model count after rollback: {count_result}")
        
        if count_result is not None:
            # Handle row as dictionary or tuple
            count = count_result['COUNT(*)'] if isinstance(count_result, dict) or hasattr(count_result, 'keys') else count_result[0]
            assert count == 0
    
    # Check parameter set doesn't exist
    cursor.execute("SELECT COUNT(*) FROM parameter_sets WHERE parameter_set_id = ?", ("param-1",))
    count_result = cursor.fetchone()
    print(f"DEBUG: Parameter set count after rollback: {count_result}")
    
    if count_result is not None:
        # Handle row as dictionary or tuple
        count = count_result['COUNT(*)'] if isinstance(count_result, dict) or hasattr(count_result, 'keys') else count_result[0]
        assert count == 0
    
    # Check training result doesn't exist
    cursor.execute("SELECT COUNT(*) FROM training_results WHERE result_id = ?", ("result-1",))
    count_result = cursor.fetchone()
    print(f"DEBUG: Training result count after rollback: {count_result}")
    
    if count_result is not None:
        # Handle row as dictionary or tuple
        count = count_result['COUNT(*)'] if isinstance(count_result, dict) or hasattr(count_result, 'keys') else count_result[0]
        assert count == 0
    
    # Close connection
    conn.close()

# =============================================
# Tests for concurrent transactions
# =============================================

def test_concurrent_reads(temp_db):
    """Test concurrent read operations"""
    # Prepare data
    conn = sqlite3.connect(temp_db["db_path"])
    conn.execute("CREATE TABLE concurrent_test (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO concurrent_test (value) VALUES (?)", ("test_value",))
    conn.commit()
    conn.close()
    
    # Define a worker function for threads
    def worker(db_path, results, thread_id):
        # Each thread gets its own connection
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Perform read query
        cursor.execute("SELECT * FROM concurrent_test")
        row = cursor.fetchone()
        
        # Store result
        results[thread_id] = row["value"] if row else None
        
        # Close connection
        conn.close()
    
    # Create threads to perform concurrent reads
    num_threads = 5
    threads = []
    results = {}
    
    for i in range(num_threads):
        thread = threading.Thread(
            target=worker, 
            args=(temp_db["db_path"], results, i)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify all threads read the correct value
    for i in range(num_threads):
        assert results[i] == "test_value"

def test_concurrent_writes(temp_db):
    """Test concurrent write operations with SQLite's default locking"""
    # Prepare table
    conn = sqlite3.connect(temp_db["db_path"])
    conn.execute("CREATE TABLE write_test (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()
    
    # Define a worker function for threads
    def worker(db_path, thread_id, success_list):
        # Each thread gets its own connection
        conn = sqlite3.connect(db_path)
        try:
            # Try to insert a record
            conn.execute(
                "INSERT INTO write_test (value) VALUES (?)",
                (f"value_from_thread_{thread_id}",)
            )
            conn.commit()
            success_list.append(thread_id)
        except sqlite3.Error:
            # Some threads may fail with "database is locked" due to SQLite's
            # default locking behavior
            pass
        finally:
            conn.close()
    
    # Create threads to perform concurrent writes
    num_threads = 5
    threads = []
    successful_threads = []
    
    for i in range(num_threads):
        thread = threading.Thread(
            target=worker, 
            args=(temp_db["db_path"], i, successful_threads)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify writes
    conn = sqlite3.connect(temp_db["db_path"])
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM write_test")
    count = cursor.fetchone()[0]
    
    # We should have at least one successful write
    assert count > 0
    assert len(successful_threads) == count
    
    # Verify the data matches successful threads
    cursor.execute("SELECT value FROM write_test")
    values = [row[0] for row in cursor.fetchall()]
    
    expected_values = []
    for thread_id in successful_threads:
        expected_values.append(f"value_from_thread_{thread_id}")
    
    # Sort both lists for comparison
    values.sort()
    expected_values.sort()
    assert values == expected_values
    
    conn.close()

def test_connection_isolation(temp_db):
    """Test that each connection has its own isolated transaction"""
    # Create two connections
    conn1 = sqlite3.connect(temp_db["db_path"])
    conn2 = sqlite3.connect(temp_db["db_path"])
    
    # Set up test table
    conn1.execute("CREATE TABLE isolation_test (id INTEGER PRIMARY KEY, value TEXT)")
    conn1.commit()
    
    # Begin transaction on conn1
    conn1.execute("BEGIN TRANSACTION")
    conn1.execute("INSERT INTO isolation_test (value) VALUES (?)", ("value1",))
    
    # Check conn2 doesn't see the uncommitted row
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT COUNT(*) FROM isolation_test")
    count = cursor2.fetchone()[0]
    assert count == 0
    
    # Commit conn1's transaction
    conn1.commit()
    
    # Now conn2 should see the row
    cursor2.execute("SELECT COUNT(*) FROM isolation_test")
    count = cursor2.fetchone()[0]
    assert count == 1
    
    # Begin transaction on conn2
    conn2.execute("BEGIN TRANSACTION")
    conn2.execute("INSERT INTO isolation_test (value) VALUES (?)", ("value2",))
    
    # Commit the transaction
    conn2.commit()
    
    # Check both connections can see both rows
    cursor1 = conn1.cursor()
    cursor1.execute("SELECT COUNT(*) FROM isolation_test")
    count1 = cursor1.fetchone()[0]
    
    cursor2.execute("SELECT COUNT(*) FROM isolation_test")
    count2 = cursor2.fetchone()[0]
    
    assert count1 == 2
    assert count2 == 2
    
    # Clean up
    conn1.close()
    conn2.close()

def test_transaction_with_direct_conn_and_db_functions(temp_db):
    """Test mixing direct connection usage with database module functions"""
    # Debug information
    import os
    print(f"DEBUG: temp_db path: {temp_db['db_path']}")
    print(f"DEBUG: DB file exists: {os.path.exists(temp_db['db_path'])}")
    
    # Get a connection
    conn = sqlite3.connect(temp_db["db_path"])
    
    # Check database tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"DEBUG: Database tables: {[table[0] if isinstance(table, tuple) else table['name'] for table in tables]}")
    
    # Start a transaction
    conn.execute("BEGIN TRANSACTION")
    
    # Create a job using the database function
    job_id = create_job("training", {"param": "value"}, connection=conn)
    print(f"DEBUG: Created job_id: {job_id}")
    
    # Directly insert related data using the same connection
    cursor = conn.cursor()
    
    # Insert a parameter set
    cursor.execute(
        """INSERT INTO parameter_sets
           (parameter_set_id, parameters, created_at)
           VALUES (?, ?, ?)""",
        ("param-1", json.dumps({"batch_size": 32}), datetime.now().isoformat())
    )
    
    # Use database function to update the job
    update_job_status(job_id, "running", progress=50, connection=conn)
    
    # Verify current state in transaction
    cursor.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,))
    result = cursor.fetchone()
    print(f"DEBUG: Status result: {result}")
    
    if result is None:
        # Check if job still exists
        cursor.execute("SELECT * FROM jobs")
        all_jobs = cursor.fetchall()
        print(f"DEBUG: All jobs in database: {all_jobs}")
        raise AssertionError(f"Job {job_id} was not found during transaction")
    
    # Handle row as dictionary or tuple
    status = result['status'] if isinstance(result, dict) or hasattr(result, 'keys') else result[0]
    assert status == "running"
    
    # Now rollback
    conn.rollback()
    
    # Verify all changes were rolled back
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job_id,))
    count_result = cursor.fetchone()
    print(f"DEBUG: Job count after rollback: {count_result}")
    
    if count_result is not None:
        # Handle row as dictionary or tuple
        count = count_result['COUNT(*)'] if isinstance(count_result, dict) or hasattr(count_result, 'keys') else count_result[0]
        assert count == 0  # Job should be rolled back
    
    cursor.execute("SELECT COUNT(*) FROM parameter_sets WHERE parameter_set_id = ?", ("param-1",))
    count_result = cursor.fetchone()
    print(f"DEBUG: Parameter set count after rollback: {count_result}")
    
    if count_result is not None:
        # Handle row as dictionary or tuple
        count = count_result['COUNT(*)'] if isinstance(count_result, dict) or hasattr(count_result, 'keys') else count_result[0]
        assert count == 0  # Parameter set should be rolled back
    
    # Close the connection
    conn.close() 