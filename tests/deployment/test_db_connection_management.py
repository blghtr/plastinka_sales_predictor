import pytest
import os
import sqlite3
import pandas as pd
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import queue
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from deployment.app.db.database import (
    get_db_connection,
    execute_query,
    execute_many,
    get_job,
    create_job,
    update_job_status,
    get_models_count,
    get_all_models
)
from deployment.app.db.schema import init_db
from deployment.app.models.api_models import JobStatus, JobType

@pytest.fixture
def db_setup():
    """Set up a test database with necessary initial data"""
    # Create a temporary directory and database
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, 'test.db')
    
    # Initialize the database schema
    init_db(db_path)
    
    # Set the environment variable for database path
    original_db_path = os.environ.get('DATABASE_PATH')
    os.environ['DATABASE_PATH'] = db_path
    
    # Yield fixture data
    yield {
        "temp_dir": temp_dir,
        "db_path": db_path,
        "original_db_path": original_db_path
    }
    
    # Teardown - restore original environment variable
    if original_db_path:
        os.environ['DATABASE_PATH'] = original_db_path
    else:
        os.environ.pop('DATABASE_PATH', None)
    
    # Clean up temporary directory
    temp_dir.cleanup()

def test_db_connection_is_properly_closed(db_setup):
    """
    Test that database connections are properly closed after use.
    
    This ensures there are no connection leaks in normal operation.
    """
    # First connection
    conn1 = get_db_connection()
    # Verify it's open
    assert not conn1.closed
    
    # Use the connection
    conn1.execute("SELECT 1")
    
    # Explicitly close
    conn1.close()
    
    # Verify it's closed
    with pytest.raises(sqlite3.ProgrammingError) as exc_info:
        conn1.execute("SELECT 1")
    
    assert "Cannot operate on a closed database" in str(exc_info.value)
    
    # Get a new connection
    conn2 = get_db_connection()
    
    # Verify it's a new connection
    assert conn2 != conn1
    
    # Verify it's open
    assert not conn2.closed
    
    # Close the second connection
    conn2.close()

def test_concurrent_connections(db_setup):
    """
    Test that multiple threads can get their own database connections.
    
    This verifies the thread safety of the connection management.
    """
    NUM_THREADS = 10
    connections = {}
    errors = queue.Queue()
    
    def thread_function(thread_id):
        try:
            # Get a connection for this thread
            conn = get_db_connection()
            # Store the connection for later verification
            connections[thread_id] = conn
            # Execute a simple query to verify it works
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version()")
            result = cursor.fetchone()
            assert result is not None
            # Sleep to simulate work
            time.sleep(0.1)
            # Close the connection
            conn.close()
        except Exception as e:
            errors.put((thread_id, str(e)))
    
    # Create threads
    threads = []
    for i in range(NUM_THREADS):
        thread = threading.Thread(target=thread_function, args=(i,))
        threads.append(thread)
    
    # Start threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check for errors
    if not errors.empty():
        error_messages = []
        while not errors.empty():
            thread_id, error = errors.get()
            error_messages.append(f"Thread {thread_id}: {error}")
        pytest.fail(f"Errors in threads: {'; '.join(error_messages)}")
    
    # Verify each thread got a connection
    assert len(connections) == NUM_THREADS
    
    # Verify connections are different objects
    connection_ids = [id(conn) for conn in connections.values()]
    assert len(set(connection_ids)) == NUM_THREADS

def test_concurrent_writes(db_setup):
    """
    Test that concurrent database writes from multiple threads
    are properly handled without corruption.
    """
    NUM_THREADS = 5
    JOBS_PER_THREAD = 5
    
    def create_jobs(thread_id):
        for i in range(JOBS_PER_THREAD):
            # Create a unique job ID for this thread and iteration
            job_id = f"job_thread_{thread_id}_iter_{i}"
            # Create the job
            create_job(JobType.TRAINING, {"thread": thread_id, "iter": i}, job_id=job_id)
            # Small delay to increase chance of interleaving
            time.sleep(0.01)
    
    # Run threads in parallel
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(create_jobs, i) for i in range(NUM_THREADS)]
        
        # Wait for all to complete
        for future in futures:
            future.result()
    
    # Verify all jobs were created properly
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM jobs")
    count = cursor.fetchone()[0]
    conn.close()
    
    # Check total job count
    assert count == NUM_THREADS * JOBS_PER_THREAD
    
    # Verify each job has correct parameters
    for thread_id in range(NUM_THREADS):
        for i in range(JOBS_PER_THREAD):
            job_id = f"job_thread_{thread_id}_iter_{i}"
            job = get_job(job_id)
            assert job is not None
            assert job["job_id"] == job_id
            
            # Parse parameters
            parameters = job["parameters"]
            assert parameters["thread"] == thread_id
            assert parameters["iter"] == i

def test_transaction_isolation(db_setup):
    """
    Test that transactions are properly isolated between connections.
    
    Changes in one transaction should not be visible to another
    until committed.
    """
    # Create a job in the first connection but don't commit yet
    conn1 = get_db_connection()
    job_id = "isolation_test_job"
    
    # Start a transaction
    conn1.execute("BEGIN TRANSACTION")
    conn1.execute(
        "INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, JobType.TRAINING.value, JobStatus.PENDING.value, "{}", 
         "2023-01-01T00:00:00", "2023-01-01T00:00:00")
    )
    
    # In a second connection, the job should not be visible
    conn2 = get_db_connection()
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job_id,))
    count = cursor2.fetchone()[0]
    
    # Job should not be visible yet
    assert count == 0
    
    # Now commit the transaction
    conn1.commit()
    
    # Now the job should be visible in the second connection
    cursor2.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job_id,))
    count = cursor2.fetchone()[0]
    
    # Job should be visible now
    assert count == 1
    
    # Close connections
    conn1.close()
    conn2.close()

def test_connection_error_recovery(db_setup):
    """
    Test that the system can recover from connection errors.
    
    After a connection error, a new connection should be established successfully.
    """
    # Create a connection
    conn1 = get_db_connection()
    
    # Simulate a connection error by closing it forcefully
    conn1.close()
    
    # Attempting to use the closed connection should fail
    with pytest.raises(sqlite3.ProgrammingError):
        conn1.execute("SELECT 1")
    
    # But getting a new connection should succeed
    conn2 = get_db_connection()
    cursor = conn2.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()[0]
    assert result == 1
    
    # Clean up
    conn2.close()

def test_connection_under_load(db_setup):
    """
    Test database performance under load with multiple connections.
    
    This verifies that connection pooling works effectively.
    """
    NUM_THREADS = 10
    OPERATIONS_PER_THREAD = 50
    results = queue.Queue()
    
    def worker(thread_id):
        start_time = time.time()
        success_count = 0
        
        try:
            for i in range(OPERATIONS_PER_THREAD):
                # Get a connection
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Alternate between reads and writes
                if i % 2 == 0:
                    # Write operation - create a job
                    job_id = f"load_test_{thread_id}_{i}"
                    cursor.execute(
                        "INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (job_id, JobType.TRAINING.value, JobStatus.PENDING.value, "{}", 
                         "2023-01-01T00:00:00", "2023-01-01T00:00:00")
                    )
                    conn.commit()
                else:
                    # Read operation - count jobs
                    cursor.execute("SELECT COUNT(*) FROM jobs")
                    count = cursor.fetchone()[0]
                    assert count > 0
                
                # Close the connection
                conn.close()
                success_count += 1
        except Exception as e:
            results.put((thread_id, False, str(e), success_count, time.time() - start_time))
            return
        
        # Record successful completion
        results.put((thread_id, True, None, success_count, time.time() - start_time))
    
    # Run threads in parallel
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(worker, i) for i in range(NUM_THREADS)]
        
        # Wait for all to complete
        for future in futures:
            future.result()
    
    # Analyze results
    successful_threads = 0
    total_operations = 0
    total_time = 0
    
    while not results.empty():
        thread_id, success, error, ops_count, elapsed = results.get()
        if success:
            successful_threads += 1
            total_operations += ops_count
            total_time += elapsed
    
    # All threads should complete successfully
    assert successful_threads == NUM_THREADS
    assert total_operations == NUM_THREADS * OPERATIONS_PER_THREAD
    
    # Calculate operations per second for informational purposes
    ops_per_second = total_operations / total_time if total_time > 0 else 0
    print(f"Performance: {ops_per_second:.2f} operations per second")

def test_execute_many_with_large_dataset(db_setup):
    """
    Test the execute_many function with a large dataset.
    
    This verifies that batch operations work correctly for large data volumes.
    """
    # Create a large dataset
    NUM_ROWS = 1000
    
    # Prepare parameter sets data
    parameter_sets = []
    for i in range(NUM_ROWS):
        parameter_sets.append((
            i + 1,  # parameter_set_id
            f'{{"input_chunk_length": 12, "output_chunk_length": 6, "test_param": {i}}}',  # parameters
            0,  # is_active
            f"2023-01-01T{i//60:02d}:{i%60:02d}:00",  # created_at
            f"2023-01-01T{i//60:02d}:{i%60:02d}:00"   # updated_at
        ))
    
    # Insert data using execute_many
    conn = get_db_connection()
    execute_many(
        conn,
        """INSERT INTO parameter_sets 
           (parameter_set_id, parameters, is_active, created_at, updated_at) 
           VALUES (?, ?, ?, ?, ?)""",
        parameter_sets
    )
    conn.commit()
    
    # Verify data was inserted correctly
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM parameter_sets")
    count = cursor.fetchone()[0]
    assert count == NUM_ROWS
    
    # Verify some random rows
    for i in [0, 10, 100, 500, 999]:
        cursor.execute("SELECT parameters FROM parameter_sets WHERE parameter_set_id = ?", (i + 1,))
        row = cursor.fetchone()
        assert row is not None
        assert f'"test_param": {i}' in row[0]
    
    conn.close()

def test_connection_leak_recovery(db_setup):
    """
    Test that the system can recover from connection leaks.
    
    This simulates a scenario where connections aren't properly closed
    and verifies the system can still operate correctly.
    """
    # Create several connections without closing them (simulating leaks)
    leaked_connections = []
    for _ in range(10):
        leaked_connections.append(get_db_connection())
    
    # Verify we can still get new connections that work
    for _ in range(5):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()[0]
        assert result == 1
        conn.close()
    
    # Finally, clean up leaked connections
    for conn in leaked_connections:
        if not conn.closed:
            conn.close()

def test_nested_transactions(db_setup):
    """
    Test handling of nested transactions.
    
    SQLite supports nested transactions using savepoints.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Start an outer transaction
    conn.execute("BEGIN TRANSACTION")
    
    # Insert a job
    job1_id = "nested_tx_job1"
    cursor.execute(
        "INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (job1_id, JobType.TRAINING.value, JobStatus.PENDING.value, "{}", 
         "2023-01-01T00:00:00", "2023-01-01T00:00:00")
    )
    
    # Create a savepoint for nested transaction
    conn.execute("SAVEPOINT nested1")
    
    # Insert another job
    job2_id = "nested_tx_job2"
    cursor.execute(
        "INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (job2_id, JobType.TRAINING.value, JobStatus.PENDING.value, "{}", 
         "2023-01-01T00:00:00", "2023-01-01T00:00:00")
    )
    
    # Rollback to the savepoint (should undo job2 but keep job1)
    conn.execute("ROLLBACK TO SAVEPOINT nested1")
    
    # Commit the outer transaction
    conn.commit()
    
    # Verify job1 exists
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job1_id,))
    count1 = cursor.fetchone()[0]
    assert count1 == 1
    
    # Verify job2 does not exist
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job2_id,))
    count2 = cursor.fetchone()[0]
    assert count2 == 0
    
    conn.close()

def test_exception_during_transaction(db_setup):
    """
    Test that transactions are properly rolled back when an exception occurs.
    
    Changes should not be committed if an exception is raised during a transaction.
    """
    # Unique job ID for this test
    job_id = "exception_test_job"
    
    try:
        # Start a connection and transaction
        conn = get_db_connection()
        conn.execute("BEGIN TRANSACTION")
        
        # Insert a valid job
        conn.execute(
            "INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, JobType.TRAINING.value, JobStatus.PENDING.value, "{}", 
             "2023-01-01T00:00:00", "2023-01-01T00:00:00")
        )
        
        # Try to insert an invalid job (without job_id, which should be NOT NULL)
        # This should raise an exception
        conn.execute(
            "INSERT INTO jobs (job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (JobType.TRAINING.value, JobStatus.PENDING.value, "{}", 
             "2023-01-01T00:00:00", "2023-01-01T00:00:00")
        )
        
        # This line should never be reached
        conn.commit()
    except sqlite3.IntegrityError:
        # Expected exception
        conn.rollback()
    finally:
        conn.close()
    
    # Verify the first job was not inserted due to rollback
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job_id,))
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count == 0  # The transaction should have been rolled back

def test_connection_cleanup_on_exception(db_setup):
    """
    Test that database connections are properly cleaned up even when exceptions occur.
    
    This simulates error scenarios and verifies connections are still managed correctly.
    """
    # Create a function that tries to use a connection but raises an exception
    def function_with_exception():
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            # Simulate an exception
            raise ValueError("Simulated exception")
        finally:
            # Even with an exception, we should close the connection
            conn.close()
    
    # Call the function, catching the exception
    with pytest.raises(ValueError) as exc_info:
        function_with_exception()
    
    assert "Simulated exception" in str(exc_info.value)
    
    # After the exception, we should still be able to get a new connection
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()[0]
    assert result == 1
    conn.close() 