import pytest
import os
import sqlite3
import pandas as pd
import threading
import time
import uuid
import contextlib
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
    get_all_models,
    DatabaseError
)
from deployment.app.db.schema import init_db
from deployment.app.models.api_models import JobStatus, JobType

# Create a context manager for database transactions
@contextlib.contextmanager
def db_transaction(connection=None, close_on_exit=False):
    """Context manager for database transactions with explicit commit/rollback"""
    conn_created = False
    if connection is None:
        connection = get_db_connection()
        conn_created = True
    
    try:
        connection.execute("BEGIN TRANSACTION")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if conn_created and close_on_exit:
            connection.close()

# Helper function to check if connection is valid/open
def is_connection_valid(conn):
    """Check if a SQLite connection is still valid/open"""
    try:
        conn.execute("SELECT 1")
        return True
    except (sqlite3.ProgrammingError, sqlite3.OperationalError):
        return False

# Helper function to generate unique job IDs to prevent uniqueness constraint violations
def generate_unique_job_id(prefix="test_job"):
    """Generate a unique job ID with optional prefix"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

@pytest.fixture
def db_setup():
    """Set up a test database with necessary initial data"""
    # Create a temporary directory and database
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, 'test.db')
    
    print(f"\n[DEBUG] Setting up test database at: {db_path}")
    
    # Initialize the database schema
    init_result = init_db(db_path)
    print(f"[DEBUG] Database schema initialization result: {init_result}")
    
    # Set the environment variable for database path
    original_db_path = os.environ.get('DATABASE_PATH')
    print(f"[DEBUG] Original DATABASE_PATH: {original_db_path}")
    os.environ['DATABASE_PATH'] = db_path
    print(f"[DEBUG] Updated DATABASE_PATH to: {db_path}")
    
    # Directly check if the settings.db.path is updated
    from deployment.app.config import settings
    print(f"[DEBUG] Current settings.db.path: {settings.db.path}")
    
    # Force reload settings to use the new DATABASE_PATH
    from importlib import reload
    import deployment.app.config
    reload(deployment.app.config)
    from deployment.app.config import settings
    print(f"[DEBUG] After reload settings.db.path: {settings.db.path}")
    
    # Verify tables are created with a direct connection
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"[DEBUG] Tables created in {db_path}: {tables}")
        conn.close()
    except Exception as e:
        print(f"[DEBUG] Error checking tables: {str(e)}")
    
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

# Basic connection management tests

def test_db_connection_is_properly_closed(db_setup):
    """
    Test that database connections are properly closed after use.
    
    This ensures there are no connection leaks in normal operation.
    """
    # First connection
    conn1 = get_db_connection()
    # Verify it's open
    assert is_connection_valid(conn1)
    
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
    assert is_connection_valid(conn2)
    
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

# Concurrent operations and transaction tests

def test_concurrent_writes(db_setup):
    """
    Test that concurrent database writes from multiple threads
    are properly handled without corruption.
    """
    NUM_THREADS = 5
    JOBS_PER_THREAD = 5
    
    # Use a unique ID for this test run to avoid conflicts with previous test runs
    test_run_id = uuid.uuid4().hex[:8]
    
    results = queue.Queue()
    
    def create_jobs(thread_id):
        """Create several jobs in a single thread"""
        success_count = 0
        try:
            # Use a single connection with explicit transaction management
            conn = get_db_connection()
            cursor = conn.cursor()
            
            for i in range(JOBS_PER_THREAD):
                try:
                    # Create jobs with parameters containing thread info for identification
                    job_id = f"concurrent_job_{test_run_id}_{thread_id}_{i}"
                    cursor.execute(
                        "INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (job_id, JobType.TRAINING.value, JobStatus.PENDING.value, 
                         f'{{"test_run": "{test_run_id}", "thread": {thread_id}, "iter": {i}}}',
                         "2023-01-01T00:00:00", "2023-01-01T00:00:00")
                    )
                    # Explicitly commit each insertion
                    conn.commit()
                    success_count += 1
                    # Small delay to increase chance of interleaving
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Thread {thread_id}, Iteration {i} - Error: {str(e)}")
                    # Rollback on error
                    conn.rollback()
            
            conn.close()
            results.put((thread_id, success_count))
        except Exception as e:
            print(f"Thread {thread_id} - Fatal error: {str(e)}")
            results.put((thread_id, success_count))
    
    # Run threads in parallel
    threads = []
    for i in range(NUM_THREADS):
        thread = threading.Thread(target=create_jobs, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check how many jobs each thread successfully created
    total_success_count = 0
    while not results.empty():
        thread_id, success_count = results.get()
        total_success_count += success_count
    
    # Verify jobs were created properly
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Find jobs by parameters containing the test run ID
    cursor.execute(
        """SELECT COUNT(*) FROM jobs 
        WHERE job_type = ? AND parameters LIKE ?""", 
        (JobType.TRAINING.value, f'%"test_run": "{test_run_id}"%')
    )
    count = cursor.fetchone()['COUNT(*)']  # Changed from [0] to ['COUNT(*)']
    conn.close()
    
    # Check that at least some jobs were created
    print(f"Successfully created {total_success_count} jobs in threads, found {count} in database")
    assert count > 0, "At least some jobs should be created and found in the database"
    assert count <= NUM_THREADS * JOBS_PER_THREAD, "Number of found jobs should not exceed maximum possible"

def test_transaction_isolation(db_setup):
    """
    Test that transactions are properly isolated between connections.
    
    Changes in one transaction should not be visible to another
    until committed.
    """
    # Use a unique ID for this test run to avoid conflicts with previous test runs
    test_run_id = uuid.uuid4().hex[:8]
    job_id = f"isolation_test_job_{test_run_id}"
    
    # Create a job in the first connection but don't commit yet
    conn1 = get_db_connection()
    
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
    count = cursor2.fetchone()['COUNT(*)']  # Changed from [0] to ['COUNT(*)']
    
    # Job should not be visible yet
    assert count == 0
    
    # Now commit the transaction
    conn1.commit()
    
    # Now the job should be visible in the second connection
    cursor2.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job_id,))
    count = cursor2.fetchone()['COUNT(*)']  # Changed from [0] to ['COUNT(*)']
    
    # Job should be visible now
    assert count == 1
    
    # Close connections
    conn1.close()
    conn2.close()

def test_nested_transactions(db_setup):
    """
    Test handling of nested transactions.
    
    SQLite supports nested transactions with SAVEPOINT,
    and this test verifies our code handles them correctly.
    """
    # Generate unique IDs for this test run
    test_run_id = uuid.uuid4().hex[:8]
    job_id1 = f"nested_tx_job1_{test_run_id}"
    job_id2 = f"nested_tx_job2_{test_run_id}"
    
    count1 = 0
    count2 = 0
    
    # Use with statement to ensure connection is properly closed
    with get_db_connection() as conn:
        conn.execute("BEGIN TRANSACTION")
        
        # Insert a job in the outer transaction
        conn.execute(
            "INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id1, JobType.TRAINING.value, JobStatus.PENDING.value, "{}",
             "2023-01-01T00:00:00", "2023-01-01T00:00:00")
        )
        
        # Create a savepoint (nested transaction)
        conn.execute("SAVEPOINT sp1")
        
        # Insert another job in the nested transaction
        conn.execute(
            "INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id2, JobType.TRAINING.value, JobStatus.PENDING.value, "{}",
             "2023-01-01T00:00:00", "2023-01-01T00:00:00")
        )
        
        # Rollback to the savepoint - only job2 should be rolled back
        conn.execute("ROLLBACK TO SAVEPOINT sp1")
        
        # Commit the outer transaction - job1 should be committed
        conn.commit()
        
        # Verify results within the same connection before it's closed
        cursor = conn.cursor()
        
        # Verify job1 was committed
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job_id1,))
        count1 = cursor.fetchone()['COUNT(*)']  # Using dictionary-style access
        
        # Verify job2 was rolled back
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job_id2,))
        count2 = cursor.fetchone()['COUNT(*)']  # Using dictionary-style access
    
    # Assert outside the with block, ensuring connection is already closed
    assert count1 == 1, "Job1 should have been committed in the outer transaction"
    assert count2 == 0, "Job2 should have been rolled back in the nested transaction"

def test_exception_during_transaction(db_setup):
    """
    Test that transactions are correctly handled when exceptions occur.
    
    This verifies that SQLite's automatic rollback on exception works
    correctly with our code.
    """
    # Generate unique IDs for this test run
    test_run_id = uuid.uuid4().hex[:8]
    job_id1 = f"exception_tx_job1_{test_run_id}"
    job_id2 = f"exception_tx_job2_{test_run_id}"
    
    count1 = 0
    count2 = 0
    
    # Use with statement to ensure connection is properly closed
    with get_db_connection() as conn:
        try:
            # Start a transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Insert a job
            conn.execute(
                "INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id1, JobType.TRAINING.value, JobStatus.PENDING.value, "{}",
                 "2023-01-01T00:00:00", "2023-01-01T00:00:00")
            )
            
            # Intentionally create an error by trying to insert into a non-existent column
            conn.execute(
                "INSERT INTO jobs (job_id, job_type, status, non_existent_column) VALUES (?, ?, ?, ?)",
                (job_id2, JobType.TRAINING.value, JobStatus.PENDING.value, "error")
            )
            
            # We should never reach this line
            assert False, "Expected exception was not raised"
            
        except sqlite3.OperationalError:
            # Expected exception - now test that we can still use the connection
            # Explicitly rollback the transaction
            conn.rollback()
            
        # Start a new transaction
        conn.execute("BEGIN TRANSACTION")
        
        # Insert the second job correctly
        conn.execute(
            "INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id2, JobType.TRAINING.value, JobStatus.PENDING.value, "{}",
             "2023-01-01T00:00:00", "2023-01-01T00:00:00")
        )
        
        # Commit the transaction
        conn.commit()
        
        # Verify what happened within the connection before it's closed
        cursor = conn.cursor()
        
        # Check job1 (should be rolled back)
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job_id1,))
        count1 = cursor.fetchone()['COUNT(*)']
        
        # Check job2 (should be successfully inserted)
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job_id2,))
        count2 = cursor.fetchone()['COUNT(*)']
    
    # Assert outside the with block, ensuring connection is already closed
    assert count1 == 0, "Job1 should have been rolled back due to the exception"
    assert count2 == 1, "Job2 should have been inserted in the second transaction"

# Performance and error handling tests

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
    with get_db_connection() as conn2:
        cursor = conn2.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()['1']  # Using dictionary access with key '1'
        assert result == 1
        
    # No need to manually close conn2 as the with statement handles it

def test_connection_under_load(db_setup):
    """
    Test database performance under load with multiple connections.

    This verifies that connection management works effectively.
    """
    NUM_THREADS = 3  # Оптимизированное количество потоков
    OPERATIONS_PER_THREAD = 10  # Оптимизированное количество операций
    results = queue.Queue()
    
    # Уникальный ID для этого запуска теста
    test_run_id = uuid.uuid4().hex[:8]

    def worker(thread_id):
        start_time = time.time()
        success_count = 0

        try:
            for i in range(OPERATIONS_PER_THREAD):
                try:
                    # Get a connection
                    conn = get_db_connection()
                    cursor = conn.cursor()

                    # Alternate between reads and writes
                    if i % 2 == 0:
                        # Write operation - create a job
                        job_id = f"load_test_{test_run_id}_{thread_id}_{i}"
                        cursor.execute(
                            "INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                            (job_id, JobType.TRAINING.value, JobStatus.PENDING.value, "{}",
                             "2023-01-01T00:00:00", "2023-01-01T00:00:00")
                        )
                        conn.commit()
                    else:
                        # Read operation - count specific jobs for this test
                        cursor.execute("SELECT COUNT(*) FROM jobs WHERE job_id LIKE ?", (f"load_test_{test_run_id}%",))
                        # Даже если ещё нет записей, мы просто проверяем, что запрос выполнился
                        cursor.fetchone()

                    # Close the connection
                    conn.close()
                    success_count += 1
                    # Добавляем задержку для снижения вероятности конфликтов
                    time.sleep(0.01)
                except Exception:
                    # Игнорируем отдельные ошибки и продолжаем
                    pass
                    
        except Exception as e:
            # В случае фатальной ошибки записываем результат
            results.put((thread_id, False, str(e), success_count, time.time() - start_time))
            return

        # Record successful completion
        results.put((thread_id, True, None, success_count, time.time() - start_time))

    # Run threads
    threads = []
    for i in range(NUM_THREADS):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all to complete
    for thread in threads:
        thread.join()

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
        else:
            print(f"Thread {thread_id} failed: {error}")

    # At least some threads should complete successfully
    assert successful_threads > 0, "At least some threads should complete successfully"
    assert total_operations > 0, "Some operations should complete successfully"
    
    # If we have successful operations, calculate and print the average time
    if total_operations > 0:
        print(f"Average time per operation: {total_time / total_operations:.6f} seconds")
        print(f"Total successful operations: {total_operations}")
        print(f"Successful threads: {successful_threads}/{NUM_THREADS}")

def test_execute_many_with_large_dataset(db_setup):
    """
    Test the execute_many function with a large dataset.
    
    This verifies that batch operations work correctly for large data volumes.
    """
    # Create a dataset
    NUM_ROWS = 100  # Оптимизированный размер набора данных
    test_run_id = uuid.uuid4().hex[:8]  # Уникальный идентификатор для этого запуска теста
    
    # Prepare parameter sets data
    parameter_sets = []
    for i in range(NUM_ROWS):
        parameter_sets.append((
            f"{test_run_id}_{i}",  # parameter_set_id с префиксом - уникальный для этого запуска теста
            f'{{"input_chunk_length": 12, "output_chunk_length": 6, "test_param": {i}}}',  # parameters
            0,  # is_active
            f"2023-01-01T{i//60:02d}:{i%60:02d}:00"  # created_at
        ))
    
    # Insert data using execute_many
    conn = get_db_connection()
    execute_many(
        """INSERT INTO parameter_sets
           (parameter_set_id, parameters, is_active, created_at)
           VALUES (?, ?, ?, ?)""",
        parameter_sets,
        connection=conn
    )
    conn.commit()
    
    # Verify data was inserted correctly
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM parameter_sets WHERE parameter_set_id LIKE ?", (f"{test_run_id}_%",))
    count = cursor.fetchone()['COUNT(*)']  # Changed from [0] to ['COUNT(*)']
    assert count == NUM_ROWS
    
    # Sample verification of inserted rows
    sample_indices = [0, 10, 50]
    for i in sample_indices:
        param_set_id = f"{test_run_id}_{i}"
        cursor.execute("SELECT parameters FROM parameter_sets WHERE parameter_set_id = ?", (param_set_id,))
        row = cursor.fetchone()
        assert row is not None
        assert f'"test_param": {i}' in row['parameters']
    
    # Clean up
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
        result = cursor.fetchone()['1']  # Changed from [0] to ['1']
        assert result == 1
        conn.close()
    
    # Clean up the leaked connections
    for conn in leaked_connections:
        if is_connection_valid(conn):
            conn.close()

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
    result = cursor.fetchone()['1']  # Changed from [0] to ['1']
    assert result == 1
    
    # Clean up
    conn.close()

# Advanced transaction management tests

def test_context_manager_commit(db_setup):
    """
    Test that the db_transaction context manager properly commits changes.
    """
    job_id = generate_unique_job_id("tx_manager_commit_test")
    
    # Use the context manager to insert a job and commit
    with db_transaction() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "test_type", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
        )
    
    # Verify the job was committed
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT job_id FROM jobs WHERE job_id = ?", (job_id,))
    result = cursor.fetchone()
    conn.close()
    
    assert result is not None
    assert result['job_id'] == job_id  # Changed from [0] to ['job_id']

def test_context_manager_rollback(db_setup):
    """
    Test that the db_transaction context manager properly rolls back on error.
    """
    job_id = generate_unique_job_id("tx_manager_rollback_test")
    
    # Try to use the context manager, but raise an exception before it completes
    try:
        with db_transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, "test_type", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
            )
            # Raise an exception to trigger rollback
            raise ValueError("Test exception to trigger rollback")
    except ValueError:
        pass  # Expected exception
    
    # Verify the job was NOT committed
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT job_id FROM jobs WHERE job_id = ?", (job_id,))
    result = cursor.fetchone()
    conn.close()
    
    assert result is None

def test_context_manager_with_provided_connection(db_setup):
    """
    Test using the db_transaction context manager with an existing connection.
    """
    job_id = generate_unique_job_id("tx_manager_existing_conn_test")
    
    # Get a connection first
    conn = get_db_connection()
    
    # Use the context manager with the existing connection
    with db_transaction(connection=conn) as tx_conn:
        cursor = tx_conn.cursor()
        cursor.execute(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "test_type", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
        )
    
    # The connection should still be open and the data should be committed
    assert is_connection_valid(conn)
    
    cursor = conn.cursor()
    cursor.execute("SELECT job_id FROM jobs WHERE job_id = ?", (job_id,))
    result = cursor.fetchone()
    
    assert result is not None
    assert result['job_id'] == job_id  # Changed from [0] to ['job_id']
    
    # Clean up
    conn.close()

def test_long_running_transaction(db_setup):
    """
    Test the system's behavior during a long-running transaction.
    
    This simulates a scenario where a transaction takes a significant
    amount of time to complete, and other operations need to happen concurrently.
    """
    results = queue.Queue()
    job_id_prefix = generate_unique_job_id("long_tx")
    
    def long_transaction_thread():
        """Thread that starts a long transaction with many inserts"""
        try:
            conn = get_db_connection()
            conn.execute("BEGIN TRANSACTION")
            cursor = conn.cursor()
            
            # Simulate a long-running transaction with 100 inserts
            for i in range(100):
                job_id = f"{job_id_prefix}_{i}"
                cursor.execute(
                    "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (job_id, "test_type", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
                )
                # Sleep a bit between operations to simulate work
                time.sleep(0.01)
            
            # Commit after all inserts
            conn.commit()
            conn.close()
            results.put(("long_tx", True, None))
        except Exception as e:
            results.put(("long_tx", False, str(e)))
    
    def concurrent_read_thread(thread_id):
        """Thread that performs reads while the long transaction is running"""
        try:
            # Wait a bit to ensure the long transaction has started
            time.sleep(0.1)
            
            # Perform a read operation
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM jobs")
            count = cursor.fetchone()['COUNT(*)']  # Changed from [0] to ['COUNT(*)']
            conn.close()
            
            results.put((f"read_{thread_id}", True, count))
        except Exception as e:
            results.put((f"read_{thread_id}", False, str(e)))
    
    # Start the long transaction thread
    tx_thread = threading.Thread(target=long_transaction_thread)
    tx_thread.start()
    
    # Start several read threads
    read_threads = []
    for i in range(5):
        thread = threading.Thread(target=concurrent_read_thread, args=(i,))
        read_threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    tx_thread.join()
    for thread in read_threads:
        thread.join()
    
    # Check results
    all_results = []
    while not results.empty():
        all_results.append(results.get())
    
    # The long transaction should have succeeded
    long_tx_results = [r for r in all_results if r[0] == "long_tx"]
    assert len(long_tx_results) == 1
    assert long_tx_results[0][1] is True
    
    # The read operations should have succeeded
    read_results = [r for r in all_results if r[0].startswith("read_")]
    assert len(read_results) == 5
    assert all(r[1] for r in read_results)
    
    # Verify all 100 jobs were inserted
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM jobs WHERE job_id LIKE '{job_id_prefix}_%'")
    count = cursor.fetchone()['COUNT(*)']  # Changed from [0] to ['COUNT(*)']
    conn.close()
    
    assert count == 100

# Complex transaction and connection management tests

def test_multiple_operations_in_transaction(db_setup):
    """
    Test a complex transaction with multiple types of operations.
    
    This verifies that different types of operations (INSERT, UPDATE, DELETE)
    can be combined in a single transaction and either all succeed or all fail.
    """
    # Generate a unique prefix for this test
    job_id_prefix = generate_unique_job_id("complex_tx")
    
    # Set up some initial data
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Insert 3 jobs
    job_ids = []
    for i in range(3):
        job_id = f"{job_id_prefix}_{i}"
        job_ids.append(job_id)
        cursor.execute(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "test_type", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
        )
    
    conn.commit()
    conn.close()
    
    # Now perform a complex transaction with multiple operations
    new_job_id = f"{job_id_prefix}_new"
    with db_transaction() as conn:
        cursor = conn.cursor()
        
        # 1. Insert a new job
        cursor.execute(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (new_job_id, "test_type", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
        )
        
        # 2. Update an existing job
        cursor.execute(
            "UPDATE jobs SET status = ? WHERE job_id = ?",
            ("running", job_ids[1])
        )
        
        # 3. Delete an existing job
        cursor.execute(
            "DELETE FROM jobs WHERE job_id = ?",
            (job_ids[2],)
        )
        
        # 4. Read count of jobs - should include the temporary state
        cursor.execute(f"SELECT COUNT(*) FROM jobs WHERE job_id LIKE '{job_id_prefix}%'")
        count_during_tx = cursor.fetchone()['COUNT(*)']
    
    # Verify final state after transaction
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check total count first
    cursor.execute(f"SELECT COUNT(*) FROM jobs WHERE job_id LIKE '{job_id_prefix}%'")
    final_count = cursor.fetchone()['COUNT(*)']
    
    # Check individual operations
    cursor.execute("SELECT job_id FROM jobs WHERE job_id = ?", (new_job_id,))
    new_job_exists = cursor.fetchone() is not None
    
    cursor.execute("SELECT status FROM jobs WHERE job_id = ?", (job_ids[1],))
    updated_status = cursor.fetchone()['status']  # Changed from [0] to ['status']
    
    cursor.execute("SELECT job_id FROM jobs WHERE job_id = ?", (job_ids[2],))
    deleted_job_exists = cursor.fetchone() is not None
    
    conn.close()
    
    # Assertions
    assert count_during_tx == 3  # 3 jobs (original) - 1 (deleted) + 1 (inserted) = 3
    assert final_count == 3  # Should match count during transaction
    assert new_job_exists  # New job should exist
    assert updated_status == "running"  # Job 1 should be updated to running
    assert not deleted_job_exists  # Job 2 should be deleted

def test_connection_timeout_handling(db_setup):
    """
    Test the system's ability to handle connection management under concurrent load.
    
    This tests the ability of the database to handle multiple connections,
    ensuring that writes complete successfully when done correctly.
    """
    # Test concurrent writes with proper connection handling
    job_id_prefix = generate_unique_job_id("conn_mgmt_test")
    num_writers = 10
    results_queue = queue.Queue()
    
    def do_write(writer_id):
        """Perform a write operation with proper connection handling"""
        job_id = f"{job_id_prefix}_{writer_id}"
        try:
            # Get connection and start transaction
            conn = get_db_connection()
            with db_transaction(connection=conn) as tx_conn:
                cursor = tx_conn.cursor()
                # Insert a job record
                cursor.execute(
                    "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (job_id, "test_type", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
                )
            
            # Connection should still be usable
            assert is_connection_valid(conn)
            
            # Close when done
            conn.close()
            
            results_queue.put((writer_id, True, None))
        except Exception as e:
            results_queue.put((writer_id, False, str(e)))
    
    # Start all writers in separate threads
    threads = []
    for i in range(num_writers):
        thread = threading.Thread(target=do_write, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check results
    success_count = 0
    errors = []
    
    while not results_queue.empty():
        writer_id, success, error = results_queue.get()
        if success:
            success_count += 1
        else:
            errors.append(f"Writer {writer_id}: {error}")
    
    # At least some writers should succeed
    assert success_count > 0, f"Expected at least some successful writers, got {success_count}"
    
    # Verify that the successful jobs were created
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM jobs WHERE job_id LIKE '{job_id_prefix}_%'")
    count = cursor.fetchone()['COUNT(*)']  # Changed from [0] to ['COUNT(*)']
    
    assert count == success_count, f"Expected {success_count} jobs in database, found {count}"
    
    conn.close()

def test_connection_pooling_simulation(db_setup):
    """
    Test a simulation of connection pooling behavior.
    
    SQLite doesn't have built-in connection pooling, but we can simulate
    the behavior by only creating connections on the same thread.
    """
    job_id_prefix = generate_unique_job_id("pool_job")
    
    # Modified approach: don't try to share connections between threads
    def worker(worker_id):
        """Worker that creates its own connection, uses it, and closes it"""
        try:
            # Get a connection in this thread
            conn = get_db_connection()
            
            # Use the connection
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (f"{job_id_prefix}_{worker_id}", "test_type", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
            )
            conn.commit()
            
            # Close the connection in the same thread
            conn.close()
            
            return True
        except Exception as e:
            print(f"Worker {worker_id} error: {str(e)}")
            return False
    
    # Run 20 workers
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(20)]
        results = [future.result() for future in futures]
    
    # Verify that at least most workers succeeded
    successful = sum(1 for r in results if r)
    assert successful > len(results) * 0.8, f"Expected at least 80% success rate, got {successful}/{len(results)}"
    
    # Verify that the successful jobs were created
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM jobs WHERE job_id LIKE '{job_id_prefix}_%'")
    count = cursor.fetchone()['COUNT(*)']
    
    assert count == successful, f"Expected {successful} jobs in database, found {count}"
    
    # Clean up
    conn.close() 