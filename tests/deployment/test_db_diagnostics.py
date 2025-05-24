"""
Database Connection Diagnostics

This module contains tests to diagnose issues with database connections.
"""

import pytest
import os
import sqlite3
import uuid
import importlib
import tempfile
import traceback
from pathlib import Path

from deployment.app.db.database import get_db_connection
from deployment.app.db.schema import init_db

@pytest.fixture
def db_setup():
    """Set up a test database for diagnostics."""
    # Create a temporary directory and database
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, 'test.db')
    
    print(f"\n[DIAG] Creating test database at: {db_path}")
    
    # Initialize the database schema
    init_db(db_path)
    
    # Set the environment variable for database path
    original_db_path = os.environ.get('DATABASE_PATH')
    os.environ['DATABASE_PATH'] = db_path
    
    # Force reload modules to use the new DATABASE_PATH
    import deployment.app.config
    importlib.reload(deployment.app.config)
    import deployment.app.db.database
    importlib.reload(deployment.app.db.database)
    
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
    
    # Explicitly close any connections to the database before cleanup
    try:
        # Force garbage collection to close any lingering connections
        import gc
        gc.collect()
        
        # Manually try to acquire and close a connection to ensure it's released
        conn = sqlite3.connect(db_path)
        conn.close()
    except Exception as e:
        print(f"[DIAG] Connection cleanup error: {e}")
    
    # Clean up temporary directory with error handling
    try:
        temp_dir.cleanup()
        print("[DIAG] Temp directory cleaned up successfully")
    except Exception as e:
        print(f"[DIAG] Failed to clean up temp directory: {str(e)}")
        traceback.print_exc()

def test_connection_reuse(db_setup):
    """Test if get_db_connection creates new connections correctly."""
    # Get two connections in the same thread
    conn1 = get_db_connection()
    conn2 = get_db_connection()
    
    # Print diagnostics
    print(f"[DIAG] Connection 1 id: {id(conn1)}, type: {type(conn1)}")
    print(f"[DIAG] Connection 2 id: {id(conn2)}, type: {type(conn2)}")
    print(f"[DIAG] Connections same object: {conn1 is conn2}")
    
    # Close connections
    conn1.close()
    if conn1 is not conn2:  # Only close if they're different objects
        conn2.close()
    
    # The current implementation creates new connections each time rather than reusing them
    # This is a design choice reflected in the implementation
    assert conn1 is not conn2, "get_db_connection should create unique connections to ensure isolation"

def test_row_factory(db_setup):
    """Test how cursor results are accessed."""
    # Get a connection
    conn = get_db_connection()
    
    # Create a test table
    conn.execute("CREATE TABLE test_access (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO test_access VALUES (1, 'test')")
    conn.commit()
    
    # Query the data
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM test_access")
    row = cursor.fetchone()
    
    # Print diagnostics
    print(f"[DIAG] Row type: {type(row)}")
    print(f"[DIAG] Row dir: {dir(row)}")
    print(f"[DIAG] Row has keys method: {hasattr(row, 'keys')}")
    
    # Try index-based access
    try:
        id_value = row[0]
        print(f"[DIAG] Can access by index: True, id value: {id_value}")
        index_access_works = True
    except Exception as e:
        print(f"[DIAG] Cannot access by index: {type(e)}, {str(e)}")
        index_access_works = False
    
    # Try name-based access
    try:
        name_value = row['name']
        print(f"[DIAG] Can access by name: True, name value: {name_value}")
        name_access_works = True
    except Exception as e:
        print(f"[DIAG] Cannot access by name: {type(e)}, {str(e)}")
        name_access_works = False
    
    # Test a COUNT query
    cursor.execute("SELECT COUNT(*) as count FROM test_access")
    count_row = cursor.fetchone()
    
    print(f"[DIAG] Count row type: {type(count_row)}")
    print(f"[DIAG] Count row repr: {repr(count_row)}")
    
    # Try index-based access for COUNT
    try:
        count_value = count_row[0]
        print(f"[DIAG] Can access COUNT by index: True, value: {count_value}")
    except Exception as e:
        print(f"[DIAG] Cannot access COUNT by index: {type(e)}, {str(e)}")
    
    # Try name-based access for COUNT
    try:
        count_value = count_row['count']
        print(f"[DIAG] Can access COUNT by name: True, value: {count_value}")
    except Exception as e:
        print(f"[DIAG] Cannot access COUNT by name: {type(e)}, {str(e)}")
    
    # Close connection
    conn.close()
    
    # Assert based on what we found
    if not index_access_works and name_access_works:
        print("[DIAG] Row factory is configured for name-based access only")
    elif index_access_works and name_access_works:
        print("[DIAG] Row factory is configured for both index and name-based access")
    else:
        print("[DIAG] Row factory configuration is unexpected")

def test_create_job_parameters(db_setup):
    """Test the parameters expected by create_job function."""
    from deployment.app.db.database import create_job
    from deployment.app.models.api_models import JobType
    
    # Get signature info
    import inspect
    signature = inspect.signature(create_job)
    
    print(f"[DIAG] create_job signature: {signature}")
    
    for name, param in signature.parameters.items():
        print(f"[DIAG] Parameter '{name}': {param.annotation}, default={param.default}")
    
    # Get a connection
    conn = get_db_connection()
    
    # Generate a unique job ID
    job_id = f"test_job_{uuid.uuid4().hex[:8]}"
    
    # Create a job with proper parameters
    try:
        print("[DIAG] Calling create_job with correct parameter types")
        job_result = create_job(job_id, JobType.TRAINING, {"test": "params"}, connection=conn)
        print(f"[DIAG] create_job result: {job_result}")
    except Exception as e:
        print(f"[DIAG] create_job exception: {type(e)}, {str(e)}")
        traceback.print_exc()
    
    # Test with dictionary instead of connection
    try:
        print("[DIAG] Calling create_job with dictionary instead of connection")
        fake_conn = {"test": "connection"}
        job_result = create_job(f"{job_id}_2", JobType.TRAINING, {"test": "params"}, connection=fake_conn)
        print(f"[DIAG] create_job result (should not reach here): {job_result}")
    except Exception as e:
        print(f"[DIAG] Expected exception with dict as connection: {type(e)}, {str(e)}")
    
    # Close connection
    conn.close() 