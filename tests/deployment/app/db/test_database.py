"""
Tests for core functionality in the database module.

This module contains comprehensive tests for the database.py module,
covering the main database operations including:

1. Connection management
2. Query execution
3. Job management
4. Parameter set management
5. Model management
6. Database result handling

Each test focuses on verifying the correct behavior of a specific database function
in isolation, using fixtures to set up the appropriate test environment.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from deployment.app.db.database import (
    DatabaseError,
    create_data_upload_result,
    create_job,
    create_model_record,
    create_or_get_config,
    create_prediction_result,
    create_processing_run,
    create_training_result,
    dict_factory,
    execute_many,
    execute_query,
    generate_id,
    get_active_config,
    get_active_model,
    get_db_connection,
    get_effective_config,
    get_job,
    list_jobs,
    set_config_active,
    set_model_active,
    update_job_status,
    update_processing_run,
)

# =============================================
# Используем фикстуры из conftest.py
# =============================================

# @pytest.fixture
# def temp_db():
#     """Create a temporary SQLite database for testing"""
#     # Create temporary file
#     db_fd, db_path = os.path.join(os.path.dirname(__file__), "test_temp.db")

#     # Initialize schema
#     conn = sqlite3.connect(db_path)
#     conn.executescript(SCHEMA_SQL)
#     conn.commit()
#     conn.close()

#     # Save original DB_PATH
#     original_db_path = os.environ.get('DATABASE_PATH')

#     # Set environment variable to point to test DB
#     os.environ['DATABASE_PATH'] = db_path

#     # Yield the database path and connection to the test
#     yield {"db_path": db_path}

#     # Restore original DB_PATH
#     if original_db_path:
#         os.environ['DATABASE_PATH'] = original_db_path
#     else:
#         os.environ.pop('DATABASE_PATH', None)

#     # Close and remove the temporary file
#     try:
#         os.unlink(db_path)
#     except:
#         pass

# @pytest.fixture
# def in_memory_db():
#     """Create an in-memory SQLite database for testing"""
#     # Create in-memory database
#     conn = sqlite3.connect(':memory:')
#     conn.executescript(SCHEMA_SQL)
#     conn.commit()

#     # Patch get_db_connection to return our test connection
#     with patch('deployment.app.db.database.get_db_connection', return_value=conn):
#         yield {"conn": conn}

#     # Close connection
#     conn.close()

# @pytest.fixture
# def sample_job_data():
#     """Sample job data for testing"""
#     return {
#         "job_id": str(uuid.uuid4()),
#         "job_type": "training",
#         "parameters": {
#             "batch_size": 32,
#             "learning_rate": 0.001
#         }
#     }

# @pytest.fixture
# def sample_model_data():
#     """Sample model data for testing"""
#     return {
#         "model_id": str(uuid.uuid4()),
#         "job_id": str(uuid.uuid4()),
#         "model_path": "/path/to/model.onnx",
#         "created_at": datetime.now(),
#         "metadata": {
#             "framework": "pytorch",
#             "version": "1.9.0"
#         }
#     }

# @pytest.fixture
# def sample_config():
#     """Sample parameter set for testing"""
#     return {
#         "input_chunk_length": 12,
#         "output_chunk_length": 6,
#         "hidden_size": 64,
#         "lstm_layers": 2,
#         "dropout": 0.2,
#         "batch_size": 32,
#         "max_epochs": 10,
#         "learning_rate": 0.001
#     }

# =============================================
# Tests for core database functions
# =============================================


def test_get_db_connection(temp_db):
    """Test that get_db_connection returns a valid connection"""
    # Use the actual function to get a connection
    conn = get_db_connection()

    # Verify it's a valid SQLite connection
    assert isinstance(conn, sqlite3.Connection)

    # Verify row_factory is set to sqlite3.Row
    assert conn.row_factory == dict_factory

    # Test that we can execute a basic query
    cursor = conn.cursor()
    cursor.execute("SELECT sqlite_version()")
    version_row = cursor.fetchone()
    assert version_row is not None
    version = version_row["sqlite_version()"]
    assert version is not None

    # Close the connection
    conn.close()


def test_get_db_connection_error():
    """Test that get_db_connection raises DatabaseError when connection fails"""
    # Patch Path.mkdir to raise an exception
    with patch("pathlib.Path.mkdir", side_effect=Exception("Forced error")):
        # Verify that the function raises DatabaseError
        with pytest.raises(DatabaseError) as exc_info:
            get_db_connection()

        # Verify error message
        assert "Database connection failed" in str(exc_info.value)
        assert exc_info.value.original_error is not None


def test_dict_factory():
    """Test that dict_factory correctly converts rows to dictionaries"""
    # Create a mock cursor
    mock_cursor = MagicMock()
    mock_cursor.description = [
        ("id", None, None, None, None, None, None),
        ("name", None, None, None, None, None, None),
    ]

    # Create a mock row
    row = (1, "test")

    # Call dict_factory
    result = dict_factory(mock_cursor, row)

    # Verify result is a dictionary with correct keys and values
    assert isinstance(result, dict)
    assert result["id"] == 1
    assert result["name"] == "test"


def test_execute_query_select(in_memory_db):
    """Test execute_query with a SELECT statement"""
    conn = in_memory_db["conn"]

    # Add some test data
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
    )
    cursor.execute("INSERT INTO test_table (name) VALUES (?)", ("test_name",))
    conn.commit()

    # Test execute_query with SELECT
    result = execute_query(
        "SELECT * FROM test_table WHERE name = ?", ("test_name",), connection=conn
    )

    # Verify result
    assert result is not None
    assert result["name"] == "test_name"

    # Test with fetchall=True
    results = execute_query("SELECT * FROM test_table", fetchall=True, connection=conn)
    assert len(results) == 1
    assert results[0]["name"] == "test_name"


def test_execute_query_insert(in_memory_db):
    """Test execute_query with an INSERT statement"""
    conn = in_memory_db["conn"]

    # Create test table
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
    )

    # Use execute_query for INSERT
    execute_query(
        "INSERT INTO test_table (name) VALUES (?)", ("new_record",), connection=conn
    )

    # Verify the record was inserted
    cursor.execute("SELECT * FROM test_table WHERE name = ?", ("new_record",))
    result = cursor.fetchone()
    assert result is not None
    assert result["name"] == "new_record"  # Access by column name due to dict_factory


def test_execute_query_error(in_memory_db):
    """Test that execute_query raises DatabaseError on SQL errors"""
    conn = in_memory_db["conn"]

    # Execute a query with syntax error
    with pytest.raises(DatabaseError) as exc_info:
        execute_query("SELECT * FROM nonexistent_table", connection=conn)

    # Verify error details
    assert "Database operation failed" in str(exc_info.value)
    assert exc_info.value.query == "SELECT * FROM nonexistent_table"
    assert exc_info.value.params == ()
    assert exc_info.value.original_error is not None


def test_execute_many(in_memory_db):
    """Test execute_many with multiple parameter sets"""
    conn = in_memory_db["conn"]

    # Create test table
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
    )

    # Define parameter sets
    params_list = [("name1",), ("name2",), ("name3",)]

    # Use execute_many
    execute_many(
        "INSERT INTO test_table (name) VALUES (?)", params_list, connection=conn
    )

    # Verify all records were inserted
    cursor.execute("SELECT COUNT(*) FROM test_table")
    count_result = cursor.fetchone()
    assert count_result is not None
    count = count_result["COUNT(*)"]  # Access by column name (alias)
    assert count == len(params_list)

    # Verify specific records
    cursor.execute("SELECT name FROM test_table ORDER BY id")
    names = [row["name"] for row in cursor.fetchall()]
    assert names == ["name1", "name2", "name3"]


def test_execute_many_error(in_memory_db):
    """Test that execute_many raises DatabaseError on SQL errors"""
    conn = in_memory_db["conn"]

    # Try to execute on a non-existent table
    params_list = [("name1",), ("name2",)]

    with pytest.raises(DatabaseError) as exc_info:
        execute_many(
            "INSERT INTO nonexistent_table (name) VALUES (?)",
            params_list,
            connection=conn,
        )

    # Verify error details
    assert "Batch database operation failed" in str(exc_info.value)
    assert exc_info.value.query == "INSERT INTO nonexistent_table (name) VALUES (?)"
    assert exc_info.value.original_error is not None


def test_execute_many_empty_params(in_memory_db):
    """Test that execute_many handles empty params lists gracefully"""
    conn = in_memory_db["conn"]

    # Call with empty params list - should return without error
    result = execute_many(
        "INSERT INTO test_table (name) VALUES (?)", [], connection=conn
    )
    assert result is None


# =============================================
# Tests for job-related functions
# =============================================


def test_generate_id():
    """Test that generate_id produces unique IDs"""
    # Generate multiple IDs and verify uniqueness
    ids = [generate_id() for _ in range(10)]
    unique_ids = set(ids)

    assert len(ids) == len(unique_ids)

    # Verify UUID format
    for id in ids:
        assert len(id) == 36  # UUID string length
        assert isinstance(uuid.UUID(id), uuid.UUID)  # Valid UUID format


def test_create_job(in_memory_db):
    """Test creating a job record"""
    conn = in_memory_db["conn"]

    # Create a job
    job_type = "training"
    parameters = {"batch_size": 32, "epochs": 10}
    job_id = create_job(job_type, parameters, connection=conn)

    # Verify job_id is returned and valid
    assert job_id is not None
    assert isinstance(job_id, str)
    assert len(job_id) == 36  # UUID string length

    # Verify job was created in database
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    job = cursor.fetchone()

    assert job is not None
    assert job["job_type"] == job_type
    assert job["status"] == "pending"
    assert job["progress"] == 0

    # Verify parameters were stored as JSON
    stored_params = json.loads(job["parameters"])
    assert stored_params == parameters


def test_update_job_status(in_memory_db, sample_job_data):
    """Test updating job status"""
    conn = in_memory_db["conn"]

    # Create a job first
    job_id = create_job(
        sample_job_data["job_type"], sample_job_data["parameters"], connection=conn
    )

    # Update job status
    update_job_status(
        job_id=job_id,
        status="running",
        progress=50,
        status_message="Halfway there",
        connection=conn,
    )

    # Verify job was updated
    job = get_job(job_id, connection=conn)
    assert job["status"] == "running"
    assert job["progress"] == 50

    # Verify status history was recorded
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM job_status_history WHERE job_id = ?", (job_id,))
    history = cursor.fetchone()

    assert history is not None
    assert history["status"] == "running"
    assert history["progress"] == 50
    assert history["status_message"] == "Halfway there"


def test_get_job_nonexistent(in_memory_db):
    """Test getting a non-existent job"""
    conn = in_memory_db["conn"]

    # Try to get a non-existent job
    non_existent_id = str(uuid.uuid4())
    result = get_job(non_existent_id, connection=conn)

    # Should return None
    assert result is None


def test_list_jobs(in_memory_db):
    """Test listing jobs with filters"""
    conn = in_memory_db["conn"]

    # Create jobs of different types and statuses
    job_1 = create_job("training", {"param": "value1"}, connection=conn)
    job_2 = create_job("prediction", {"param": "value2"}, connection=conn)
    job_3 = create_job("training", {"param": "value3"}, connection=conn)

    # Update statuses
    update_job_status(job_1, "completed", connection=conn)
    update_job_status(job_2, "running", connection=conn)
    update_job_status(job_3, "running", connection=conn)

    # List all jobs
    all_jobs = list_jobs(connection=conn)
    assert len(all_jobs) == 3

    # List by job_type
    training_jobs = list_jobs(job_type="training", connection=conn)
    assert len(training_jobs) == 2
    assert all(j["job_type"] == "training" for j in training_jobs)

    # List by status
    running_jobs = list_jobs(status="running", connection=conn)
    assert len(running_jobs) == 2
    assert all(j["status"] == "running" for j in running_jobs)

    # List with both filters
    filtered_jobs = list_jobs(job_type="training", status="running", connection=conn)
    assert len(filtered_jobs) == 1
    assert filtered_jobs[0]["job_id"] == job_3


# =============================================
# Tests for model and parameter set management
# =============================================


def test_create_model_record(in_memory_db, sample_model_data):
    """Test creating a model record"""
    conn = in_memory_db["conn"]

    # Create a job first (required by FK constraint)
    job_id = create_job("training", {}, connection=conn)

    # Create model record
    model_id = sample_model_data["model_id"]
    model_path = sample_model_data["model_path"]
    created_at = sample_model_data["created_at"]
    metadata = sample_model_data["metadata"]

    # Use the connection directly instead of patching
    create_model_record(
        model_id=model_id,
        job_id=job_id,  # Use the created job_id instead of sample_model_data["job_id"]
        model_path=model_path,
        created_at=created_at,
        metadata=metadata,
        is_active=True,
        connection=conn,  # Pass the connection explicitly
    )

    # Verify model was created
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
    model = cursor.fetchone()

    assert model is not None
    assert model["model_id"] == model_id
    assert model["job_id"] == job_id
    assert model["model_path"] == model_path
    assert model["is_active"] == 1

    # Verify metadata was stored correctly
    stored_metadata = json.loads(model["metadata"])
    assert stored_metadata == metadata


def test_get_active_model(in_memory_db, sample_model_data):
    """Test getting the active model"""
    conn = in_memory_db["conn"]

    # Create a job first
    job_id = create_job("training", {}, connection=conn)

    # Create a model and set it as active
    model_id = sample_model_data["model_id"]
    model_path = sample_model_data["model_path"]
    created_at = sample_model_data["created_at"]
    metadata = sample_model_data["metadata"]

    # Insert directly to avoid complicating the test with multiple patches
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO models
           (model_id, job_id, model_path, created_at, metadata, is_active)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (model_id, job_id, model_path, created_at.isoformat(), json.dumps(metadata), 1),
    )
    conn.commit()

    # Get active model
    active_model = get_active_model(connection=conn)

    # Verify correct model was returned
    assert active_model is not None
    assert active_model["model_id"] == model_id
    assert active_model["model_path"] == model_path
    assert active_model["metadata"] == metadata


def test_set_model_active(in_memory_db):
    """Test setting a model as active"""
    conn = in_memory_db["conn"]

    # Create a job
    job_id = create_job("training", {}, connection=conn)

    # Create two models
    cursor = conn.cursor()
    model_id_1 = str(uuid.uuid4())
    model_id_2 = str(uuid.uuid4())
    now = datetime.now().isoformat()

    cursor.execute(
        "INSERT INTO models (model_id, job_id, model_path, created_at, is_active) VALUES (?, ?, ?, ?, ?)",
        (model_id_1, job_id, "/path/to/model1.onnx", now, 0),
    )
    cursor.execute(
        "INSERT INTO models (model_id, job_id, model_path, created_at, is_active) VALUES (?, ?, ?, ?, ?)",
        (model_id_2, job_id, "/path/to/model2.onnx", now, 0),
    )
    conn.commit()

    # Set the first model as active
    result = set_model_active(model_id_1, connection=conn)
    assert result is True

    # Verify first model is active
    cursor.execute("SELECT is_active FROM models WHERE model_id = ?", (model_id_1,))
    is_active_1_row = cursor.fetchone()
    # Handle both sqlite.Row and tuple
    is_active_1 = (
        is_active_1_row["is_active"]
        if hasattr(is_active_1_row, "keys")
        else is_active_1_row[0]
    )
    assert is_active_1 == 1

    # Verify second model is inactive
    cursor.execute("SELECT is_active FROM models WHERE model_id = ?", (model_id_2,))
    is_active_2_row = cursor.fetchone()
    # Handle both sqlite.Row and tuple
    is_active_2 = (
        is_active_2_row["is_active"]
        if hasattr(is_active_2_row, "keys")
        else is_active_2_row[0]
    )
    assert is_active_2 == 0

    # Set the second model as active
    result = set_model_active(model_id_2, connection=conn)
    assert result is True

    # Verify first model is now inactive
    cursor.execute("SELECT is_active FROM models WHERE model_id = ?", (model_id_1,))
    is_active_1_row = cursor.fetchone()
    # Handle both sqlite.Row and tuple
    is_active_1 = (
        is_active_1_row["is_active"]
        if hasattr(is_active_1_row, "keys")
        else is_active_1_row[0]
    )
    assert is_active_1 == 0

    # Verify second model is now active
    cursor.execute("SELECT is_active FROM models WHERE model_id = ?", (model_id_2,))
    is_active_2_row = cursor.fetchone()
    # Handle both sqlite.Row and tuple
    is_active_2 = (
        is_active_2_row["is_active"]
        if hasattr(is_active_2_row, "keys")
        else is_active_2_row[0]
    )
    assert is_active_2 == 1


def test_create_or_get_config(in_memory_db, sample_config):
    """Test creating a config record"""
    # Test creating a new config
    config_id = create_or_get_config(
        sample_config, is_active=True, connection=in_memory_db["conn"]
    )
    assert config_id is not None

    retrieved_config = execute_query(
        "SELECT * FROM configs WHERE config_id = ?",
        (config_id,),
        connection=in_memory_db["conn"],
    )
    assert retrieved_config is not None
    assert json.loads(retrieved_config["config"]) == sample_config
    assert retrieved_config["is_active"] == 1

    # Test idempotency: calling with same config should return same ID
    same_config_id = create_or_get_config(
        sample_config, connection=in_memory_db["conn"]
    )
    assert same_config_id == config_id

    # Test creating a new config with is_active=True deactivates others
    new_config_data = {
        "input_chunk_length": 24,
        "output_chunk_length": 12,
        "hidden_size": 128,
        "lstm_layers": 3,
        "dropout": 0.3,
        "batch_size": 64,
        "max_epochs": 20,
        "learning_rate": 0.0005,
    }
    new_config_id = create_or_get_config(
        new_config_data, is_active=True, connection=in_memory_db["conn"]
    )

    old_config = execute_query(
        "SELECT is_active FROM configs WHERE config_id = ?",
        (config_id,),
        connection=in_memory_db["conn"],
    )
    assert old_config["is_active"] == 0  # Old config should be inactive

    new_config = execute_query(
        "SELECT is_active FROM configs WHERE config_id = ?",
        (new_config_id,),
        connection=in_memory_db["conn"],
    )
    assert new_config["is_active"] == 1  # New config should be active


def test_get_active_config(in_memory_db, sample_config):
    """Test retrieving the active config"""
    # Create a config and set it active
    config_id = create_or_get_config(
        sample_config, is_active=True, connection=in_memory_db["conn"]
    )

    active_config = get_active_config(connection=in_memory_db["conn"])
    assert active_config is not None
    assert active_config["config_id"] == config_id
    assert active_config["config"] == sample_config

    # Test no active config
    execute_query("UPDATE configs SET is_active = 0", connection=in_memory_db["conn"])
    assert get_active_config(connection=in_memory_db["conn"]) is None


def test_set_config_active(in_memory_db, sample_config):
    """Test setting a config as active"""
    # Create two configs
    config_id1 = create_or_get_config(
        sample_config, is_active=False, connection=in_memory_db["conn"]
    )
    config_id2 = create_or_get_config(
        {"key": "value"}, is_active=False, connection=in_memory_db["conn"]
    )

    # Set config1 active
    success = set_config_active(config_id1, connection=in_memory_db["conn"])
    assert success is True

    cfg1 = execute_query(
        "SELECT is_active FROM configs WHERE config_id = ?",
        (config_id1,),
        connection=in_memory_db["conn"],
    )
    cfg2 = execute_query(
        "SELECT is_active FROM configs WHERE config_id = ?",
        (config_id2,),
        connection=in_memory_db["conn"],
    )
    assert cfg1["is_active"] == 1
    assert cfg2["is_active"] == 0

    # Set config2 active, deactivating config1
    success = set_config_active(config_id2, connection=in_memory_db["conn"])
    assert success is True

    cfg1 = execute_query(
        "SELECT is_active FROM configs WHERE config_id = ?",
        (config_id1,),
        connection=in_memory_db["conn"],
    )
    cfg2 = execute_query(
        "SELECT is_active FROM configs WHERE config_id = ?",
        (config_id2,),
        connection=in_memory_db["conn"],
    )
    assert cfg1["is_active"] == 0
    assert cfg2["is_active"] == 1

    # Test setting non-existent config active
    success = set_config_active("nonexistent-id", connection=in_memory_db["conn"])
    assert success is False


# =============================================
# Tests for result-related functions
# =============================================


def test_create_data_upload_result(in_memory_db):
    """Test creating a data upload result"""
    conn = in_memory_db["conn"]

    # Create a job first
    job_id = create_job("data_upload", {}, connection=conn)

    # Create a processing run
    run_id = create_processing_run(
        start_time=datetime.now(),
        status="completed",
        cutoff_date="2023-01-01",
        source_files="file1.csv,file2.csv",
        connection=conn,
    )

    # Create data upload result
    records_processed = 100
    features_generated = ["sales", "stock"]

    result_id = create_data_upload_result(
        job_id=job_id,
        records_processed=records_processed,
        features_generated=features_generated,
        processing_run_id=run_id,
        connection=conn,
    )

    # Verify result was created
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM data_upload_results WHERE result_id = ?", (result_id,)
    )
    result = cursor.fetchone()

    assert result is not None
    assert result["job_id"] == job_id
    assert result["records_processed"] == records_processed
    assert result["processing_run_id"] == run_id

    # Verify features_generated was stored as JSON
    stored_features = json.loads(result["features_generated"])
    assert stored_features == features_generated


def test_create_prediction_result(in_memory_db):
    """Test creating a prediction result"""
    conn = in_memory_db["conn"]

    # Create a job and model
    job_id = create_job("prediction", {}, connection=conn)

    cursor = conn.cursor()
    model_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    cursor.execute(
        "INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)",
        (model_id, job_id, "/path/to/model.onnx", now),
    )
    conn.commit()

    # Create prediction result
    output_path = "/path/to/predictions.csv"
    summary_metrics = {"mape": 15.3, "rmse": 5.2}

    result_id = create_prediction_result(
        job_id=job_id,
        model_id=model_id,
        output_path=output_path,
        summary_metrics=summary_metrics,
        connection=conn,
    )

    # Verify result was created
    cursor.execute("SELECT * FROM prediction_results WHERE result_id = ?", (result_id,))
    result = cursor.fetchone()

    assert result is not None
    assert result["job_id"] == job_id
    assert result["model_id"] == model_id
    assert result["output_path"] == output_path

    # Verify metrics were stored as JSON
    stored_metrics = json.loads(result["summary_metrics"])
    assert stored_metrics == summary_metrics


# =============================================
# Tests for processing runs
# =============================================


def test_create_and_update_processing_run(in_memory_db):
    """Test creating and updating a processing run"""
    conn = in_memory_db["conn"]

    # Create a processing run
    start_time = datetime.now()
    status = "running"
    cutoff_date = "2023-01-01"
    source_files = "file1.csv,file2.csv"

    run_id = create_processing_run(
        start_time=start_time,
        status=status,
        cutoff_date=cutoff_date,
        source_files=source_files,
        connection=conn,
    )

    # Verify run was created
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM processing_runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()

    assert run is not None
    assert run["status"] == status
    assert run["cutoff_date"] == cutoff_date
    assert run["source_files"] == source_files

    # Update run
    new_status = "completed"
    end_time = datetime.now()

    update_processing_run(
        run_id=run_id, status=new_status, end_time=end_time, connection=conn
    )

    # Verify run was updated
    cursor.execute("SELECT * FROM processing_runs WHERE run_id = ?", (run_id,))
    updated_run = cursor.fetchone()

    assert updated_run["status"] == new_status
    assert updated_run["end_time"] == end_time.isoformat()


def test_get_effective_config_active_and_best(in_memory_db, sample_config):
    """Test get_effective_config when an active config exists (should be prioritized)"""

    # Create a dummy settings object for the test
    class MockSettings:
        def __init__(self):
            self.default_metric = "mae"
            self.default_metric_higher_is_better = False

    mock_settings = MockSettings()

    # Create a job and model first to satisfy FK constraints
    job_id = create_job("training", {}, connection=in_memory_db["conn"])
    model_id = create_model_record(
        model_id="model1",
        job_id=job_id,
        model_path="/path/to/model",
        created_at=datetime.now(),
        connection=in_memory_db["conn"],
    )

    # Create a training result with metrics (so get_best_config_by_metric can find something)
    # First, create a config that will be "the best" by metric
    best_metric_config_data = {"key": "best_metric_config"}
    best_metric_config_id = create_or_get_config(
        best_metric_config_data, connection=in_memory_db["conn"]
    )

    create_training_result(
        job_id=job_id,
        model_id=model_id,
        config_id=best_metric_config_id,
        metrics={"mae": 0.5, "rmse": 0.6},
        config=best_metric_config_data,
        duration=100,
        connection=in_memory_db["conn"],
    )

    # Create an active config (this should be returned by get_effective_config)
    active_config_data = sample_config  # Use the sample_config for the active one
    active_config_id = create_or_get_config(
        active_config_data, is_active=True, connection=in_memory_db["conn"]
    )

    effective_config = get_effective_config(
        mock_settings, logger=MagicMock(), connection=in_memory_db["conn"]
    )

    assert effective_config is not None
    assert effective_config["config_id"] == active_config_id
    assert effective_config["config"] == active_config_data


def test_get_effective_config_only_best(in_memory_db, sample_config):
    """Test get_effective_config when no active config, but a best by metric exists"""

    class MockSettings:
        def __init__(self):
            self.default_metric = "mae"
            self.default_metric_higher_is_better = False

    mock_settings = MockSettings()

    # Create a job and model first to satisfy FK constraints
    job_id = create_job("training", {}, connection=in_memory_db["conn"])
    model_id = create_model_record(
        model_id="model2",
        job_id=job_id,
        model_path="/path/to/model",
        created_at=datetime.now(),
        connection=in_memory_db["conn"],
    )

    # No active config
    execute_query("UPDATE configs SET is_active = 0", connection=in_memory_db["conn"])

    # Create a training result with metrics (so get_best_config_by_metric can find something)
    best_metric_config_data = (
        sample_config  # Use sample_config as the best metric config
    )
    best_metric_config_id = create_or_get_config(
        best_metric_config_data, connection=in_memory_db["conn"]
    )

    create_training_result(
        job_id=job_id,
        model_id=model_id,
        config_id=best_metric_config_id,
        metrics={"mae": 0.1, "rmse": 0.2},
        config=best_metric_config_data,
        duration=120,
        connection=in_memory_db["conn"],
    )

    effective_config = get_effective_config(
        mock_settings, logger=MagicMock(), connection=in_memory_db["conn"]
    )

    assert effective_config is not None
    assert effective_config["config_id"] == best_metric_config_id
    assert effective_config["config"] == best_metric_config_data


def test_get_effective_config_no_config(in_memory_db):
    """Test get_effective_config when no active or best config exists"""

    class MockSettings:
        def __init__(self):
            self.default_metric = "mae"
            self.default_metric_higher_is_better = False

    mock_settings = MockSettings()

    # Ensure no configs exist or are active
    execute_query("DELETE FROM configs", connection=in_memory_db["conn"])
    execute_query("DELETE FROM training_results", connection=in_memory_db["conn"])

    with pytest.raises(
        ValueError, match="No active config and no best config by metric available"
    ):
        get_effective_config(
            mock_settings, logger=MagicMock(), connection=in_memory_db["conn"]
        )
