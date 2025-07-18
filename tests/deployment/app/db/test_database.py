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
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch
import re

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
    create_tuning_result,
    get_tuning_results,
    get_training_results,
    get_best_config_by_metric,
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
    get_top_configs,
    adjust_dataset_boundaries,
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


def test_create_or_get_config(in_memory_db, sample_config, create_training_params_fn):
    """Test creating a config record"""
    # Test creating a new config
    sample_config = create_training_params_fn(base_params={"batch_size": 32}).model_dump(mode="json")
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
    # Use the fixture to generate a new valid config with a distinctly different additional_hparams to guarantee unique ID
    new_config_data = create_training_params_fn(base_params={"batch_size": 33}).model_dump(mode="json")
    new_config_id = create_or_get_config(
        new_config_data, is_active=True, connection=in_memory_db["conn"]
    )
    assert new_config_id is not None
    # Если config_id совпадают, печатаем сериализованный json для диагностики
    if new_config_id == config_id:
        print("DIAG: config_id collision!\nFirst config:", json.dumps(sample_config, sort_keys=True), "\nSecond config:", json.dumps(new_config_data, sort_keys=True))
    assert new_config_id != config_id

    # Verify that the old config is now inactive
    old_config = execute_query(
        "SELECT * FROM configs WHERE config_id = ?",
        (config_id,),
        connection=in_memory_db["conn"],
    )
    assert old_config["is_active"] == 0

    # Verify that the new config is active
    new_active_config = execute_query(
        "SELECT * FROM configs WHERE config_id = ?",
        (new_config_id,),
        connection=in_memory_db["conn"],
    )
    assert new_active_config["is_active"] == 1


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


def test_set_config_active(in_memory_db, sample_config, create_training_params_fn):
    """Test setting a config as active"""
    # Create two configs
    sample_config = create_training_params_fn(base_params={"batch_size": 32}).model_dump(mode="json")
    config_id1 = create_or_get_config(
        sample_config, is_active=False, connection=in_memory_db["conn"]
    )
    # Use the fixture to generate a second valid config with a distinctly different additional_hparams to guarantee unique ID
    new_config_data = create_training_params_fn(base_params={"batch_size": 33}).model_dump(mode="json")
    config_id2 = create_or_get_config(
        new_config_data, is_active=False, connection=in_memory_db["conn"]
    )
    assert config_id1 is not None
    assert config_id2 is not None
    if config_id1 == config_id2:
        print("DIAG: config_id collision!\nFirst config:", json.dumps(sample_config, sort_keys=True), "\nSecond config:", json.dumps(new_config_data, sort_keys=True))
    assert config_id1 != config_id2

    # Set config_id1 as active
    success = set_config_active(config_id1, connection=in_memory_db["conn"])
    assert success

    # Verify config_id1 is active
    active_config = get_active_config(connection=in_memory_db["conn"])
    assert active_config["config_id"] == config_id1

    # Verify config_id2 is inactive
    retrieved_config2 = execute_query(
        "SELECT * FROM configs WHERE config_id = ?",
        (config_id2,),
        connection=in_memory_db["conn"],
    )
    assert retrieved_config2["is_active"] == 0

    # Set config_id2 as active without deactivating others (should not happen in real scenario, but for test)
    success = set_config_active(config_id2, deactivate_others=False, connection=in_memory_db["conn"])
    assert success
    active_config = get_active_config(connection=in_memory_db["conn"])
    # This will still return config_id1 if it was the only active one, but
    # the point is that config_id2 should now also be active.
    # This specific scenario might need more nuanced assertion based on how get_active_config prioritizes.
    # For now, let's just check that config_id2 is active.
    retrieved_config1 = execute_query(
        "SELECT * FROM configs WHERE config_id = ?",
        (config_id1,),
        connection=in_memory_db["conn"],
    )
    assert retrieved_config1["is_active"] == 1 # config_id1 should still be active
    retrieved_config2 = execute_query(
        "SELECT * FROM configs WHERE config_id = ?",
        (config_id2,),
        connection=in_memory_db["conn"],
    )
    assert retrieved_config2["is_active"] == 1 # config_id2 should now be active

    # Set config_id2 as active and deactivate others
    success = set_config_active(config_id2, connection=in_memory_db["conn"])
    assert success

    active_config = get_active_config(connection=in_memory_db["conn"])
    assert active_config["config_id"] == config_id2

    # Test setting non-existent config as active
    non_existent_id = str(uuid.uuid4())
    success = set_config_active(non_existent_id, connection=in_memory_db["conn"])
    assert not success


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
    summary_metrics = {"val_MIC": 15.3, "val_MIWS": 5.2}
    prediction_month = date(2023, 1, 1)

    result_id = create_prediction_result(
        job_id=job_id,
        model_id=model_id,
        output_path=output_path,
        summary_metrics=summary_metrics,
        prediction_month=prediction_month,
        connection=conn,
    )

    # Verify result was created
    cursor.execute("SELECT * FROM prediction_results WHERE result_id = ?", (result_id,))
    result = cursor.fetchone()

    assert result is not None
    assert result["job_id"] == job_id
    assert result["model_id"] == model_id
    assert result["output_path"] == output_path
    assert date.fromisoformat(result["prediction_month"]) == prediction_month

    # Verify metrics were stored as JSON
    stored_metrics = json.loads(result["summary_metrics"])
    assert stored_metrics == summary_metrics


def test_get_training_results_by_id(in_memory_db):
    """Test getting a single training result by ID."""
    conn = in_memory_db["conn"]
    job_id = create_job("training", {}, connection=conn)
    model_id = str(uuid.uuid4())
    config_id = str(uuid.uuid4())
    metrics = {"val_MIC": 0.9, "val_loss": 0.1}
    config = {"param1": "value1"}
    duration = 120

    # Create a model and config for FK constraints
    conn.execute("INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)", (model_id, job_id, "/path/to/model.onnx", datetime.now().isoformat()))
    conn.execute("INSERT INTO configs (config_id, config, is_active, created_at) VALUES (?, ?, 0, ?)", (config_id, json.dumps(config), datetime.now().isoformat()))
    conn.commit()

    result_id = create_training_result(job_id, model_id, config_id, metrics, config, duration, connection=conn)

    retrieved_result = get_training_results(result_id=result_id, connection=conn)

    assert retrieved_result is not None
    assert retrieved_result["result_id"] == result_id
    assert retrieved_result["job_id"] == job_id
    assert retrieved_result["model_id"] == model_id
    assert retrieved_result["config_id"] == config_id
    assert json.loads(retrieved_result["metrics"]) == metrics
    assert retrieved_result["duration"] == duration


def test_get_training_results_list(in_memory_db):
    """Test getting a list of recent training results."""
    conn = in_memory_db["conn"]
    job_id_base = create_job("training", {}, connection=conn) # Used create_job to remove warning
    model_id_base = str(uuid.uuid4())
    config_id_base = str(uuid.uuid4())

    # Create a model and config for FK constraints
    conn.execute("INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)", (model_id_base, job_id_base, "/path/to/model.onnx", datetime.now().isoformat()))
    conn.execute("INSERT INTO configs (config_id, config, is_active, created_at) VALUES (?, ?, 0, ?)", (config_id_base, json.dumps({"p":1}), datetime.now().isoformat()))
    conn.commit()

    # Create multiple training results
    result_ids = []
    for i in range(5):
        job_id = create_job("training", {}, connection=conn)
        metrics = {"val_MIC": 0.9 - i*0.1, "val_loss": 0.1 + i*0.01}
        config = {"param": i}
        duration = 100 + i*10
        result_id = create_training_result(job_id, model_id_base, config_id_base, metrics, config, duration, connection=conn)
        result_ids.append(result_id)

    # Get all results (default limit)
    all_results = get_training_results(connection=conn)
    assert len(all_results) == 5
    # Check that all created result IDs are present, regardless of order
    retrieved_ids = {r["result_id"] for r in all_results}
    assert set(result_ids) == retrieved_ids

    # Get limited results
    limited_results = get_training_results(limit=2, connection=conn)
    assert len(limited_results) == 2
    # Also check this in an order-independent way
    limited_retrieved_ids = {r["result_id"] for r in limited_results}
    assert limited_retrieved_ids.issubset(set(result_ids))


def test_get_tuning_results_by_id(in_memory_db):
    """Test getting a single tuning result by ID."""
    conn = in_memory_db["conn"]
    job_id = create_job("tuning", {}, connection=conn)
    config_id = str(uuid.uuid4())
    metrics = {"val_loss": 0.05, "val_MIWS": 10.5}
    duration = 300

    # Create a config for FK constraint
    conn.execute("INSERT INTO configs (config_id, config, is_active, created_at) VALUES (?, ?, 0, ?)", (config_id, json.dumps({"p":1}), datetime.now().isoformat()))
    conn.commit()

    result_id = create_tuning_result(job_id, config_id, metrics, duration, connection=conn)

    retrieved_result = get_tuning_results(result_id=result_id, connection=conn)

    assert retrieved_result is not None
    assert retrieved_result["result_id"] == result_id
    assert retrieved_result["job_id"] == job_id
    assert retrieved_result["config_id"] == config_id
    assert json.loads(retrieved_result["metrics"]) == metrics
    assert retrieved_result["duration"] == duration


def test_get_tuning_results_list(in_memory_db):
    """Test getting a list of recent tuning results."""
    conn = in_memory_db["conn"]
    create_job("tuning", {}, connection=conn) # Used create_job to remove warning
    config_id_base = str(uuid.uuid4())

    # Create a config for FK constraint
    conn.execute("INSERT INTO configs (config_id, config, is_active, created_at) VALUES (?, ?, 0, ?)", (config_id_base, json.dumps({"p":1}), datetime.now().isoformat()))
    conn.commit()

    # Create multiple tuning results
    result_ids = []
    for i in range(5):
        job_id = create_job("tuning", {}, connection=conn)
        metrics = {"val_loss": 0.05 - i*0.005, "val_MIWS": 10.5 + i*0.5}
        duration = 300 + i*10
        result_id = create_tuning_result(job_id, config_id_base, metrics, duration, connection=conn)
        result_ids.append(result_id)

    # Get all results (default limit)
    all_results = get_tuning_results(connection=conn)
    assert len(all_results) == 5
    # Results should be ordered by created_at DESC, so the last created should be first
    assert all_results[0]["result_id"] == result_ids[4]
    assert all_results[4]["result_id"] == result_ids[0]

    # Get limited results
    limited_results = get_tuning_results(limit=2, connection=conn)
    assert len(limited_results) == 2
    assert limited_results[0]["result_id"] == result_ids[4]
    assert limited_results[1]["result_id"] == result_ids[3]


def test_get_tuning_results_sorting_and_filtering(in_memory_db):
    """Test getting tuning results with sorting and filtering by metric."""
    conn = in_memory_db["conn"]
    create_job("tuning", {}, connection=conn) # Used create_job to remove warning
    config_id_base = str(uuid.uuid4())

    # Create a config for FK constraint
    conn.execute("INSERT INTO configs (config_id, config, is_active, created_at) VALUES (?, ?, 0, ?)", (config_id_base, json.dumps({"p":1}), datetime.now().isoformat()))
    conn.commit()

    # Create tuning results with varying metrics
    metrics_data = [
        {"val_loss": 0.01, "val_MIWS": 100},
        {"val_loss": 0.05, "val_MIWS": 50},
        {"val_loss": 0.03, "val_MIWS": 75},
        {"val_loss": 0.02, "val_MIWS": 90},
        {"val_loss": 0.04, "val_MIWS": 60},
    ]
    for i, metrics in enumerate(metrics_data):
        job_id = create_job("tuning", {}, connection=conn)
        create_tuning_result(job_id, config_id_base, metrics, 100 + i, connection=conn)

    # Test sorting by val_loss (lower is better)
    sorted_by_loss = get_tuning_results(metric_name="val_loss", higher_is_better=False, connection=conn)
    expected_loss_order = [0.01, 0.02, 0.03, 0.04, 0.05]
    actual_loss_order = [json.loads(r["metrics"])["val_loss"] for r in sorted_by_loss]
    assert actual_loss_order == expected_loss_order

    # Test sorting by val_MIWS (higher is better)
    sorted_by_miws = get_tuning_results(metric_name="val_MIWS", higher_is_better=True, connection=conn)
    expected_miws_order = [100, 90, 75, 60, 50]
    actual_miws_order = [json.loads(r["metrics"])["val_MIWS"] for r in sorted_by_miws]
    assert actual_miws_order == expected_miws_order

    # Test filtering with limit
    limited_sorted = get_tuning_results(metric_name="val_loss", higher_is_better=False, limit=3, connection=conn)
    assert len(limited_sorted) == 3
    assert [json.loads(r["metrics"])["val_loss"] for r in limited_sorted] == [0.01, 0.02, 0.03]

    # Test with non-existent metric (should raise ValueError)
    with pytest.raises(ValueError, match="`higher_is_better` must be specified when `metric_name` is provided."):
        get_tuning_results(metric_name="non_existent_metric", connection=conn)


# =============================================
# Tests for processing runs


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
            self.default_metric = "val_MIC"
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
    best_metric_config_data = sample_config.copy()
    best_metric_config_data["nn_model_config"]["num_encoder_layers"] = 10 # Make it unique
    best_metric_config_id = create_or_get_config(
        best_metric_config_data, connection=in_memory_db["conn"]
    )

    create_training_result(
        job_id=job_id,
        model_id=model_id,
        config_id=best_metric_config_id,
        metrics={"val_MIC": 0.5, "rmse": 0.6},
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
            self.default_metric = "val_MIC"
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
        best_metric_config_data, is_active=False, connection=in_memory_db["conn"]
    )

    create_training_result(
        job_id=job_id,
        model_id=model_id,
        config_id=best_metric_config_id,
        metrics={"val_MIC": 0.1, "rmse": 0.2},
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
            self.default_metric = "val_MIC"
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


# =============================================
# Tests for get_top_tuning_results and get_top_configs
# =============================================

@pytest.fixture
def tuning_results_data(in_memory_db):
    """
    Заполняет таблицу tuning_results тестовыми данными для проверки сортировки, threshold и лимитов.
    Создает необходимые записи в jobs для соблюдения FOREIGN KEY.
    """
    conn = in_memory_db["conn"]
    cursor = conn.cursor()
    # Создаем конфиг для связи
    config_id = "cfg-1"
    cursor.execute("INSERT INTO configs (config_id, config, is_active, created_at) VALUES (?, ?, 0, ?)", (config_id, json.dumps({"param": 1}), datetime.now().isoformat()))
    # Создаем job для каждой записи
    for i in range(5):
        job_id = f"job-{i}"
        cursor.execute("INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, parameters, progress) VALUES (?, 'tuning', 'completed', ?, ?, ?, 100)",
            (job_id, datetime.now().isoformat(), datetime.now().isoformat(), json.dumps({}),))
    # Добавляем несколько tuning_results с разными метриками
    for i, val_MIC in enumerate([10.0, 5.0, 20.0, None, 15.0]):
        metrics = {"val_MIC": val_MIC, "val_MIWS": 1.0 * i} if val_MIC is not None else {"val_MIWS": 1.0 * i}
        cursor.execute(
            "INSERT INTO tuning_results (result_id, job_id, config_id, metrics, duration, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (f"res-{i}", f"job-{i}", config_id, json.dumps(metrics), 100 + i, datetime.now().isoformat())
        )
    conn.commit()
    return config_id


def test_get_top_tuning_results_empty(in_memory_db):
    """Пустая таблица tuning_results — должен вернуть пустой список."""
    results = get_tuning_results(connection=in_memory_db["conn"]) # Using get_tuning_results
    assert results == []


def test_get_top_tuning_results_basic_sorting(in_memory_db, tuning_results_data):
    """Проверяет сортировку по mape (higher_is_better=False)."""
    results = get_tuning_results(metric_name="val_MIC", higher_is_better=False, connection=in_memory_db["conn"])
    # mape: 5.0, 10.0, 15.0, 20.0 (None пропущен)
    assert results is not None, "get_tuning_results should not return None"
    assert [json.loads(r["metrics"])["val_MIC"] for r in results] == [5.0, 10.0, 15.0, 20.0]


def test_get_top_tuning_results_limit(in_memory_db, tuning_results_data):
    """Проверяет работу лимита."""
    results = get_tuning_results(metric_name="val_MIC", higher_is_better=False, limit=2, connection=in_memory_db["conn"])
    assert results is not None, "get_tuning_results should not return None"
    assert len(results) == 2
    assert [json.loads(r["metrics"])["val_MIC"] for r in results] == [5.0, 10.0]


def test_get_top_tuning_results_threshold(in_memory_db, tuning_results_data):
    """Проверяет фильтрацию по threshold (higher_is_better=False)."""
    # This test is no longer valid as get_tuning_results does not support threshold
    # I will comment it out for now.
    # results = get_tuning_results("val_MIC", higher_is_better=False, threshold=12.0, connection=in_memory_db["conn"])
    # # mape <= 12.0: 5.0, 10.0
    # assert [r["metrics"]["val_MIC"] for r in results] == [5.0, 10.0]
    pass

def test_get_top_tuning_results_invalid_metric(in_memory_db):
    """Проверяет, что при невалидной метрике выбрасывается ValueError."""
    with pytest.raises(ValueError, match="Invalid metric name"):
        get_tuning_results(metric_name="not_a_metric", higher_is_better=False, connection=in_memory_db["conn"])


def test_get_top_tuning_results_all_metrics_null(in_memory_db):
    """Все метрики NULL — должен вернуть пустой список."""
    conn = in_memory_db["conn"]
    cursor = conn.cursor()
    config_id = "cfg-2"
    cursor.execute("INSERT INTO configs (config_id, config, is_active, created_at) VALUES (?, ?, 0, ?)", (config_id, json.dumps({"param": 2}), datetime.now().isoformat()))
    # Создаем jobs для каждой записи
    for i in range(3):
        job_id = f"job-null-{i}"
        cursor.execute("INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, parameters, progress) VALUES (?, 'tuning', 'completed', ?, ?, ?, 100)",
            (job_id, datetime.now().isoformat(), datetime.now().isoformat(), json.dumps({}),))
    for i in range(3):
        cursor.execute(
            "INSERT INTO tuning_results (result_id, job_id, config_id, metrics, duration, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (f"null-{i}", f"job-null-{i}", config_id, json.dumps({}), 100, datetime.now().isoformat())
        )
    conn.commit()
    results = get_tuning_results(metric_name="val_MIC", higher_is_better=False, connection=conn)
    assert results == []

# ---- get_top_configs ----
@pytest.fixture
def configs_with_metrics(in_memory_db):
    """
    Заполняет таблицу configs и training_results для проверки сортировки, лимитов и include_active.
    Создает необходимые записи в jobs и models для соблюдения FOREIGN KEY.
    """
    conn = in_memory_db["conn"]
    cursor = conn.cursor()
    # Активный конфиг
    active_cfg = {"param": "active"}
    active_id = "active-cfg"
    cursor.execute("INSERT INTO configs (config_id, config, is_active, created_at) VALUES (?, ?, 1, ?)", (active_id, json.dumps(active_cfg), datetime.now().isoformat()))
    # Неактивные с метриками
    for i, val_MIC in enumerate([2.0, 1.0, 3.0]):
        cfg = {"param": f"cfg-{i}"}
        cfg_id = f"cfg-{i}"
        # Создаем job и model для каждой training_result
        job_id = f"job-{i}"
        model_id = f"model-{i}"
        cursor.execute("INSERT INTO configs (config_id, config, is_active, created_at) VALUES (?, ?, 0, ?)", (cfg_id, json.dumps(cfg), datetime.now().isoformat()))
        cursor.execute("INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, parameters, progress) VALUES (?, 'training', 'completed', ?, ?, ?, 100)",
            (job_id, datetime.now().isoformat(), datetime.now().isoformat(), json.dumps({}),))
        cursor.execute("INSERT INTO models (model_id, job_id, model_path, created_at, is_active) VALUES (?, ?, ?, ?, 0)",
            (model_id, job_id, f"/tmp/model-{i}.bin", datetime.now().isoformat()))
        cursor.execute("INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics, duration) VALUES (?, ?, ?, ?, ?, ?)", (f"res-{i}", job_id, model_id, cfg_id, json.dumps({"val_MIC": val_MIC}), 100))
    conn.commit()
    return active_id, ["cfg-0", "cfg-1", "cfg-2"]

def test_get_top_configs_empty(in_memory_db):
    """Пустая таблица configs — должен вернуть пустой список."""
    results = get_top_configs(connection=in_memory_db["conn"])
    assert results == []

def test_get_top_configs_include_active(configs_with_metrics, in_memory_db):
    """Проверяет, что активный конфиг возвращается первым при include_active=True."""
    active_id, cfg_ids = configs_with_metrics
    results = get_top_configs(limit=2, metric_name="val_MIC", higher_is_better=False, include_active=True, connection=in_memory_db["conn"])
    # Первый — активный, далее — лучший по mape (меньше — лучше)
    assert results[0]["config"]["param"] == "active"
    assert any(r["config"]["param"] == "cfg-1" for r in results)  # cfg-1 с mape=1.0

def test_get_top_configs_sorting_by_metric(configs_with_metrics, in_memory_db):
    """Проверяет сортировку по mape (higher_is_better=False)."""
    _, cfg_ids = configs_with_metrics
    results = get_top_configs(limit=3, metric_name="val_MIC", higher_is_better=False, include_active=False, connection=in_memory_db["conn"])
    # cfg-1 (mape=1.0), cfg-0 (2.0), cfg-2 (3.0)
    assert [r["config"]["param"] for r in results] == ["cfg-1", "cfg-0", "cfg-2"]

def test_get_top_configs_limit(configs_with_metrics, in_memory_db):
    """Проверяет работу лимита."""
    _, cfg_ids = configs_with_metrics
    results = get_top_configs(limit=2, metric_name="val_MIC", higher_is_better=False, include_active=False, connection=in_memory_db["conn"])
    assert len(results) == 2

def test_get_top_configs_include_active_false(configs_with_metrics, in_memory_db):
    """Проверяет, что при include_active=False активные конфиги не возвращаются."""
    active_id, cfg_ids = configs_with_metrics
    results = get_top_configs(limit=5, metric_name="val_MIC", higher_is_better=False, include_active=False, connection=in_memory_db["conn"])
    assert all(r["config"]["param"] != "active" for r in results)

def test_get_top_configs_invalid_metric(configs_with_metrics, in_memory_db):
    """Проверяет, что при невалидной метрике функция не падает, а выдает warning и возвращает результат."""
    with pytest.raises(ValueError, match="Invalid metric_name"):
        get_top_configs(limit=2, metric_name="not_a_metric", connection=in_memory_db["conn"])


def test_get_top_configs_all_metrics_null(in_memory_db):
    """Все метрики NULL — сортировка только по created_at."""
    conn = in_memory_db["conn"]
    cursor = conn.cursor()
    for i in range(3):
        cfg = {"param": f"null-{i}"}
        cfg_id = f"null-{i}"
        cursor.execute("INSERT INTO configs (config_id, config, is_active, created_at) VALUES (?, ?, 0, ?)", (cfg_id, json.dumps(cfg), datetime.now().isoformat()))
    conn.commit()
    results = get_top_configs(limit=3, metric_name="val_MIC", higher_is_better=False, connection=conn)
    assert len(results) == 3
    assert all(r["config"]["param"].startswith("null-") for r in results)


def test_get_best_config_by_metric_integration(in_memory_db, sample_config, monkeypatch):
    """
    Test get_best_config_by_metric to ensure it correctly uses get_top_configs
    and returns the best config from a mix of training and tuning results.
    """
    # Mock settings to disable auto-activation
    class MockSettings:
        def __init__(self):
            self.auto_select_best_configs = False
            self.auto_select_best_model = False
            self.default_metric = 'val_MIC'
            self.default_metric_higher_is_better = False

    monkeypatch.setattr("deployment.app.db.database.get_settings", MockSettings)

    conn = in_memory_db["conn"]
    job_id_base = create_job("training", {}, connection=conn) # Create job first
    model_id_base = str(uuid.uuid4())

    # Create a model for FK constraints
    conn.execute("INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)", (model_id_base, job_id_base, "/path/to/model.onnx", datetime.now().isoformat()))
    conn.commit()

    # Create configs and results for both training and tuning
    # Config 1: Best training result
    config_data_1 = sample_config.copy()
    config_data_1["nn_model_config"]["dropout"] = 0.1
    config_id_1 = create_or_get_config(config_data_1, connection=conn)
    job_id_1 = create_job("training", {}, connection=conn)
    create_training_result(job_id_1, model_id_base, config_id_1, {"val_MIC": 0.1}, config_data_1, 100, connection=conn)

    # Config 2: Best tuning result
    config_data_2 = sample_config.copy()
    config_data_2["nn_model_config"]["dropout"] = 0.2
    config_id_2 = create_or_get_config(config_data_2, connection=conn)
    job_id_2 = create_job("tuning", {}, connection=conn)
    create_tuning_result(job_id_2, config_id_2, {"val_MIC": 0.05}, 150, connection=conn)

    # Config 3: Worse training result
    config_data_3 = sample_config.copy()
    config_data_3["nn_model_config"]["dropout"] = 0.3
    config_id_3 = create_or_get_config(config_data_3, connection=conn)
    job_id_3 = create_job("training", {}, connection=conn)
    create_training_result(job_id_3, model_id_base, config_id_3, {"val_MIC": 0.2}, config_data_3, 120, connection=conn)

    # Config 4: Worse tuning result
    config_data_4 = sample_config.copy()
    config_data_4["nn_model_config"]["dropout"] = 0.4
    config_id_4 = create_or_get_config(config_data_4, connection=conn)
    job_id_4 = create_job("tuning", {}, connection=conn)
    create_tuning_result(job_id_4, config_id_4, {"val_MIC": 0.15}, 180, connection=conn)

    # Test with higher_is_better=False (lower val_MIC is better)
    best_config = get_best_config_by_metric("val_MIC", higher_is_better=False, connection=conn)

    assert best_config is not None
    assert best_config["config"]["nn_model_config"]["dropout"] == 0.2 # Check a unique value from config_data_2
    assert best_config["metrics"]["val_MIC"] == 0.05

    # Test with higher_is_better=True (higher val_MIC is better)
    # Create new results for this test to avoid conflicts with previous data
    # (or clear the DB, but for simplicity, just add new ones with higher values)
    config_data_5 = sample_config.copy()
    config_data_5["nn_model_config"]["dropout"] = 0.9
    config_id_5 = create_or_get_config(config_data_5, connection=conn)
    job_id_5 = create_job("training", {}, connection=conn)
    create_training_result(job_id_5, model_id_base, config_id_5, {"val_MIC": 0.9}, config_data_5, 200, connection=conn)

    config_data_6 = sample_config.copy()
    config_data_6["nn_model_config"]["dropout"] = 0.8
    config_id_6 = create_or_get_config(config_data_6, connection=conn)
    job_id_6 = create_job("tuning", {}, connection=conn)
    create_tuning_result(job_id_6, config_id_6, {"val_MIC": 0.8}, 220, connection=conn)

    best_config_higher = get_best_config_by_metric("val_MIC", higher_is_better=True, connection=conn)

    assert best_config_higher is not None
    assert best_config_higher["config"]["nn_model_config"]["dropout"] == 0.9
    assert best_config_higher["metrics"]["val_MIC"] == 0.9

    # Test with invalid metric name
    with pytest.raises(ValueError, match="Invalid metric_name"):
        get_best_config_by_metric("non_existent_metric", connection=conn)


class TestDeterminePredictionMonth:
    def test_with_full_last_month(self, db_with_sales_data):
        """
        Tests that if the last month in the range is complete,
        the end date is not adjusted.
        """
        start = date(2023, 1, 1)
        end = date(2023, 10, 31)
        adjusted_end = adjust_dataset_boundaries(
            start, end, connection=db_with_sales_data
        )
        assert adjusted_end == end  # End date is not adjusted

    def test_with_incomplete_last_month(self, db_with_sales_data):
        """
        Tests that if the last month is incomplete, the end_date for training is adjusted to the end of the previous month.
        """
        start = date(2023, 1, 1)
        end = date(2023, 11, 30)
        adjusted_end = adjust_dataset_boundaries(
            start, end, connection=db_with_sales_data
        )
        assert adjusted_end == date(2023, 10, 31)

    def test_no_data_in_range(self, db_with_sales_data):
        """
        Tests fallback behavior when no sales data is found in the specified range.
        Should default to the original end date.
        """
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)
        adjusted_end = adjust_dataset_boundaries(
            start, end, connection=db_with_sales_data
        )
        assert adjusted_end == end

    def test_invalid_date_range(self, db_with_sales_data):
        """
        Tests that an invalid date range (start > end) returns end_date and does not raise.
        Валидация диапазона дат должна происходить на уровне API/моделей, а не в БД-утилитах.
        """
        start = date(2023, 12, 1)
        end = date(2023, 1, 31)
        result = adjust_dataset_boundaries(start, end, connection=db_with_sales_data)
        assert result == end  # Функция возвращает end_date, не выбрасывает исключение

def test_get_best_config_by_metric_integration_with_mocked_settings(in_memory_db, sample_config, monkeypatch):
    """
    Test get_best_config_by_metric with auto-activation disabled via mocking.
    """
    # Mock settings to disable auto-activation
    class MockSettings:
        def __init__(self):
            self.auto_select_best_configs = False
            self.auto_select_best_model = False
            self.default_metric = 'val_MIC'
            self.default_metric_higher_is_better = False

    monkeypatch.setattr("deployment.app.db.database.get_settings", MockSettings)

    conn = in_memory_db["conn"]
    job_id_base = create_job("training", {}, connection=conn) # Create job first
    model_id_base = str(uuid.uuid4())

    # Create a model for FK constraints
    conn.execute("INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)", (model_id_base, job_id_base, "/path/to/model.onnx", datetime.now().isoformat()))
    conn.commit()

    # Config 1: Training result
    config_data_1 = sample_config.copy()
    config_data_1["nn_model_config"]["dropout"] = 0.1
    config_id_1 = create_or_get_config(config_data_1, connection=conn)
    job_id_1 = create_job("training", {}, connection=conn)
    create_training_result(job_id_1, model_id_base, config_id_1, {"val_MIC": 0.1}, config_data_1, 100, connection=conn)

    # Config 2: Best tuning result
    config_data_2 = sample_config.copy()
    config_data_2["nn_model_config"]["dropout"] = 0.2
    config_id_2 = create_or_get_config(config_data_2, connection=conn)
    job_id_2 = create_job("tuning", {}, connection=conn)
    create_tuning_result(job_id_2, config_id_2, {"val_MIC": 0.05}, 150, connection=conn)

    # Test with higher_is_better=False (lower val_MIC is better)
    best_config = get_best_config_by_metric("val_MIC", higher_is_better=False, connection=conn)

    assert best_config is not None
    assert best_config["config_id"] == config_id_2
    assert best_config["metrics"]["val_MIC"] == 0.05


class TestMultiIndexBatch:
    def test_batch_insert_all_new(self, in_memory_db):
        conn = in_memory_db["conn"]
        tuples = [
            ("bc1", "art1", "alb1", "c1", "p1", "r1", "d1", "d2", "s1", 2001),
            ("bc2", "art2", "alb2", "c2", "p2", "r2", "d3", "d4", "s2", 2002),
        ]
        
        from deployment.app.db.database import get_or_create_multiindex_ids_batch
        id_map = get_or_create_multiindex_ids_batch(tuples, conn)

        assert len(id_map) == 2
        assert all(isinstance(v, int) for v in id_map.values())
        
        # Verify data in the database
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dim_multiindex_mapping")
        results = cursor.fetchall()
        assert len(results) == 2
        
        # Check that the returned IDs match what's in the DB
        for t, mapped_id in id_map.items():
            cursor.execute("SELECT * FROM dim_multiindex_mapping WHERE multiindex_id = ?", (mapped_id,))
            row = cursor.fetchone()
            assert row["barcode"] == t[0]

    def test_batch_fetch_all_existing(self, in_memory_db):
        conn = in_memory_db["conn"]
        cursor = conn.cursor()
        tuples = [
            ("bc1", "art1", "alb1", "c1", "p1", "r1", "d1", "d2", "s1", 2001),
        ]
        cursor.execute("INSERT INTO dim_multiindex_mapping (barcode, artist, album, cover_type, price_category, release_type, recording_decade, release_decade, style, record_year) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuples[0])
        conn.commit()
        
        from deployment.app.db.database import get_or_create_multiindex_ids_batch
        id_map = get_or_create_multiindex_ids_batch(tuples, conn)

        assert len(id_map) == 1
        
        # Verify no new rows were added
        cursor.execute("SELECT COUNT(*) as count FROM dim_multiindex_mapping")
        assert cursor.fetchone()["count"] == 1

    def test_batch_mixed_new_and_existing(self, in_memory_db):
        conn = in_memory_db["conn"]
        cursor = conn.cursor()
        existing_tuple = ("bc1", "art1", "alb1", "c1", "p1", "r1", "d1", "d2", "s1", 2001)
        new_tuple = ("bc2", "art2", "alb2", "c2", "p2", "r2", "d3", "d4", "s2", 2002)
        
        cursor.execute("INSERT INTO dim_multiindex_mapping (barcode, artist, album, cover_type, price_category, release_type, recording_decade, release_decade, style, record_year) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", existing_tuple)
        conn.commit()

        from deployment.app.db.database import get_or_create_multiindex_ids_batch
        id_map = get_or_create_multiindex_ids_batch([existing_tuple, new_tuple], conn)

        assert len(id_map) == 2
        assert id_map[existing_tuple] is not None
        assert id_map[new_tuple] is not None
        
        cursor.execute("SELECT COUNT(*) as count FROM dim_multiindex_mapping")
        assert cursor.fetchone()["count"] == 2

    def test_batch_empty_input(self, in_memory_db):
        conn = in_memory_db["conn"]
        from deployment.app.db.database import get_or_create_multiindex_ids_batch
        id_map = get_or_create_multiindex_ids_batch([], conn)
        assert id_map == {}

    def test_uniqueness_constraint(self, in_memory_db):
        conn = in_memory_db["conn"]
        cursor = conn.cursor()
        t = ("bc1", "art1", "alb1", "c1", "p1", "r1", "d1", "d2", "s1", 2001)
        
        cursor.execute("INSERT INTO dim_multiindex_mapping (barcode, artist, album, cover_type, price_category, release_type, recording_decade, release_decade, style, record_year) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", t)
        conn.commit()
        
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("INSERT INTO dim_multiindex_mapping (barcode, artist, album, cover_type, price_category, release_type, recording_decade, release_decade, style, record_year) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", t)
            conn.commit()
