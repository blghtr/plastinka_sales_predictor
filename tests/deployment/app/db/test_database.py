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
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from deployment.app.db.database import (
    DatabaseError,
    adjust_dataset_boundaries,
    create_data_upload_result,
    create_job,
    create_model_record,
    create_or_get_config,
    create_prediction_result,
    create_processing_run,
    create_training_result,
    create_tuning_result,
    dict_factory,
    execute_many,
    execute_query,
    generate_id,
    get_active_config,
    get_active_model,
    get_best_config_by_metric,
    get_db_connection,
    get_effective_config,
    get_feature_dataframe,
    get_job,
    get_top_configs,
    get_training_results,
    get_tuning_results,
    list_jobs,
    set_config_active,
    set_model_active,
    update_job_status,
    update_processing_run,
)

# =============================================
# Используем фикстуры из conftest.py
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
    conn = in_memory_db._connection

    # Add some test data
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
    )
    cursor.execute("INSERT INTO test_table (name) VALUES (?)", ("test_name",))
    conn.commit()

    # Test execute_query with SELECT
    result = execute_query(
        "SELECT * FROM test_table WHERE id = ?",
        conn,
        (1,),
    )
    assert result is not None
    assert result["id"] == 1
    assert result["name"] == "test_name"

    # Test execute_query with fetchall=True
    results = execute_query("SELECT * FROM test_table", conn, (), True)
    assert len(results) == 1
    assert results[0]["name"] == "test_name"


def test_execute_query_insert(in_memory_db):
    """Test execute_query with an INSERT statement"""
    conn = in_memory_db._connection

    # Create test table
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
    )

    # Use execute_query for INSERT
    execute_query(
        "INSERT INTO test_table (id, name) VALUES (?, ?)",
        conn,
        (2, "test_name2"),
    )

    # Verify the record was inserted
    cursor.execute("SELECT * FROM test_table WHERE name = ?", ("test_name2",))
    result = cursor.fetchone()
    assert result is not None
    assert result["name"] == "test_name2"  # Access by column name due to dict_factory


def test_execute_query_error(in_memory_db):
    """Test that execute_query raises DatabaseError on SQL errors"""
    conn = in_memory_db._connection

    # Execute a query with syntax error
    with pytest.raises(DatabaseError):
        execute_query("SELECT * FROM nonexistent_table", conn)


def test_execute_many(in_memory_db):
    """Test execute_many with multiple parameter sets"""
    conn = in_memory_db._connection

    # Create test table
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
    )

    # Define parameter sets
    params_list = [("name1",), ("name2",), ("name3",)]

    # Use execute_many
    execute_many(
        "INSERT INTO test_table (name) VALUES (?)", params_list=params_list, connection=conn
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
    conn = in_memory_db._connection

    # Try to execute on a non-existent table
    params_list = [("name1",), ("name2",)]

    with pytest.raises(DatabaseError) as exc_info:
        execute_many(
            "INSERT INTO nonexistent_table (name) VALUES (?)",
            params_list=params_list,
            connection=conn,
        )

    assert "Batch database operation failed" in str(exc_info.value)


def test_execute_many_empty_params(in_memory_db):
    """Test that execute_many handles empty params lists gracefully"""
    conn = in_memory_db._connection

    # Call with empty params list - should return without error
    result = execute_many(
        "INSERT INTO test_table (name) VALUES (?)", params_list=[], connection=conn
    )
    assert result is None


def test_generate_id():
    """Test that generate_id produces unique IDs"""
    ids = [generate_id() for _ in range(10)]
    unique_ids = set(ids)
    assert len(ids) == len(unique_ids)


def test_create_job(in_memory_db):
    """Test creating a job record"""
    conn = in_memory_db._connection
    job_type = "training"
    parameters = {"batch_size": 32, "epochs": 10}
    job_id = create_job(job_type, parameters, connection=conn)
    assert job_id is not None
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    job = cursor.fetchone()
    assert job["job_type"] == job_type
    assert json.loads(job["parameters"]) == parameters

def test_update_job_status(in_memory_db, sample_job_data):
    """Test updating job status"""
    conn = in_memory_db._connection
    job_id = create_job(
        sample_job_data["job_type"], sample_job_data["parameters"], connection=conn
    )
    update_job_status(
        job_id=job_id,
        status="running",
        progress=50,
        status_message="Halfway there",
        connection=conn,
    )
    job = get_job(job_id, connection=conn)
    assert job["status"] == "running"
    assert job["progress"] == 50

def test_get_job_nonexistent(in_memory_db):
    """Test getting a non-existent job"""
    conn = in_memory_db._connection
    result = get_job(str(uuid.uuid4()), connection=conn)
    assert result is None

def test_list_jobs(in_memory_db):
    """Test listing jobs with filters"""
    conn = in_memory_db._connection
    job_1 = create_job("training", {"param": "value1"}, connection=conn)
    job_2 = create_job("prediction", {"param": "value2"}, connection=conn)
    job_3 = create_job("training", {"param": "value3"}, connection=conn)
    update_job_status(job_1, "completed", connection=conn)
    update_job_status(job_2, "running", connection=conn)
    update_job_status(job_3, "running", connection=conn)
    assert len(list_jobs(connection=conn)) == 3
    assert len(list_jobs(job_type="training", connection=conn)) == 2
    assert len(list_jobs(status="running", connection=conn)) == 2
    assert len(list_jobs(job_type="training", status="running", connection=conn)) == 1

def test_create_model_record(in_memory_db, sample_model_data):
    """Test creating a model record"""
    conn = in_memory_db._connection
    job_id = create_job("training", {}, connection=conn)
    create_model_record(
        model_id=sample_model_data["model_id"],
        job_id=job_id,
        model_path=sample_model_data["model_path"],
        created_at=sample_model_data["created_at"],
        metadata=sample_model_data["metadata"],
        is_active=True,
        connection=conn,
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE model_id = ?", (sample_model_data["model_id"],))
    model = cursor.fetchone()
    assert model["model_id"] == sample_model_data["model_id"]

def test_get_active_model(in_memory_db, sample_model_data):
    """Test getting the active model"""
    conn = in_memory_db._connection
    job_id = create_job("training", {}, connection=conn)
    create_model_record(
        model_id=sample_model_data["model_id"],
        job_id=job_id,
        model_path=sample_model_data["model_path"],
        created_at=sample_model_data["created_at"],
        metadata=sample_model_data["metadata"],
        is_active=True,
        connection=conn,
    )
    active_model = get_active_model(connection=conn)
    assert active_model["model_id"] == sample_model_data["model_id"]

def test_set_model_active(in_memory_db):
    """Test setting a model as active"""
    conn = in_memory_db._connection
    job_id = create_job("training", {}, connection=conn)
    model_id_1 = str(uuid.uuid4())
    model_id_2 = str(uuid.uuid4())
    create_model_record(model_id_1, job_id, "/path/1", datetime.now(), connection=conn)
    create_model_record(model_id_2, job_id, "/path/2", datetime.now(), connection=conn)
    set_model_active(model_id_1, connection=conn)
    active_model = get_active_model(connection=conn)
    assert active_model["model_id"] == model_id_1

def test_create_or_get_config(in_memory_db, sample_config):
    """Test creating a config record"""
    config_id = create_or_get_config(
        sample_config, is_active=True, connection=in_memory_db._connection
    )
    assert config_id is not None
    retrieved = execute_query("SELECT * FROM configs WHERE config_id = ?", in_memory_db._connection, (config_id,))
    assert json.loads(retrieved["config"]) == sample_config

def test_get_active_config(in_memory_db, sample_config):
    """Test retrieving the active config"""
    config_id = create_or_get_config(
        sample_config, is_active=True, connection=in_memory_db._connection
    )
    active_config = get_active_config(connection=in_memory_db._connection)
    assert active_config["config_id"] == config_id

def test_set_config_active(in_memory_db, sample_config):
    """Test setting a config as active"""
    config_id1 = create_or_get_config(sample_config, connection=in_memory_db._connection)
    config_id2 = create_or_get_config({"a":1}, connection=in_memory_db._connection)
    set_config_active(config_id1, connection=in_memory_db._connection)
    active = get_active_config(in_memory_db._connection)
    assert active["config_id"] == config_id1

def test_create_data_upload_result(in_memory_db):
    """Test creating a data upload result"""
    conn = in_memory_db._connection
    job_id = create_job("data_upload", {}, connection=conn)
    run_id = create_processing_run(datetime.now(), "running", "f.csv", connection=conn)
    result_id = create_data_upload_result(job_id, 100, ["sales"], run_id, conn)
    res = execute_query("SELECT * FROM data_upload_results WHERE result_id = ?", conn, (result_id,))
    assert res["records_processed"] == 100

def test_create_prediction_result(in_memory_db):
    """Test creating a prediction result"""
    conn = in_memory_db._connection
    job_id = create_job("prediction", {}, connection=conn)
    model_id = str(uuid.uuid4())
    create_model_record(model_id, job_id, "/path/m.onnx", datetime.now(), connection=conn)
    result_id = create_prediction_result(job_id, model_id, "/out.csv", {"a":1}, date(2023,1,1), conn)
    res = execute_query("SELECT * FROM prediction_results WHERE result_id = ?", conn, (result_id,))
    assert res["model_id"] == model_id

def test_get_training_results_by_id(in_memory_db):
    """Test getting a single training result by ID."""
    conn = in_memory_db._connection
    job_id = create_job("training", {}, connection=conn)
    model_id = str(uuid.uuid4())
    config_id = str(uuid.uuid4())
    conn.execute("INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)", (model_id, job_id, "/path", datetime.now().isoformat()))
    conn.execute("INSERT INTO configs (config_id, config, created_at) VALUES (?, ?, ?)", (config_id, '{"a":1}', datetime.now().isoformat()))
    result_id = create_training_result(job_id, model_id, config_id, {"m":1}, 120, conn)
    retrieved = get_training_results(result_id=result_id, connection=conn)
    assert retrieved["result_id"] == result_id

def test_get_training_results_list(in_memory_db):
    """Test getting a list of recent training results."""
    conn = in_memory_db._connection
    job_id = create_job("training", {}, connection=conn)
    model_id = str(uuid.uuid4())
    config_id = str(uuid.uuid4())
    conn.execute("INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)", (model_id, job_id, "/path", datetime.now().isoformat()))
    conn.execute("INSERT INTO configs (config_id, config, created_at) VALUES (?, ?, ?)", (config_id, '{"a":1}', datetime.now().isoformat()))
    for i in range(5):
        create_training_result(job_id, model_id, config_id, {"m":i}, 100+i, conn)
    assert len(get_training_results(connection=conn)) == 5
    assert len(get_training_results(limit=2, connection=conn)) == 2

def test_get_tuning_results_by_id(in_memory_db):
    """Test getting a single tuning result by ID."""
    conn = in_memory_db._connection
    job_id = create_job("tuning", {}, connection=conn)
    config_id = str(uuid.uuid4())
    conn.execute("INSERT INTO configs (config_id, config, created_at) VALUES (?, ?, ?)", (config_id, '{"a":1}', datetime.now().isoformat()))
    result_id = create_tuning_result(job_id, config_id, {"m":1}, 300, conn)
    retrieved = get_tuning_results(result_id=result_id, connection=conn)
    assert retrieved["result_id"] == result_id

def test_get_tuning_results_list(in_memory_db):
    """Test getting a list of recent tuning results."""
    conn = in_memory_db._connection
    config_id = str(uuid.uuid4())
    conn.execute("INSERT INTO configs (config_id, config, created_at) VALUES (?, ?, ?)", (config_id, '{"a":1}', datetime.now().isoformat()))
    ids = []
    for i in range(5):
        job_id = create_job("tuning", {}, connection=conn)
        ids.append(create_tuning_result(job_id, config_id, {"m":i}, 300+i, conn))
    results = get_tuning_results(connection=conn)
    assert len(results) == 5
    assert results[0]["result_id"] == ids[4]

def test_create_and_update_processing_run(in_memory_db):
    """Test creating and updating a processing run"""
    conn = in_memory_db._connection
    start_time = datetime.now()
    run_id = create_processing_run(start_time, "running", "f.csv", connection=conn)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM processing_runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()
    assert run["status"] == "running"
    end_time = datetime.now()
    update_processing_run(run_id, "completed", end_time, conn)
    cursor.execute("SELECT * FROM processing_runs WHERE run_id = ?", (run_id,))
    updated_run = cursor.fetchone()
    assert updated_run["status"] == "completed"

def test_get_feature_dataframe(in_memory_db):
    """Test the generic get_feature_dataframe function."""
    conn = in_memory_db._connection
    from deployment.app.db.database import get_feature_dataframe

    # 1. Setup data
    try:
        conn.execute("INSERT INTO dim_multiindex_mapping (multiindex_id, barcode, artist, album, cover_type, price_category, release_type, recording_decade, release_decade, style, recording_year) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (1, '1', '1', '1', '1', '1', '1', '1', '1', '1', 1))
    except sqlite3.IntegrityError:
        pass # Already exists

    conn.execute("INSERT INTO fact_sales (multiindex_id, data_date, value) VALUES (?, ?, ?)", (1, "2023-01-15", 100))
    conn.execute("INSERT INTO report_features (multiindex_id, data_date, availability, confidence) VALUES (?, ?, ?, ?)", (1, "2023-01-15", 0.9, 0.8))
    conn.commit()

    # 2. Test fetching from fact_sales (single value column)
    sales_data = get_feature_dataframe(
        table_name="fact_sales",
        columns=["value"],
        connection=conn,
        start_date="2023-01-01",
        end_date="2023-01-31"
    )
    assert len(sales_data) == 1
    assert sales_data[0]["multiindex_id"] == 1
    assert sales_data[0]["data_date"] == "2023-01-15"
    assert sales_data[0]["value"] == 100

    # 3. Test fetching from report_features (multiple value columns)
    report_data = get_feature_dataframe(
        table_name="report_features",
        columns=["availability", "confidence"],
        connection=conn,
        start_date="2023-01-01"
    )
    assert len(report_data) == 1
    assert report_data[0]["multiindex_id"] == 1
    assert report_data[0]["availability"] == 0.9
    assert report_data[0]["confidence"] == 0.8

    # 4. Test date filtering
    no_data = get_feature_dataframe(
        table_name="fact_sales",
        columns=["value"],
        connection=conn,
        start_date="2024-01-01"
    )
    assert len(no_data) == 0

    # 5. Test invalid column name (should raise ValueError)
    with pytest.raises(ValueError):
        get_feature_dataframe(
            table_name="fact_sales",
            columns=["value; DROP TABLE users"],
            connection=conn
        )