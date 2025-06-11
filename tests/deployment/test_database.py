import sqlite3
import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from deployment.app.db.database import (
    get_db_connection, execute_query, execute_many, generate_id,
    create_job, update_job_status, get_job, list_jobs,
    create_data_upload_result, create_training_result,
    create_prediction_result, create_report_result,
    get_data_upload_result, get_training_result,
    get_prediction_result, get_report_result,
    create_processing_run, update_processing_run,
    get_or_create_multiindex_id, DatabaseError
)

# Note: No longer using unittest.TestCase. All tests are now pytest functions.
# The temp_db and temp_db_with_data fixtures from conftest.py handle setup and teardown.

# ======================================================================================
# Tests for Basic Query Functions
# ======================================================================================

def test_execute_query_select(temp_db):
    """Test execute_query for SELECT statements."""
    job_id = "test-job-1"
    now = datetime.now().isoformat()
    temp_db.execute(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "test_type", "pending", now, now)
    )
    temp_db.commit()

    result = execute_query("SELECT * FROM jobs WHERE job_id = ?", (job_id,), connection=temp_db)

    assert result is not None
    assert result["job_id"] == job_id
    assert result["status"] == "pending"

def test_execute_query_select_all(temp_db):
    """Test execute_query for SELECT multiple rows."""
    now = datetime.now().isoformat()
    job_ids = ["multi-1", "multi-2", "multi-3"]
    for job_id in job_ids:
        temp_db.execute(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "multi_test", "pending", now, now)
        )
    temp_db.commit()

    results = execute_query("SELECT * FROM jobs WHERE job_type = ?", ("multi_test",), fetchall=True, connection=temp_db)

    assert len(results) == 3
    retrieved_ids = [job["job_id"] for job in results]
    assert all(job_id in retrieved_ids for job_id in job_ids)

def test_execute_query_select_non_existent(temp_db):
    """Test execute_query for SELECT with non-existent record."""
    result = execute_query("SELECT * FROM jobs WHERE job_id = ?", ("non-existent-id",), connection=temp_db)
    assert result is None

def test_execute_query_select_empty_result(temp_db):
    """Test execute_query for SELECT returning empty result set."""
    results = execute_query("SELECT * FROM jobs WHERE job_type = ?", ("non-existent-type",), fetchall=True, connection=temp_db)
    assert len(results) == 0

def test_execute_query_insert(temp_db):
    """Test execute_query for INSERT statements."""
    job_id = "test-job-2"
    now = datetime.now().isoformat()
    
    execute_query(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "test_type", "pending", now, now),
        connection=temp_db
    )
    
    result = temp_db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    assert result is not None
    assert result["job_id"] == job_id

def test_execute_query_update(temp_db):
    """Test execute_query for UPDATE statements."""
    job_id = "test-job-3"
    now = datetime.now().isoformat()
    temp_db.execute(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "test_type", "pending", now, now)
    )
    temp_db.commit()
    
    execute_query("UPDATE jobs SET status = ? WHERE job_id = ?", ("running", job_id), connection=temp_db)
    
    result = temp_db.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    assert result["status"] == "running"

def test_execute_query_delete(temp_db):
    """Test execute_query for DELETE statements."""
    job_id = "delete-test-job"
    now = datetime.now().isoformat()
    temp_db.execute(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "test_type", "pending", now, now)
    )
    temp_db.commit()
    
    assert temp_db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone() is not None
    
    execute_query("DELETE FROM jobs WHERE job_id = ?", (job_id,), connection=temp_db)
    
    assert temp_db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone() is None

@patch('deployment.app.db.database.get_db_connection')
def test_execute_query_error(mock_get_conn):
    """Test execute_query error handling."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.execute.side_effect = sqlite3.Error("SQL syntax error")
    mock_get_conn.return_value = mock_conn
    
    with pytest.raises(DatabaseError, match="Database operation failed") as excinfo:
        execute_query("INVALID SQL QUERY")
    
    assert excinfo.value.query == "INVALID SQL QUERY"

def test_execute_many(temp_db):
    """Test execute_many functionality."""
    now = datetime.now().isoformat()
    params_list = [
        ("batch-job-1", "test_type", "pending", now, now),
        ("batch-job-2", "test_type", "pending", now, now),
        ("batch-job-3", "test_type", "pending", now, now)
    ]
    
    execute_many(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        params_list,
        connection=temp_db
    )
    
    result = temp_db.execute("SELECT COUNT(*) as count FROM jobs WHERE job_type = ?", ("test_type",)).fetchone()
    assert result["count"] >= 3

def test_execute_many_empty_list(temp_db):
    """Test execute_many with empty params list. Should not raise any errors."""
    execute_many("INSERT INTO jobs VALUES (?, ?, ?, ?, ?)", [], connection=temp_db)

@patch('deployment.app.db.database.get_db_connection')
def test_execute_many_error(mock_get_conn):
    """Test execute_many error handling."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.executemany.side_effect = sqlite3.Error("SQL error in batch")
    mock_get_conn.return_value = mock_conn
    
    params_list = [("batch-error-1",), ("batch-error-2",)]
    
    with pytest.raises(DatabaseError, match="Batch database operation failed"):
        execute_many("INVALID SQL BATCH", params_list)

# ======================================================================================
# Tests for Higher-Level Database Functions (Operations)
# ======================================================================================

# This new class replaces the old TestDatabaseOperations and uses fixtures correctly.
class TestDatabaseOperations:

    def test_create_job(self, temp_db):
        job_id = create_job("training", {"param": "value"}, connection=temp_db)
        assert job_id is not None
        
        job = get_job(job_id, connection=temp_db)
        assert job["job_type"] == "training"
        assert job["status"] == "pending"
        assert json.loads(job["parameters"])["param"] == "value"

    def test_update_job_status(self, temp_db_with_data):
        conn = temp_db_with_data["conn"]
        job_id = temp_db_with_data["job_id"]
        
        update_job_status(job_id, "running", connection=conn)
        job = get_job(job_id, connection=conn)
        assert job["status"] == "running"
    
    def test_update_job_status_with_result(self, temp_db_with_data):
        conn = temp_db_with_data["conn"]
        job_id = temp_db_with_data["job_id"]
        
        # Create a result ID for the job
        result_id = generate_id()
        
        # Update job with result_id (not result)
        update_job_status(job_id, "completed", result_id=result_id, connection=conn)
        
        # Verify job status and result_id
        job = get_job(job_id, connection=conn)
        assert job["status"] == "completed"
        assert job["result_id"] == result_id

    def test_update_job_status_with_error(self, temp_db_with_data):
        conn = temp_db_with_data["conn"]
        job_id = temp_db_with_data["job_id"]
        error_message = "Something went wrong"
        
        # Use error_message parameter (not error)
        update_job_status(job_id, "failed", error_message=error_message, connection=conn)
        
        job = get_job(job_id, connection=conn)
        assert job["status"] == "failed"
        assert job["error_message"] == error_message

    def test_update_job_status_not_found(self, temp_db, caplog):
        # The function logs a warning but doesn't raise an exception when job not found
        update_job_status("non-existent-job", "running", connection=temp_db)
        
        # Check that a warning was logged
        assert "Job with ID non-existent-job not found" in caplog.text

    def test_get_job(self, temp_db_with_data):
        conn = temp_db_with_data["conn"]
        job_id = temp_db_with_data["job_id"]
        
        job = get_job(job_id, connection=conn)
        assert job is not None
        assert job["job_id"] == job_id

    def test_list_jobs(self, temp_db_with_data):
        conn = temp_db_with_data["conn"]
        jobs = list_jobs(connection=conn)
        assert len(jobs) >= 2 # Based on temp_db_with_data

    def test_create_and_get_all_result_types(self, temp_db_with_data):
        conn = temp_db_with_data["conn"]
        job_id = temp_db_with_data["job_id"]
        
        # First create a processing run to satisfy the foreign key constraint
        processing_run_id = create_processing_run(
            start_time=datetime.now(),
            status="completed",
            cutoff_date="2023-01-01",
            source_files="file1.csv,file2.csv",
            connection=conn
        )
        
        # Data Upload - fix argument order and types
        # create_data_upload_result(job_id: str, records_processed: int, features_generated: List[str], processing_run_id: int)
        upload_id = create_data_upload_result(
            job_id=job_id, 
            records_processed=100, 
            features_generated=["feature1", "feature2"], 
            processing_run_id=processing_run_id,  # Use the created processing run ID
            connection=conn
        )
        upload_res = get_data_upload_result(upload_id, connection=conn)
        assert upload_res["records_processed"] == 100
        
        # Training
        model_id = temp_db_with_data["model_id"]
        config_id = temp_db_with_data["config_id"]
        train_id = create_training_result(
            job_id=job_id, 
            model_id=model_id,
            config_id=config_id, 
            metrics={"acc": 1}, 
            config={"param": "value"},
            duration=60,
            connection=conn
        )
        train_res = get_training_result(train_id, connection=conn)
        assert json.loads(train_res["metrics"])["acc"] == 1
        
        # Prediction
        pred_id = create_prediction_result(
            job_id=job_id, 
            model_id=model_id,
            output_path="/path/preds", 
            summary_metrics={"rows_predicted": 200},
            connection=conn
        )
        pred_res = get_prediction_result(pred_id, connection=conn)
        assert pred_res["output_path"] == "/path/preds"

        # Report
        report_id = create_report_result(
            job_id=job_id, 
            report_type="html", 
            parameters={"template": "default"},
            output_path="/path/report",
            connection=conn
        )
        report_res = get_report_result(report_id, connection=conn)
        assert report_res["report_type"] == "html"

    def test_create_and_update_processing_run(self, temp_db_with_data):
        conn = temp_db_with_data["conn"]
        
        # Fix: provide all required arguments
        run_id = create_processing_run(
            start_time=datetime.now(),
            status="data_prep", 
            cutoff_date="2023-01-01",
            source_files="file1.csv,file2.csv",
            connection=conn
        )
        assert run_id is not None
        
        update_processing_run(run_id, "completed", datetime.now(), connection=conn)
        run_res_updated = conn.execute("SELECT * FROM processing_runs WHERE run_id = ?", (run_id,)).fetchone()
        assert run_res_updated["status"] == "completed"

    def test_get_or_create_multiindex_id(self, temp_db):
        """Test the get_or_create_multiindex_id function for idempotency."""
        params = {
            "barcode": "12345",
            "artist": "Test Artist",
            "album": "Test Album",
            "cover_type": "Gatefold",
            "price_category": "A",
            "release_type": "LP",
            "recording_decade": "1980s",
            "release_decade": "1980s",
            "style": "Rock",
            "record_year": 1985
        }

        # Create new
        id1 = get_or_create_multiindex_id(**params, connection=temp_db)
        assert isinstance(id1, int)

        # Get existing
        id2 = get_or_create_multiindex_id(**params, connection=temp_db)
        assert id1 == id2

        # Verify record exists in DB
        res = temp_db.execute("SELECT * FROM dim_multiindex_mapping WHERE multiindex_id = ?", (id1,)).fetchone()
        assert res is not None
        assert res["artist"] == "Test Artist"
        assert res["record_year"] == 1985 