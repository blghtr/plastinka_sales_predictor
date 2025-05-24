import unittest
import sqlite3
import json
import os
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Add the parent directory to sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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
from deployment.app.db.schema import init_db, SCHEMA_SQL

class TestDatabase(unittest.TestCase):
    """Test class for database.py functionality."""

    def setUp(self):
        """Set up test environment with a temporary in-memory SQLite database."""
        # Create patcher for the DB_PATH
        self.db_path_patcher = patch('deployment.app.db.database.DB_PATH', ':memory:')
        self.mock_db_path = self.db_path_patcher.start()
        
        # Create a connection for in-memory database that will be shared
        # This connection must stay open during tests to preserve the in-memory database
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Initialize schema directly on this connection
        self.cursor.executescript(SCHEMA_SQL)
        self.conn.commit()
        
        # Patch get_db_connection to return our connection
        self.get_conn_patcher = patch('deployment.app.db.database.get_db_connection', return_value=self.conn)
        self.mock_get_conn = self.get_conn_patcher.start()
    
    def tearDown(self):
        """Clean up after tests are complete."""
        # Stop patchers
        self.get_conn_patcher.stop()
        self.db_path_patcher.stop()
        
        # Close the connection
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def test_execute_query_select(self):
        """Test execute_query for SELECT statements."""
        # Arrange
        # Insert test data
        job_id = "test-job-1"
        self.cursor.execute(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "test_type", "pending", datetime.now().isoformat(), datetime.now().isoformat())
        )
        self.conn.commit()
        
        # Act
        result = execute_query("SELECT * FROM jobs WHERE job_id = ?", (job_id,), connection=self.conn)
        
        # Assert
        self.assertIsNotNone(result)
        self.assertEqual(result["job_id"], job_id)
        self.assertEqual(result["status"], "pending")
    
    def test_execute_query_select_all(self):
        """Test execute_query for SELECT multiple rows."""
        # Arrange
        # Insert test data
        now = datetime.now().isoformat()
        job_ids = ["multi-1", "multi-2", "multi-3"]
        for job_id in job_ids:
            self.cursor.execute(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, "multi_test", "pending", now, now)
            )
        self.conn.commit()
        
        # Act
        results = execute_query("SELECT * FROM jobs WHERE job_type = ?", ("multi_test",), fetchall=True, connection=self.conn)
        
        # Assert
        self.assertEqual(len(results), 3)
        retrieved_ids = [job["job_id"] for job in results]
        for job_id in job_ids:
            self.assertIn(job_id, retrieved_ids)
    
    def test_execute_query_select_non_existent(self):
        """Test execute_query for SELECT with non-existent record."""
        # Act
        result = execute_query("SELECT * FROM jobs WHERE job_id = ?", ("non-existent-id",), connection=self.conn)
        
        # Assert
        self.assertIsNone(result)
    
    def test_execute_query_select_empty_result(self):
        """Test execute_query for SELECT returning empty result set."""
        # Act 
        results = execute_query("SELECT * FROM jobs WHERE job_type = ?", ("non-existent-type",), fetchall=True, connection=self.conn)
        
        # Assert
        self.assertEqual(len(results), 0)
    
    def test_execute_query_insert(self):
        """Test execute_query for INSERT statements."""
        # Arrange
        job_id = "test-job-2"
        now = datetime.now().isoformat()
        
        # Act
        execute_query(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "test_type", "pending", now, now),
            connection=self.conn
        )
        
        # Assert - Verify insertion worked
        result = execute_query("SELECT * FROM jobs WHERE job_id = ?", (job_id,), connection=self.conn)
        self.assertIsNotNone(result)
        self.assertEqual(result["job_id"], job_id)
    
    def test_execute_query_update(self):
        """Test execute_query for UPDATE statements."""
        # Arrange
        job_id = "test-job-3"
        now = datetime.now().isoformat()
        
        # Insert test data
        execute_query(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "test_type", "pending", now, now),
            connection=self.conn
        )
        
        # Act
        execute_query(
            "UPDATE jobs SET status = ? WHERE job_id = ?",
            ("running", job_id),
            connection=self.conn
        )
        
        # Assert
        result = execute_query("SELECT status FROM jobs WHERE job_id = ?", (job_id,), connection=self.conn)
        self.assertEqual(result["status"], "running")
    
    def test_execute_query_delete(self):
        """Test execute_query for DELETE statements."""
        # Arrange
        job_id = "delete-test-job"
        now = datetime.now().isoformat()
        
        # Insert test data
        execute_query(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "test_type", "pending", now, now),
            connection=self.conn
        )
        
        # Verify insertion worked
        result = execute_query("SELECT * FROM jobs WHERE job_id = ?", (job_id,), connection=self.conn)
        self.assertIsNotNone(result)
        
        # Act
        execute_query("DELETE FROM jobs WHERE job_id = ?", (job_id,), connection=self.conn)
        
        # Assert
        result = execute_query("SELECT * FROM jobs WHERE job_id = ?", (job_id,), connection=self.conn)
        self.assertIsNone(result)
    
    @patch('deployment.app.db.database.get_db_connection')
    def test_execute_query_error(self, mock_get_conn):
        """Test execute_query error handling."""
        # Arrange
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        # Simulate database error
        mock_cursor.execute.side_effect = sqlite3.Error("SQL syntax error")
        mock_get_conn.return_value = mock_conn
        
        # Act & Assert
        with self.assertRaises(DatabaseError) as context:
            execute_query("INVALID SQL QUERY")
        
        # Verify correct error handling
        self.assertIn("Database operation failed", str(context.exception))
        self.assertEqual(context.exception.query, "INVALID SQL QUERY")
    
    def test_execute_many(self):
        """Test execute_many functionality."""
        # Arrange
        now = datetime.now().isoformat()
        params_list = [
            ("batch-job-1", "test_type", "pending", now, now),
            ("batch-job-2", "test_type", "pending", now, now),
            ("batch-job-3", "test_type", "pending", now, now)
        ]
        
        # Act
        execute_many(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            params_list,
            connection=self.conn
        )
        
        # Assert
        result = execute_query("SELECT COUNT(*) as count FROM jobs WHERE job_type = ?", ("test_type",), connection=self.conn)
        self.assertGreaterEqual(result["count"], 3)  # At least 3 inserted
    
    def test_execute_many_empty_list(self):
        """Test execute_many with empty params list."""
        # This should not raise any errors
        execute_many("INSERT INTO jobs VALUES (?, ?, ?, ?, ?)", [], connection=self.conn)
    
    @patch('deployment.app.db.database.get_db_connection')
    def test_execute_many_error(self, mock_get_conn):
        """Test execute_many error handling."""
        # Arrange
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        # Simulate database error
        mock_cursor.executemany.side_effect = sqlite3.Error("SQL error in batch")
        mock_get_conn.return_value = mock_conn
        
        params_list = [
            ("batch-error-1", "test_type", "pending", "now", "now"),
            ("batch-error-2", "test_type", "pending", "now", "now")
        ]
        
        # Act & Assert
        with self.assertRaises(DatabaseError) as context:
            execute_many("INVALID SQL BATCH", params_list)
        
        # Verify correct error handling
        self.assertIn("Batch database operation failed", str(context.exception))
        self.assertEqual(context.exception.query, "INVALID SQL BATCH")
    
    def test_generate_id(self):
        """Test generate_id produces valid UUIDs."""
        # Act
        id1 = generate_id()
        id2 = generate_id()
        
        # Assert
        self.assertIsInstance(id1, str)
        self.assertNotEqual(id1, id2)  # IDs should be different
        self.assertEqual(len(id1), 36)  # UUID length
    
    def test_create_job(self):
        """Test create_job functionality."""
        # Arrange
        job_type = "test_job"
        parameters = {"param1": "value1", "param2": 42}
        
        # Act
        job_id = create_job(job_type, parameters, connection=self.conn)
        
        # Assert
        job = execute_query(
            "SELECT * FROM jobs WHERE job_id = ?", 
            (job_id,), 
            connection=self.conn
        )
        self.assertIsNotNone(job)
        self.assertEqual(job["job_type"], job_type)
        self.assertEqual(job["status"], "pending")
        
        # Verify parameters were stored correctly
        stored_params = json.loads(job["parameters"])
        self.assertEqual(stored_params, parameters)
    
    def test_create_job_no_parameters(self):
        """Test create_job without parameters."""
        # Act
        job_id = create_job("test_job_no_params", connection=self.conn)
        
        # Assert
        job = execute_query(
            "SELECT * FROM jobs WHERE job_id = ?", 
            (job_id,), 
            connection=self.conn
        )
        self.assertIsNotNone(job)
        self.assertIsNone(job["parameters"])
    
    @patch('deployment.app.db.database.execute_query')
    def test_create_job_db_error(self, mock_execute_query):
        """Test create_job error handling."""
        # Arrange
        mock_execute_query.side_effect = DatabaseError("Insert failed", "INSERT ...", ())
        
        # Act & Assert
        with self.assertRaises(DatabaseError):
            create_job("test_job_error")
    
    def test_update_job_status(self):
        """Test update_job_status functionality."""
        # Arrange
        job_id = create_job("test_job", connection=self.conn)
        
        # Act
        update_job_status(
            job_id, 
            "running", 
            progress=50, 
            connection=self.conn
        )
        
        # Assert
        job = execute_query(
            "SELECT * FROM jobs WHERE job_id = ?", 
            (job_id,), 
            connection=self.conn
        )
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["progress"], 50)
    
    def test_update_job_status_with_result(self):
        """Test update_job_status with result_id."""
        # Arrange
        job_id = create_job("test_job", connection=self.conn)
        result_id = "test-result-1"
        
        # Act
        update_job_status(
            job_id, 
            "completed", 
            progress=100, 
            result_id=result_id, 
            connection=self.conn
        )
        
        # Assert
        job = execute_query(
            "SELECT * FROM jobs WHERE job_id = ?", 
            (job_id,), 
            connection=self.conn
        )
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["progress"], 100)
        self.assertEqual(job["result_id"], result_id)
    
    def test_update_job_status_with_error(self):
        """Test update_job_status with error_message."""
        # Arrange
        job_id = create_job("test_job", connection=self.conn)
        error_message = "Test error occurred"
        
        # Act
        update_job_status(job_id, "failed", connection=self.conn, error_message=error_message)
        
        # Assert
        job = execute_query(
            "SELECT * FROM jobs WHERE job_id = ?", 
            (job_id,), 
            connection=self.conn
        )
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error_message"], error_message)
    
    @patch('deployment.app.db.database.execute_query')
    def test_update_job_status_db_error(self, mock_execute_query):
        """Test update_job_status error handling."""
        # Arrange
        job_id = create_job("test_job", connection=self.conn)
        mock_execute_query.side_effect = DatabaseError("Update failed", "UPDATE ...", ())
        
        # Act & Assert
        with self.assertRaises(DatabaseError):
            update_job_status(job_id, "running", connection=self.conn)
    
    def test_get_job(self):
        """Test get_job functionality."""
        # Arrange
        job_type = "test_job_get"
        parameters = {"param1": "value1"}
        job_id = create_job(job_type, parameters, connection=self.conn)
        
        # Act
        job = get_job(job_id, connection=self.conn)
        
        # Assert
        self.assertIsNotNone(job)
        self.assertEqual(job["job_id"], job_id)
        self.assertEqual(job["job_type"], job_type)
    
    def test_get_job_non_existent(self):
        """Test get_job with non-existent job ID."""
        # Act
        job = get_job("non-existent-job-id", connection=self.conn)
        
        # Assert
        self.assertIsNone(job)
    
    @patch('deployment.app.db.database.execute_query')
    def test_get_job_db_error(self, mock_execute_query):
        """Test get_job error handling."""
        # Arrange
        mock_execute_query.side_effect = DatabaseError("Select failed", "SELECT ...", ())
        
        # Act & Assert
        with self.assertRaises(DatabaseError):
            get_job("test-job-db-error")
    
    def test_list_jobs(self):
        """Test list_jobs functionality."""
        # Arrange
        # Create some test jobs
        for i in range(5):
            create_job("list_test_job", {"index": i}, connection=self.conn)
        
        # Act
        jobs = list_jobs(job_type="list_test_job", connection=self.conn)
        
        # Assert
        self.assertGreaterEqual(len(jobs), 5)
        for job in jobs:
            self.assertEqual(job["job_type"], "list_test_job")
    
    def test_list_jobs_with_status_filter(self):
        """Test list_jobs with status filter."""
        # Arrange
        # Create test jobs with different statuses
        job_id1 = create_job("status_test", {"seq": 1}, connection=self.conn)
        job_id2 = create_job("status_test", {"seq": 2}, connection=self.conn)
        update_job_status(job_id1, "running", connection=self.conn)
        
        # Act
        pending_jobs = list_jobs(job_type="status_test", status="pending", connection=self.conn)
        running_jobs = list_jobs(job_type="status_test", status="running", connection=self.conn)
        
        # Assert
        self.assertEqual(len(pending_jobs), 1)
        self.assertEqual(len(running_jobs), 1)
        self.assertEqual(pending_jobs[0]["job_id"], job_id2)
        self.assertEqual(running_jobs[0]["job_id"], job_id1)
    
    def test_list_jobs_with_limit(self):
        """Test list_jobs with limit."""
        # Arrange
        # Create more jobs than the limit
        for i in range(10):
            create_job("limit_test", {"index": i}, connection=self.conn)
        
        # Act
        jobs = list_jobs(job_type="limit_test", limit=5, connection=self.conn)
        
        # Assert
        self.assertEqual(len(jobs), 5)
    
    def test_list_empty_jobs(self):
        """Test list_jobs when no matching jobs exist."""
        # Act
        jobs = list_jobs(job_type="non_existent_type", connection=self.conn)
        
        # Assert
        self.assertEqual(len(jobs), 0)
    
    @patch('deployment.app.db.database.execute_query')
    def test_list_jobs_db_error(self, mock_execute_query):
        """Test list_jobs error handling."""
        # Arrange
        mock_execute_query.side_effect = DatabaseError("Select failed", "SELECT ...", ())
        
        # Act & Assert
        with self.assertRaises(DatabaseError):
            list_jobs()
    
    def test_create_data_upload_result(self):
        """Test create_data_upload_result functionality."""
        # Arrange
        job_id = create_job("data_upload", connection=self.conn)
        records_processed = 1000
        features_generated = ["feature1", "feature2"]
        
        # Create a processing run
        run_id = create_processing_run(
            datetime.now(), "completed", "2023-01-01", "file.csv", connection=self.conn
        )
        
        # Act
        result_id = create_data_upload_result(
            job_id, records_processed, features_generated, run_id, connection=self.conn
        )
        
        # Assert
        result = get_data_upload_result(result_id, connection=self.conn)
        self.assertIsNotNone(result)
        self.assertEqual(result["job_id"], job_id)
        self.assertEqual(result["records_processed"], records_processed)
        self.assertEqual(result["processing_run_id"], run_id)
        
        # Verify features were stored correctly
        stored_features = json.loads(result["features_generated"])
        self.assertEqual(stored_features, features_generated)
    
    def test_get_data_upload_result_non_existent(self):
        """Test get_data_upload_result with non-existent result ID."""
        # Act
        result = get_data_upload_result("non-existent-result-id", connection=self.conn)
        
        # Assert
        self.assertIsNone(result)
    
    def test_create_training_result(self):
        """Test create_training_result functionality."""
        # Arrange
        job_id = create_job("training", connection=self.conn)
        model_id = "model-123"
        parameter_set_id = "param-set-123"
        metrics = {"accuracy": 0.95, "loss": 0.05}
        parameters = {"epochs": 100, "batch_size": 32}
        duration = 3600
        
        # Act
        result_id = create_training_result(
            job_id, model_id, parameter_set_id, metrics, parameters, duration, connection=self.conn
        )
        
        # Assert
        result = get_training_result(result_id, connection=self.conn)
        self.assertIsNotNone(result)
        self.assertEqual(result["job_id"], job_id)
        self.assertEqual(result["model_id"], model_id)
        self.assertEqual(result["parameter_set_id"], parameter_set_id)
        self.assertEqual(result["duration"], duration)
        
        # Verify metrics and parameters were stored correctly
        stored_metrics = json.loads(result["metrics"])
        stored_params = json.loads(result["parameters"])
        self.assertEqual(stored_metrics, metrics)
        self.assertEqual(stored_params, parameters)
    
    def test_get_training_result_non_existent(self):
        """Test get_training_result with non-existent result ID."""
        # Act
        result = get_training_result("non-existent-result-id", connection=self.conn)
        
        # Assert
        self.assertIsNone(result)
    
    def test_create_prediction_result(self):
        """Test create_prediction_result functionality."""
        # Arrange
        job_id = create_job("prediction", connection=self.conn)
        model_id = "model-123"
        output_path = "/path/to/predictions.csv"
        summary_metrics = {"mse": 0.01, "mae": 0.05}
        
        # Act
        result_id = create_prediction_result(
            job_id, model_id, output_path, summary_metrics, connection=self.conn
        )
        
        # Assert
        result = get_prediction_result(result_id, connection=self.conn)
        self.assertIsNotNone(result)
        self.assertEqual(result["job_id"], job_id)
        self.assertEqual(result["model_id"], model_id)
        self.assertEqual(result["output_path"], output_path)
        
        # Verify metrics were stored correctly
        stored_metrics = json.loads(result["summary_metrics"])
        self.assertEqual(stored_metrics, summary_metrics)
    
    def test_get_prediction_result_non_existent(self):
        """Test get_prediction_result with non-existent result ID."""
        # Act
        result = get_prediction_result("non-existent-result-id", connection=self.conn)
        
        # Assert
        self.assertIsNone(result)
    
    def test_create_report_result(self):
        """Test create_report_result functionality."""
        # Arrange
        job_id = create_job("report", connection=self.conn)
        report_type = "sales_analysis"
        parameters = {"start_date": "2023-01-01", "end_date": "2023-12-31"}
        output_path = "/path/to/report.pdf"
        
        # Act
        result_id = create_report_result(
            job_id, report_type, parameters, output_path, connection=self.conn
        )
        
        # Assert
        result = get_report_result(result_id, connection=self.conn)
        self.assertIsNotNone(result)
        self.assertEqual(result["job_id"], job_id)
        self.assertEqual(result["report_type"], report_type)
        self.assertEqual(result["output_path"], output_path)
        
        # Verify parameters were stored correctly
        stored_params = json.loads(result["parameters"])
        self.assertEqual(stored_params, parameters)
    
    def test_get_report_result_non_existent(self):
        """Test get_report_result with non-existent result ID."""
        # Act
        result = get_report_result("non-existent-result-id", connection=self.conn)
        
        # Assert
        self.assertIsNone(result)
    
    def test_create_processing_run(self):
        """Test create_processing_run functionality."""
        # Arrange
        start_time = datetime.now()
        status = "running"
        cutoff_date = "2023-01-01"
        source_files = "file1.csv,file2.csv"
        
        # Act
        run_id = create_processing_run(
            start_time, status, cutoff_date, source_files, connection=self.conn
        )
        
        # Assert
        result = execute_query("SELECT * FROM processing_runs WHERE run_id = ?", (run_id,), connection=self.conn)
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], status)
        self.assertEqual(result["cutoff_date"], cutoff_date)
        self.assertEqual(result["source_files"], source_files)
        self.assertEqual(result["start_time"], start_time.isoformat())
        self.assertIsNone(result["end_time"])
    
    def test_create_processing_run_with_end_time(self):
        """Test create_processing_run with end_time."""
        # Arrange
        start_time = datetime.now()
        end_time = datetime.now()
        status = "completed"
        cutoff_date = "2023-01-01"
        source_files = "file1.csv,file2.csv"
        
        # Act
        run_id = create_processing_run(
            start_time, status, cutoff_date, source_files, end_time, connection=self.conn
        )
        
        # Assert
        result = execute_query("SELECT * FROM processing_runs WHERE run_id = ?", (run_id,), connection=self.conn)
        self.assertIsNotNone(result)
        self.assertEqual(result["end_time"], end_time.isoformat())
    
    def test_update_processing_run(self):
        """Test update_processing_run functionality."""
        # Arrange
        start_time = datetime.now()
        run_id = create_processing_run(
            start_time, "running", "2023-01-01", "file.csv", connection=self.conn
        )
        end_time = datetime.now()
        
        # Act
        update_processing_run(run_id, "completed", end_time, connection=self.conn)
        
        # Assert
        result = execute_query("SELECT * FROM processing_runs WHERE run_id = ?", (run_id,), connection=self.conn)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["end_time"], end_time.isoformat())
    
    def test_update_processing_run_status_only(self):
        """Test update_processing_run with status update only."""
        # Arrange
        start_time = datetime.now()
        run_id = create_processing_run(
            start_time, "running", "2023-01-01", "file.csv", connection=self.conn
        )
        
        # Act
        update_processing_run(run_id, "failed", connection=self.conn)
        
        # Assert
        result = execute_query("SELECT * FROM processing_runs WHERE run_id = ?", (run_id,), connection=self.conn)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["end_time"])
    
    def test_get_or_create_multiindex_id_new(self):
        """Test get_or_create_multiindex_id creates a new ID when not found."""
        # Arrange
        barcode = "1234567890"
        artist = "Test Artist"
        album = "Test Album"
        cover_type = "CD"
        price_category = "Standard"
        release_type = "Studio"
        recording_decade = "2010s"
        release_decade = "2020s"
        style = "Rock"
        record_year = 2019
        
        # Act
        multiindex_id = get_or_create_multiindex_id(
            barcode, artist, album, cover_type, price_category,
            release_type, recording_decade, release_decade, style, record_year,
            connection=self.conn
        )
        
        # Assert
        result = execute_query(
            "SELECT * FROM dim_multiindex_mapping WHERE multiindex_id = ?", 
            (multiindex_id,),
            connection=self.conn
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["barcode"], barcode)
        self.assertEqual(result["artist"], artist)
        self.assertEqual(result["album"], album)
        self.assertEqual(result["record_year"], record_year)
    
    def test_get_or_create_multiindex_id_existing(self):
        """Test get_or_create_multiindex_id returns existing ID when found."""
        # Arrange
        barcode = "0987654321"
        artist = "Existing Artist"
        album = "Existing Album"
        cover_type = "Vinyl"
        price_category = "Premium"
        release_type = "Live"
        recording_decade = "2000s"
        release_decade = "2010s"
        style = "Jazz"
        record_year = 2008
        
        # Create record first
        first_id = get_or_create_multiindex_id(
            barcode, artist, album, cover_type, price_category,
            release_type, recording_decade, release_decade, style, record_year,
            connection=self.conn
        )
        
        # Act
        second_id = get_or_create_multiindex_id(
            barcode, artist, album, cover_type, price_category,
            release_type, recording_decade, release_decade, style, record_year,
            connection=self.conn
        )
        
        # Assert
        self.assertEqual(first_id, second_id)


if __name__ == '__main__':
    unittest.main() 