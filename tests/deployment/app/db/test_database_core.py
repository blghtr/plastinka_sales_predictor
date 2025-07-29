"""
Comprehensive tests for deployment.app.db.database core functionality

This test suite consolidates core database operations testing from multiple files
into a unified, well-organized structure following comprehensive testing patterns.

Testing Approach:
- Use in-memory SQLite databases for test isolation
- Mock all external dependencies using @patch decorators
- Test both success and failure scenarios for all operations
- Test connection management, transactions, and error handling
- Test CRUD operations with proper rollback verification
- Test foreign key constraints and referential integrity
- Test concurrent access patterns and connection isolation
- Integration tests for module imports and database schema

Consolidated from:
- test_database.py (core CRUD operations)
- test_database_resilience.py (error handling, edge cases)
- test_database_transactions.py (transaction safety, concurrency)
- test_foreign_key_constraints.py (FK integrity validation)

All external imports and dependencies are mocked to ensure test isolation.
"""

import json
import os
import sqlite3
import tempfile
import threading
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Import the modules under test
from deployment.app.db.database import (
    DatabaseError,
    create_job,
    create_model_record,
    create_or_get_config,
    dict_factory,
    execute_many,
    execute_query,
    generate_id,
    get_active_config,
    get_active_model,
    get_db_connection,
    get_job,
    list_jobs,
    set_config_active,
    set_model_active,
    update_job_status,
)
from deployment.app.db.schema import SCHEMA_SQL


class TestConnectionManagement:
    """Test suite for database connection handling and management."""

    def test_get_db_connection_success(self, temp_db):
        """Test that get_db_connection returns a valid connection."""
        conn = get_db_connection()

        # Verify it's a valid SQLite connection
        assert isinstance(conn, sqlite3.Connection)

        # Verify row_factory is set correctly
        assert conn.row_factory == dict_factory

        # Test that we can execute a basic query
        cursor = conn.cursor()
        cursor.execute("SELECT sqlite_version()")
        version_row = cursor.fetchone()
        assert version_row is not None
        version = version_row["sqlite_version()"]
        assert version is not None

        conn.close()

    def test_get_db_connection_error_handling(self):
        """Test that get_db_connection raises DatabaseError when connection fails."""
        with patch("pathlib.Path.mkdir", side_effect=Exception("Forced error")):
            with pytest.raises(DatabaseError) as exc_info:
                get_db_connection()

            assert "Database connection failed" in str(exc_info.value)
            assert exc_info.value.original_error is not None

    def test_dict_factory(self):
        """Test that dict_factory correctly converts rows to dictionaries."""
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("id", None, None, None, None, None, None),
            ("name", None, None, None, None, None, None),
        ]

        row = (1, "test")
        result = dict_factory(mock_cursor, row)

        assert isinstance(result, dict)
        assert result["id"] == 1
        assert result["name"] == "test"


class TestQueryExecution:
    """Test suite for basic query execution and parameter handling."""

    def test_execute_query_select(self, in_memory_db):
        """Test execute_query with SELECT operations."""
        conn = in_memory_db._connection

        # Insert test data
        conn.execute(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (
                "test-job",
                "training",
                "running",
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

        # Test SELECT query
        result = execute_query(
            "SELECT job_id, status FROM jobs WHERE job_id = ?",
            conn,
            ("test-job",),
        )

        assert result is not None
        assert result["job_id"] == "test-job"
        assert result["status"] == "running"

    def test_execute_query_insert(self, in_memory_db):
        """Test execute_query with INSERT operations."""
        conn = in_memory_db._connection

        job_id = str(uuid.uuid4())
        result = execute_query(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            conn,
            (
                job_id,
                "training",
                "pending",
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
        )

        # For INSERT operations, result should be None
        assert result is None

        # Verify the record was inserted
        verify_result = execute_query(
            "SELECT job_id FROM jobs WHERE job_id = ?", conn, (job_id,)
        )
        assert verify_result["job_id"] == job_id

    def test_execute_query_error_handling(self, in_memory_db):
        """Test execute_query handles SQL errors properly."""
        conn = in_memory_db._connection

        with pytest.raises(DatabaseError) as exc_info:
            execute_query("SELECT * FROM nonexistent_table", conn)

        assert "no such table" in str(exc_info.value).lower()

    def test_execute_many_success(self, in_memory_db):
        """Test execute_many with multiple parameter sets."""
        conn = in_memory_db._connection

        # Test data for batch insert
        jobs_data = [
            (
                str(uuid.uuid4()),
                "training",
                "pending",
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
            (
                str(uuid.uuid4()),
                "prediction",
                "pending",
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
            (
                str(uuid.uuid4()),
                "training",
                "running",
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
        ]

        # Execute batch insert
        execute_many(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            jobs_data,
            conn,
        )

        # Verify all records were inserted
        result = execute_query("SELECT COUNT(*) as count FROM jobs", conn)
        assert result["count"] == 3

    def test_execute_many_error_handling(self, in_memory_db):
        """Test execute_many handles errors in parameter data."""
        conn = in_memory_db._connection

        # Invalid parameter data (missing required fields)
        invalid_data = [
            ("job1", "training"),  # Missing required fields
            ("job2", "prediction", "pending"),  # Still missing fields
        ]

        with pytest.raises(DatabaseError):
            execute_many(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                invalid_data,
                connection=conn,
            )

    def test_execute_many_empty_params(self, in_memory_db):
        """Test execute_many with empty parameter list."""
        conn = in_memory_db._connection

        # Should not raise an error with empty params
        execute_many(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            [],
            connection=conn,
        )

        # Verify no records were inserted
        result = execute_query("SELECT COUNT(*) as count FROM jobs", conn)
        assert result["count"] == 0


class TestCRUDOperations:
    """Test suite for Create, Read, Update, Delete operations."""

    def test_generate_id(self):
        """Test ID generation utility."""
        id1 = generate_id()
        id2 = generate_id()

        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0

    def test_create_job_success(self, in_memory_db, sample_job_data):
        """Test successful job creation."""
        conn = in_memory_db._connection

        job_id = create_job(
            job_type=sample_job_data["job_type"],
            parameters=sample_job_data["parameters"],
            connection=conn,
        )

        assert isinstance(job_id, str)
        assert len(job_id) > 0

        # Verify job was created
        result = execute_query(
            "SELECT * FROM jobs WHERE job_id = ?", conn, (job_id,)
        )
        assert result["job_type"] == sample_job_data["job_type"]
        assert json.loads(result["parameters"]) == sample_job_data["parameters"]

    def test_update_job_status_success(self, in_memory_db, sample_job_data):
        """Test successful job status update."""
        conn = in_memory_db._connection

        # Create a job first
        job_id = create_job(
            job_type=sample_job_data["job_type"],
            parameters=sample_job_data["parameters"],
            connection=conn,
        )

        # Update status (function returns None, not boolean)
        update_job_status(job_id, "completed", progress=100, connection=conn)

        # Verify update
        job = execute_query(
            "SELECT status, progress FROM jobs WHERE job_id = ?",
            conn,
            (job_id,),
        )
        assert job["status"] == "completed"
        assert job["progress"] == 100

    def test_get_job_success(self, in_memory_db, sample_job_data):
        """Test successful job retrieval."""
        conn = in_memory_db._connection

        # Create a job first
        job_id = create_job(
            job_type=sample_job_data["job_type"],
            parameters=sample_job_data["parameters"],
            connection=conn,
        )

        # Retrieve job
        job = get_job(job_id, connection=conn)
        assert job is not None
        assert job["job_id"] == job_id
        assert job["job_type"] == sample_job_data["job_type"]

    def test_get_job_nonexistent(self, in_memory_db):
        """Test retrieving a non-existent job."""
        conn = in_memory_db._connection

        result = get_job("nonexistent-job-id", connection=conn)
        assert result is None

    def test_list_jobs_success(self, in_memory_db):
        """Test listing all jobs."""
        conn = in_memory_db._connection

        # Create multiple jobs
        job_ids = []
        for i in range(3):
            job_id = create_job(
                job_type="training", parameters={"test": f"job_{i}"}, connection=conn
            )
            job_ids.append(job_id)

        # List jobs
        jobs = list_jobs(connection=conn)
        assert len(jobs) == 3

        # Verify all job IDs are present
        retrieved_ids = [job["job_id"] for job in jobs]
        for job_id in job_ids:
            assert job_id in retrieved_ids


class TestModelOperations:
    """Test suite for model management operations."""

    def test_create_model_record_success(self, in_memory_db, sample_model_data):
        """Test successful model record creation."""
        conn = in_memory_db._connection

        # Create a job first (required for foreign key)
        job_id = create_job(job_type="training", parameters={}, connection=conn)
        sample_model_data["job_id"] = job_id

        create_model_record(
            model_id=sample_model_data["model_id"],
            job_id=sample_model_data["job_id"],
            model_path=sample_model_data["model_path"],
            created_at=sample_model_data["created_at"],
            metadata=sample_model_data["metadata"],
            connection=conn,
        )

        # Verify model was created
        result = execute_query(
            "SELECT * FROM models WHERE model_id = ?",
            conn,
            (sample_model_data["model_id"],),
        )
        assert result["model_id"] == sample_model_data["model_id"]
        assert result["job_id"] == sample_model_data["job_id"]
        assert result["model_path"] == sample_model_data["model_path"]

    def test_get_active_model_success(self, in_memory_db, sample_model_data):
        """Test retrieving active model."""
        conn = in_memory_db._connection

        # Create a job first
        job_id = create_job(job_type="training", parameters={}, connection=conn)
        sample_model_data["job_id"] = job_id

        # Create and activate model
        create_model_record(
            model_id=sample_model_data["model_id"],
            job_id=sample_model_data["job_id"],
            model_path=sample_model_data["model_path"],
            created_at=sample_model_data["created_at"],
            metadata=sample_model_data["metadata"],
            is_active=True,
            connection=conn,
        )

        # Get active model
        active_model = get_active_model(connection=conn)
        assert active_model is not None
        assert active_model["model_id"] == sample_model_data["model_id"]
        # get_active_model only returns model_id, model_path, metadata (not is_active field)
        assert "model_path" in active_model
        assert "metadata" in active_model

    def test_set_model_active_success(self, in_memory_db):
        """Test activating a model."""
        conn = in_memory_db._connection

        # Create a job first
        job_id = create_job(job_type="training", parameters={}, connection=conn)

        # Create multiple models
        model_ids = []
        for i in range(3):
            model_id = f"model_{i}"
            create_model_record(
                model_id=model_id,
                job_id=job_id,
                model_path=f"/path/to/model_{i}.onnx",
                created_at=datetime.now(),
                metadata={},
                is_active=False,
                connection=conn,
            )
            model_ids.append(model_id)

        # Activate second model
        result = set_model_active("model_1", deactivate_others=True, connection=conn)
        assert result is True

        # Verify only model_1 is active
        active_model = get_active_model(connection=conn)
        assert active_model["model_id"] == "model_1"


class TestConfigurationOperations:
    """Test suite for configuration management operations."""

    def test_create_or_get_config_success(self, in_memory_db, sample_config):
        """Test creating or retrieving configuration."""
        conn = in_memory_db._connection

        # Create config
        config_id = create_or_get_config(sample_config, connection=conn)
        assert isinstance(config_id, str)
        assert len(config_id) > 0

        # Verify config was created
        result = execute_query(
            "SELECT * FROM configs WHERE config_id = ?", conn, (config_id,)
        )
        assert result is not None
        assert json.loads(result["config"]) == sample_config

    def test_get_active_config_success(self, in_memory_db, sample_config):
        """Test retrieving active configuration."""
        conn = in_memory_db._connection

        # Create and activate config
        config_id = create_or_get_config(sample_config, connection=conn)
        set_config_active(config_id, connection=conn)

        # Get active config
        active_config = get_active_config(connection=conn)
        assert active_config is not None
        assert active_config["config_id"] == config_id
        # get_active_config only returns config_id and config fields (not is_active field)
        assert "config" in active_config

    def test_set_config_active_success(self, in_memory_db, sample_config):
        """Test activating a configuration."""
        conn = in_memory_db._connection

        # Create multiple configs
        config_ids = []
        for i in range(3):
            modified_config = sample_config.copy()
            modified_config["test_param"] = f"value_{i}"
            config_id = create_or_get_config(modified_config, connection=conn)
            config_ids.append(config_id)

        # Activate second config
        result = set_config_active(config_ids[1], connection=conn)
        assert result is True

        # Verify only second config is active
        active_config = get_active_config(connection=conn)
        assert active_config["config_id"] == config_ids[1]


class TestTransactionHandling:
    """Test suite for transaction management and safety."""

    def test_execute_query_transaction_commit(self, isolated_db_session):
        """Test that execute_query properly commits transactions."""
        conn = isolated_db_session["dal"]._connection
        cursor = conn.cursor()

        # Create test table
        cursor.execute(
            "CREATE TABLE test_transactions (id INTEGER PRIMARY KEY, value TEXT)"
        )
        conn.commit()

        # Test transaction commit
        execute_query(
            "INSERT INTO test_transactions (value) VALUES (?)",
            conn,
            ("test_value",),
        )

        # Verify data was committed
        cursor.execute("SELECT value FROM test_transactions WHERE id = 1")
        result = cursor.fetchone()
        assert result is not None
        value = result[0] if isinstance(result, tuple) else result["value"]
        assert value == "test_value"

        conn.close()

    def test_execute_query_transaction_rollback(self, isolated_db_session):
        """Test that execute_query properly handles rollback on error."""
        conn = isolated_db_session["dal"]._connection
        cursor = conn.cursor()

        # Create test table
        cursor.execute(
            "CREATE TABLE test_rollback (id INTEGER PRIMARY KEY, value TEXT)"
        )
        conn.commit()

        # Start a transaction and cause an error
        try:
            execute_query(
                "INSERT INTO test_rollback (id, value) VALUES (?, ?)",
                conn,
                (1,),  # Missing parameter - should cause error
            )
        except DatabaseError:
            pass  # Expected error

        # Verify nothing was inserted due to rollback
        cursor.execute("SELECT COUNT(*) FROM test_rollback")
        result = cursor.fetchone()
        count = result[0] if isinstance(result, tuple) else result["COUNT(*)"]
        assert count == 0

        conn.close()

    def test_concurrent_access_isolation(self, isolated_db_session):
        """Test that concurrent database access maintains isolation."""
        db_path = isolated_db_session["db_path"]
        results = []

        def worker(thread_id):
            # Each thread gets its own connection
            conn = sqlite3.connect(db_path)
            try:
                job_id = create_job(
                    job_type="training",
                    parameters={"thread_id": thread_id},
                    connection=conn,
                )
                results.append(job_id)
            finally:
                conn.close()

        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all operations completed successfully
        assert len(results) == 5
        assert len(set(results)) == 5  # All job IDs should be unique


class TestErrorHandling:
    """Test suite for error scenarios and resilience."""

    def test_database_connection_nonexistent_directory(self):
        """Test that database creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_path = os.path.join(temp_dir, "nonexistent")
            db_file_path = os.path.join(non_existent_path, "database", "plastinka.db")

            # Use db_path_override instead of mocking settings
            conn = get_db_connection(db_path_override=db_file_path)
            assert isinstance(conn, sqlite3.Connection)

            # Verify directory was created
            expected_db_dir = os.path.dirname(db_file_path)
            assert os.path.exists(expected_db_dir)

            conn.close()

    def test_database_error_details(self):
        """Test that DatabaseError captures detailed error information."""
        query = "SELECT * FROM nonexistent"
        params = ("param1", "param2")
        original_error = sqlite3.OperationalError("no such table: nonexistent")

        error = DatabaseError(
            message="Database operation failed",
            query=query,
            params=params,
            original_error=original_error,
        )

        assert error.message == "Database operation failed"
        assert error.query == query
        assert error.params == params
        assert error.original_error == original_error
        assert str(error) == "Database operation failed"

    def test_execute_query_connection_error(self):
        """Test that execute_query handles connection errors properly."""
        # Create a closed connection to simulate connection error
        conn = sqlite3.connect(":memory:")
        conn.close()  # Close the connection to make it invalid

        # Using a closed connection should raise DatabaseError
        with pytest.raises(DatabaseError) as exc_info:
            execute_query("SELECT 1", conn)

        assert "Connection validation failed" in str(
            exc_info.value
        ) or "Database operation failed" in str(exc_info.value)

    def test_create_job_duplicate_id(self, in_memory_db):
        """Test handling of duplicate job ID creation by inserting directly."""
        conn = in_memory_db._connection

        # Create first job normally
        job_id1 = create_job(job_type="training", parameters={}, connection=conn)

        # Manually insert a job with the same ID to create duplicate
        with pytest.raises(DatabaseError) as exc_info:
            execute_query(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, progress) VALUES (?, ?, ?, ?, ?, ?)",
                conn,
                (
                    job_id1,
                    "prediction",
                    "pending",
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    0,
                ),
            )

        # Check that it's a constraint violation
        assert "UNIQUE constraint failed" in str(
            exc_info.value
        ) or "PRIMARY KEY" in str(exc_info.value)

    def test_update_job_status_nonexistent_job(self, in_memory_db):
        """Test updating status of non-existent job."""
        conn = in_memory_db._connection

        # Function returns None (no return value) and logs warning for non-existent job
        update_job_status("nonexistent-job", "completed", connection=conn)
        # Verify job was not created
        job = execute_query(
            "SELECT * FROM jobs WHERE job_id = ?", conn, ("nonexistent-job",)
        )
        assert job is None

    def test_set_model_active_nonexistent_id(self, in_memory_db):
        """Test activating non-existent model."""
        conn = in_memory_db._connection

        result = set_model_active("nonexistent-model", connection=conn)
        assert result is False

    def test_set_config_active_nonexistent_id(self, in_memory_db):
        """Test activating non-existent config."""
        conn = in_memory_db._connection

        result = set_config_active("nonexistent-config", connection=conn)
        assert result is False


class TestForeignKeyConstraints:
    """Test suite for foreign key constraints and referential integrity."""

    def test_foreign_key_enforcement_jobs_history(self, in_memory_db):
        """Test foreign key constraint between jobs_history and jobs."""
        conn = in_memory_db._connection

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Try to insert into jobs_history without parent job - should fail
        with pytest.raises(DatabaseError):
            execute_query(
                "INSERT INTO jobs_history (job_id, previous_status, new_status, changed_at) VALUES (?, ?, ?, ?)",
                conn,
                ("nonexistent-job", "pending", "running", datetime.now().isoformat()),
            )

    def test_foreign_key_enforcement_training_results_model(self, in_memory_db):
        """Test foreign key constraint between training_results and models."""
        conn = in_memory_db._connection

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Try to insert training_results without parent model - should fail
        with pytest.raises(DatabaseError):
            execute_query(
                "INSERT INTO training_results (model_id, metrics, created_at) VALUES (?, ?, ?)",
                conn,
                (
                    "nonexistent-model",
                    json.dumps({"val_loss": 0.95}),
                    datetime.now().isoformat(),
                ),
            )

    def test_foreign_key_enforcement_training_results_config(
        self, in_memory_db, sample_config
    ):
        """Test foreign key constraint between training_results and configs."""
        conn = in_memory_db._connection

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Try to insert training_results without parent config - should fail
        with pytest.raises(DatabaseError):
            execute_query(
                "INSERT INTO training_results (config_id, metrics, created_at) VALUES (?, ?, ?)",
                conn,
                (
                    "nonexistent-config",
                    json.dumps({"val_loss": 0.95}),
                    datetime.now().isoformat(),
                ),
            )

    def test_foreign_key_cascade_delete_jobs(self, in_memory_db):
        """Test that deleting jobs should fail due to foreign key constraint (not cascade)."""
        conn = in_memory_db._connection

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Create job
        job_id = create_job(job_type="training", parameters={}, connection=conn)

        # Add job history entry (using correct table name)
        execute_query(
            "INSERT INTO job_status_history (job_id, status, status_message, updated_at) VALUES (?, ?, ?, ?)",
            conn,
            (
                job_id,
                "running",
                "Status changed to: running",
                datetime.now().isoformat(),
            ),
        )

        # Verify history entry exists
        history_before = execute_query(
            "SELECT COUNT(*) as count FROM job_status_history WHERE job_id = ?",
            conn,
            (job_id,),
        )
        assert history_before["count"] == 1

        # Delete job should fail due to foreign key constraint (schema doesn't use CASCADE)
        with pytest.raises(DatabaseError) as exc_info:
            execute_query(
                "DELETE FROM jobs WHERE job_id = ?", conn, (job_id,)
            )

        assert "FOREIGN KEY constraint failed" in str(exc_info.value)

        # Verify history entry still exists
        history_after = execute_query(
            "SELECT COUNT(*) as count FROM job_status_history WHERE job_id = ?",
            conn,
            (job_id,),
        )
        assert history_after["count"] == 1

    def test_successful_insert_with_valid_foreign_keys(
        self, in_memory_db, sample_config
    ):
        """Test successful insertion with valid foreign key references."""
        conn = in_memory_db._connection

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Create job and config
        job_id = create_job(job_type="training", parameters={}, connection=conn)
        config_id = create_or_get_config(sample_config, connection=conn)

        # Create model
        model_id = "test-model"
        create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/test/model.onnx",
            created_at=datetime.now(),
            metadata={},
            connection=conn,
        )

        # Create training result with valid foreign keys (job_id is required in schema)
        execute_query(
            "INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics) VALUES (?, ?, ?, ?, ?)",
            conn,
            (
                str(uuid.uuid4()),
                job_id,
                model_id,
                config_id,
                json.dumps({"val_loss": 0.95}),
            ),
        )

        # Verify insertion succeeded
        result = execute_query(
            "SELECT * FROM training_results WHERE model_id = ? AND config_id = ?",
            conn,
            (model_id, config_id),
        )
        assert result is not None
        assert result["model_id"] == model_id
        assert result["config_id"] == config_id


class TestIntegration:
    """Integration tests for the complete database module."""

    def test_module_imports_successfully(self):
        """Test that the database module can be imported without errors."""
        # This test verifies that all imports work correctly
        from deployment.app.db import database

        # Verify key functions are available
        assert hasattr(database, "get_db_connection")
        assert hasattr(database, "execute_query")
        assert hasattr(database, "create_job")
        assert hasattr(database, "update_job_status")
        assert hasattr(database, "create_model_record")
        assert hasattr(database, "create_or_get_config")

    def test_constants_defined(self):
        """Test that all expected constants are defined."""
        from deployment.app.db.database import DatabaseError

        # Verify exception class is properly defined
        assert issubclass(DatabaseError, Exception)

        # Test that we can create an instance
        error = DatabaseError("test message")
        assert str(error) == "test message"

    def test_schema_integration(self):
        """Test that database schema integration works correctly."""
        # Verify schema is not empty
        assert isinstance(SCHEMA_SQL, str)
        assert len(SCHEMA_SQL) > 0

        # Test that schema can be executed
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

            # Verify key tables exist
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = ["jobs", "models", "configs", "training_results"]
            for table in expected_tables:
                assert table in tables
        finally:
            conn.close()

    def test_end_to_end_workflow(self, in_memory_db, sample_config):
        """Test complete end-to-end database workflow."""
        conn = in_memory_db._connection

        # 1. Create configuration
        config_id = create_or_get_config(sample_config, connection=conn)
        assert config_id is not None

        # Set config as active so get_active_config will return it
        set_config_active(config_id, connection=conn)

        # 2. Create job
        job_id = create_job(
            job_type="training", parameters=sample_config, connection=conn
        )
        assert job_id is not None

        # 3. Update job status
        # Update status (function returns None)
        update_job_status(job_id, "running", progress=50, connection=conn)

        # 4. Create model
        model_id = "end-to-end-model"
        create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/test/model.onnx",
            created_at=datetime.now(),
            metadata={"val_loss": 0.95},
            connection=conn,
        )

        # 5. Activate model
        model_activated = set_model_active(
            model_id, deactivate_others=True, connection=conn
        )
        assert model_activated is True

        # 6. Verify final state
        final_job = get_job(job_id, connection=conn)
        assert final_job["status"] == "running"
        assert final_job["progress"] == 50

        active_model = get_active_model(connection=conn)
        assert active_model["model_id"] == model_id

        active_config = get_active_config(connection=conn)
        assert active_config["config_id"] == config_id
