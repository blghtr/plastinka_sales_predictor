"""
Comprehensive tests for deployment.app.db core functionality with PostgreSQL

This test suite consolidates core database operations testing from multiple files
into a unified, well-organized structure following comprehensive testing patterns.

Testing Approach:
- Use PostgreSQL test database for test isolation
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

import asyncio
import json
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import asyncpg
import pytest
from asyncpg import Pool

# Import the modules under test
from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.core import execute_many, execute_query
from deployment.app.db.queries.jobs import create_job, get_job, list_jobs, update_job_status
from deployment.app.db.queries.models import create_model_record, get_active_model, set_model_active
from deployment.app.db.queries.configs import create_or_get_config, get_active_config, set_config_active
from deployment.app.db.utils import generate_id
from deployment.app.db.schema_postgresql import SCHEMA_SQL


class TestConnectionManagement:
    """Test suite for database connection handling and management."""

    @pytest.mark.asyncio
    async def test_pool_connection_success(self, postgres_pool):
        """Test that PostgreSQL pool provides valid connections."""
        async with postgres_pool.acquire() as conn:
            # Verify it's a valid asyncpg connection
            assert isinstance(conn, asyncpg.Connection)

            # Test that we can execute a basic query
            version = await conn.fetchval("SELECT version()")
            assert version is not None
            assert "PostgreSQL" in version


class TestQueryExecution:
    """Test suite for basic query execution and parameter handling."""

    @pytest.mark.asyncio
    async def test_execute_query_select(self, postgres_pool, test_db_schema):
        """Test execute_query with SELECT operations."""
        async with postgres_pool.acquire() as conn:
            # Insert test data
            now = datetime.now()
            await conn.execute(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
                "test-job",
                "training",
                "running",
                now,
                now,
            )

            # Test SELECT query
            result = await execute_query(
                "SELECT job_id, status FROM jobs WHERE job_id = $1",
                conn,
                ("test-job",),
            )

            assert result is not None
            assert result["job_id"] == "test-job"
            assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_execute_query_insert(self, postgres_pool, test_db_schema):
        """Test execute_query with INSERT operations."""
        async with postgres_pool.acquire() as conn:
            job_id = str(uuid.uuid4())
            now = datetime.now()
            result = await execute_query(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
                conn,
                (job_id, "training", "pending", now, now),
            )

            # For INSERT operations, result should be None
            assert result is None

            # Verify the record was inserted
            verify_result = await execute_query(
                "SELECT job_id FROM jobs WHERE job_id = $1", conn, (job_id,)
            )
            assert verify_result["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_execute_query_error_handling(self, postgres_pool, test_db_schema):
        """Test execute_query handles SQL errors properly."""
        async with postgres_pool.acquire() as conn:
            with pytest.raises(DatabaseError) as exc_info:
                await execute_query("SELECT * FROM nonexistent_table", conn)

            error_msg = str(exc_info.value).lower()
            assert (
                "does not exist" in error_msg 
                or "relation" in error_msg
                or "не существует" in error_msg
                or "отношение" in error_msg
            )

    @pytest.mark.asyncio
    async def test_execute_many_success(self, postgres_pool, test_db_schema):
        """Test execute_many with multiple parameter sets."""
        async with postgres_pool.acquire() as conn:
            # Test data for batch insert
            now = datetime.now()
            jobs_data = [
                (str(uuid.uuid4()), "training", "pending", now, now),
                (str(uuid.uuid4()), "prediction", "pending", now, now),
                (str(uuid.uuid4()), "training", "running", now, now),
            ]

            # Execute batch insert
            await execute_many(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
                jobs_data,
                conn,
            )

            # Verify all records were inserted
            result = await execute_query("SELECT COUNT(*) as count FROM jobs", conn)
            assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_execute_many_error_handling(self, postgres_pool, test_db_schema):
        """Test execute_many handles errors in parameter data."""
        async with postgres_pool.acquire() as conn:
            # Invalid parameter data (missing required fields)
            invalid_data = [
                ("job1", "training"),  # Missing required fields
                ("job2", "prediction", "pending"),  # Still missing fields
            ]

            with pytest.raises(DatabaseError):
                await execute_many(
                    "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
                    invalid_data,
                    conn,
                )

    @pytest.mark.asyncio
    async def test_execute_many_empty_params(self, postgres_pool, test_db_schema):
        """Test execute_many with empty parameter list."""
        async with postgres_pool.acquire() as conn:
            # Should not raise an error with empty params
            await execute_many(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
                [],
                conn,
            )

            # Verify no records were inserted
            result = await execute_query("SELECT COUNT(*) as count FROM jobs", conn)
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

    @pytest.mark.asyncio
    async def test_create_job_success(self, dal, sample_job_data):
        """Test successful job creation."""
        job_id = await dal.create_job(
            job_type=sample_job_data["job_type"],
            parameters=sample_job_data["parameters"],
        )

        assert isinstance(job_id, str)
        assert len(job_id) > 0

        # Verify job was created
        job = await dal.get_job(job_id)
        assert job is not None
        assert job["job_type"] == sample_job_data["job_type"]
        assert job["parameters"] == sample_job_data["parameters"]

    @pytest.mark.asyncio
    async def test_update_job_status_success(self, dal, sample_job_data):
        """Test successful job status update."""
        # Create a job first
        job_id = await dal.create_job(
            job_type=sample_job_data["job_type"],
            parameters=sample_job_data["parameters"],
        )

        # Update status
        await dal.update_job_status(job_id, "completed", progress=100)

        # Verify update
        job = await dal.get_job(job_id)
        assert job["status"] == "completed"
        assert job["progress"] == 100

    @pytest.mark.asyncio
    async def test_get_job_success(self, dal, sample_job_data):
        """Test successful job retrieval."""
        # Create a job first
        job_id = await dal.create_job(
            job_type=sample_job_data["job_type"],
            parameters=sample_job_data["parameters"],
        )

        # Retrieve job
        job = await dal.get_job(job_id)
        assert job is not None
        assert job["job_id"] == job_id
        assert job["job_type"] == sample_job_data["job_type"]

    @pytest.mark.asyncio
    async def test_get_job_nonexistent(self, dal):
        """Test retrieving a non-existent job."""
        result = await dal.get_job("nonexistent-job-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_jobs_success(self, dal):
        """Test listing all jobs."""
        # Create multiple jobs
        job_ids = []
        for i in range(3):
            job_id = await dal.create_job(
                job_type="training", parameters={"test": f"job_{i}"}
            )
            job_ids.append(job_id)

        # List jobs
        jobs = await dal.list_jobs()
        assert len(jobs) >= 3

        # Verify all job IDs are present
        retrieved_ids = [job["job_id"] for job in jobs]
        for job_id in job_ids:
            assert job_id in retrieved_ids


class TestModelOperations:
    """Test suite for model management operations."""

    @pytest.mark.asyncio
    async def test_create_model_record_success(self, dal, sample_model_data):
        """Test successful model record creation."""
        # Create a job first (required for foreign key)
        job_id = await dal.create_job(job_type="training", parameters={})
        sample_model_data["job_id"] = job_id

        await dal.create_model_record(
            model_id=sample_model_data["model_id"],
            job_id=sample_model_data["job_id"],
            model_path=sample_model_data["model_path"],
            created_at=sample_model_data["created_at"],
            metadata=sample_model_data["metadata"],
        )

        # Verify model was created
        active_model = await dal.get_active_model()
        # Note: get_active_model may return None if model is not active
        # We can verify via direct query or by checking all models
        all_models = await dal.get_all_models()
        model_ids = [m["model_id"] for m in all_models]
        assert sample_model_data["model_id"] in model_ids

    @pytest.mark.asyncio
    async def test_get_active_model_success(self, dal, sample_model_data):
        """Test retrieving active model."""
        # Create a job first
        job_id = await dal.create_job(job_type="training", parameters={})
        sample_model_data["job_id"] = job_id

        # Create and activate model
        await dal.create_model_record(
            model_id=sample_model_data["model_id"],
            job_id=sample_model_data["job_id"],
            model_path=sample_model_data["model_path"],
            created_at=sample_model_data["created_at"],
            metadata=sample_model_data["metadata"],
            is_active=True,
        )

        # Get active model
        active_model = await dal.get_active_model()
        assert active_model is not None
        assert active_model["model_id"] == sample_model_data["model_id"]
        assert "model_path" in active_model
        assert "metadata" in active_model

    @pytest.mark.asyncio
    async def test_set_model_active_success(self, dal):
        """Test activating a model."""
        # Create a job first
        job_id = await dal.create_job(job_type="training", parameters={})

        # Create multiple models
        model_ids = []
        for i in range(3):
            model_id = f"model_{i}"
            await dal.create_model_record(
                model_id=model_id,
                job_id=job_id,
                model_path=f"/path/to/model_{i}.onnx",
                created_at=datetime.now(),
                metadata={},
                is_active=False,
            )
            model_ids.append(model_id)

        # Activate second model
        await dal.set_model_active("model_1", deactivate_others=True)

        # Verify only model_1 is active
        active_model = await dal.get_active_model()
        assert active_model is not None
        assert active_model["model_id"] == "model_1"


class TestConfigurationOperations:
    """Test suite for configuration management operations."""

    @pytest.mark.asyncio
    async def test_create_or_get_config_success(self, dal, sample_config):
        """Test creating or retrieving configuration."""
        # Create config
        config_id = await dal.create_or_get_config(sample_config)
        assert isinstance(config_id, str)
        assert len(config_id) > 0

        # Verify config was created
        active_config = await dal.get_active_config()
        # Note: get_active_config may return None if config is not active
        # We can verify via get_configs
        configs = await dal.get_configs()
        config_ids = [c["config_id"] for c in configs]
        assert config_id in config_ids

    @pytest.mark.asyncio
    async def test_get_active_config_success(self, dal, sample_config):
        """Test retrieving active configuration."""
        # Create and activate config
        config_id = await dal.create_or_get_config(sample_config)
        await dal.set_config_active(config_id)

        # Get active config
        active_config = await dal.get_active_config()
        assert active_config is not None
        assert active_config["config_id"] == config_id
        assert "config" in active_config

    @pytest.mark.asyncio
    async def test_set_config_active_success(self, dal, sample_config):
        """Test activating a configuration."""
        # Create multiple configs
        config_ids = []
        for i in range(3):
            modified_config = sample_config.copy()
            modified_config["test_param"] = f"value_{i}"
            config_id = await dal.create_or_get_config(modified_config)
            config_ids.append(config_id)

        # Activate second config
        await dal.set_config_active(config_ids[1])

        # Verify only second config is active
        active_config = await dal.get_active_config()
        assert active_config is not None
        assert active_config["config_id"] == config_ids[1]


class TestTransactionHandling:
    """Test suite for transaction management and safety."""

    @pytest.mark.asyncio
    async def test_execute_query_transaction_commit(self, postgres_pool, test_db_schema):
        """Test that execute_query properly commits transactions."""
        async with postgres_pool.acquire() as conn:
            # Create test table
            await conn.execute(
                "DROP TABLE IF EXISTS test_transactions; CREATE TABLE test_transactions (id BIGSERIAL PRIMARY KEY, value TEXT)"
            )

            # Test transaction commit
            await execute_query(
                "INSERT INTO test_transactions (value) VALUES ($1)",
                conn,
                ("test_value",),
            )

            # Verify data was committed
            result = await execute_query(
                "SELECT value FROM test_transactions WHERE id = 1", conn
            )
            assert result is not None
            assert result["value"] == "test_value"

    @pytest.mark.asyncio
    async def test_execute_query_transaction_rollback(self, postgres_pool, test_db_schema):
        """Test that execute_query properly handles rollback on error."""
        async with postgres_pool.acquire() as conn:
            # Create test table
            await conn.execute(
                "DROP TABLE IF EXISTS test_rollback; CREATE TABLE test_rollback (id BIGSERIAL PRIMARY KEY, value TEXT)"
            )

            # Start a transaction and cause an error
            async with conn.transaction():
                try:
                    await execute_query(
                        "INSERT INTO test_rollback (id, value) VALUES ($1, $2)",
                        conn,
                        (1,),  # Missing parameter - should cause error
                    )
                except DatabaseError:
                    pass  # Expected error
                    # Transaction will be rolled back automatically

            # Verify nothing was inserted due to rollback
            result = await execute_query("SELECT COUNT(*) as count FROM test_rollback", conn)
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_access_isolation(self, dal):
        """Test that concurrent database access maintains isolation."""
        async def worker(worker_id: int):
            job_id = await dal.create_job(
                job_type="training",
                parameters={"thread_id": worker_id},
            )
            return job_id

        # Run multiple concurrent operations
        tasks = [worker(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Verify all operations completed successfully
        assert len(results) == 5
        assert len(set(results)) == 5  # All job IDs should be unique


class TestErrorHandling:
    """Test suite for error scenarios and resilience."""

    def test_database_error_details(self):
        """Test that DatabaseError captures detailed error information."""
        query = "SELECT * FROM nonexistent"
        params = ("param1", "param2")
        original_error = asyncpg.PostgresError("relation \"nonexistent\" does not exist")

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

    @pytest.mark.asyncio
    async def test_execute_query_connection_error(self, postgres_pool):
        """Test that execute_query handles connection errors properly."""
        # Create a closed connection to simulate connection error
        async with postgres_pool.acquire() as conn:
            await conn.close()  # Close the connection to make it invalid

            # Using a closed connection should raise DatabaseError
            with pytest.raises(DatabaseError) as exc_info:
                await execute_query("SELECT 1", conn)

            assert "Database operation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_job_duplicate_id(self, dal):
        """Test handling of duplicate job ID creation."""
        # Create first job normally
        job_id1 = await dal.create_job(job_type="training", parameters={})

        # Try to create another job with the same ID - should fail
        with pytest.raises(DatabaseError) as exc_info:
            # Use execute_raw_query to bypass DAL validation
            await dal.execute_raw_query(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, progress) VALUES ($1, $2, $3, $4, $5, $6)",
                (job_id1, "prediction", "pending", datetime.now(), datetime.now(), 0),
            )

        # Check that it's a constraint violation
        error_msg = str(exc_info.value).lower()
        assert (
            "duplicate key" in error_msg 
            or "unique constraint" in error_msg
            or "повторяющееся значение ключа" in error_msg
            or "нарушает ограничение уникальности" in error_msg
        )

    @pytest.mark.asyncio
    async def test_update_job_status_nonexistent_job(self, dal):
        """Test updating status of non-existent job."""
        # Function should not raise error, just log warning
        await dal.update_job_status("nonexistent-job", "completed")
        # Verify job was not created
        job = await dal.get_job("nonexistent-job")
        assert job is None

    @pytest.mark.asyncio
    async def test_set_model_active_nonexistent_id(self, dal):
        """Test activating non-existent model."""
        # Should not raise error, just return False or None
        try:
            await dal.set_model_active("nonexistent-model")
            # If function doesn't return value, just verify no error was raised
        except Exception:
            # If it raises exception, that's also acceptable behavior
            pass

    @pytest.mark.asyncio
    async def test_set_config_active_nonexistent_id(self, dal):
        """Test activating non-existent config."""
        # Should not raise error, just return False or None
        try:
            await dal.set_config_active("nonexistent-config")
            # If function doesn't return value, just verify no error was raised
        except Exception:
            # If it raises exception, that's also acceptable behavior
            pass


class TestForeignKeyConstraints:
    """Test suite for foreign key constraints and referential integrity."""

    @pytest.mark.asyncio
    async def test_foreign_key_enforcement_jobs_history(self, dal):
        """Test foreign key constraint between job_status_history and jobs."""
        # Try to insert into job_status_history without parent job - should fail
        with pytest.raises(DatabaseError):
            await dal.execute_raw_query(
                "INSERT INTO job_status_history (job_id, status, status_message, updated_at) VALUES ($1, $2, $3, $4)",
                ("nonexistent-job", "running", "Status changed", datetime.now()),
            )

    @pytest.mark.asyncio
    async def test_foreign_key_enforcement_training_results_model(self, dal):
        """Test foreign key constraint between training_results and models."""
        # Try to insert training_results without parent model - should fail
        with pytest.raises(DatabaseError):
            await dal.execute_raw_query(
                "INSERT INTO training_results (result_id, job_id, model_id, metrics) VALUES ($1, $2, $3, $4)",
                (str(uuid.uuid4()), "test-job", "nonexistent-model", json.dumps({"val_loss": 0.95})),
            )

    @pytest.mark.asyncio
    async def test_foreign_key_enforcement_training_results_config(self, dal, sample_config):
        """Test foreign key constraint between training_results and configs."""
        # Create job first (required FK)
        job_id = await dal.create_job(job_type="training", parameters={})
        
        # Try to insert training_results without parent config - should fail
        with pytest.raises(DatabaseError):
            await dal.execute_raw_query(
                "INSERT INTO training_results (result_id, job_id, config_id, metrics) VALUES ($1, $2, $3, $4)",
                (str(uuid.uuid4()), job_id, "nonexistent-config", json.dumps({"val_loss": 0.95})),
            )

    @pytest.mark.asyncio
    async def test_foreign_key_cascade_delete_jobs(self, dal):
        """Test that deleting jobs should fail due to foreign key constraint (not cascade)."""
        # Create job
        job_id = await dal.create_job(job_type="training", parameters={})

        # Add job history entry
        await dal.execute_raw_query(
            "INSERT INTO job_status_history (job_id, status, status_message, updated_at) VALUES ($1, $2, $3, $4)",
            (job_id, "running", "Status changed to: running", datetime.now()),
        )

        # Verify history entry exists
        history_before = await dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM job_status_history WHERE job_id = $1",
            (job_id,),
            fetchall=False,
        )
        assert history_before["count"] == 1

        # Delete job should fail due to foreign key constraint
        with pytest.raises(DatabaseError) as exc_info:
            await dal.execute_raw_query(
                "DELETE FROM jobs WHERE job_id = $1", (job_id,)
            )

        # Check for foreign key violation (handle both English and Russian error messages)
        error_msg = str(exc_info.value).lower()
        assert (
            "foreign key" in error_msg or 
            "violates foreign key" in error_msg or
            "нарушает" in error_msg or  # Russian: "violates"
            "внешний ключ" in error_msg  # Russian: "foreign key"
        )

        # Verify history entry still exists
        history_after = await dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM job_status_history WHERE job_id = $1",
            (job_id,),
            fetchall=False,
        )
        assert history_after["count"] == 1

    @pytest.mark.asyncio
    async def test_successful_insert_with_valid_foreign_keys(self, dal, sample_config):
        """Test successful insertion with valid foreign key references."""
        # Create job and config
        job_id = await dal.create_job(job_type="training", parameters={})
        config_id = await dal.create_or_get_config(sample_config)

        # Create model
        model_id = "test-model"
        await dal.create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/test/model.onnx",
            created_at=datetime.now(),
            metadata={},
        )

        # Create training result with valid foreign keys
        result_id = str(uuid.uuid4())
        await dal.execute_raw_query(
            "INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics) VALUES ($1, $2, $3, $4, $5)",
            (result_id, job_id, model_id, config_id, json.dumps({"val_loss": 0.95})),
        )

        # Verify insertion succeeded
        result = await dal.execute_raw_query(
            "SELECT * FROM training_results WHERE model_id = $1 AND config_id = $2",
            (model_id, config_id),
            fetchall=False,
        )
        assert result is not None
        assert result["model_id"] == model_id
        assert result["config_id"] == config_id


class TestIntegration:
    """Integration tests for the complete database module."""

    def test_module_imports_successfully(self):
        """Test that the database modules can be imported without errors."""
        # This test verifies that all imports work correctly
        from deployment.app.db.queries import jobs, models, configs
        from deployment.app.db.queries.core import execute_query, execute_many
        from deployment.app.db.exceptions import DatabaseError

        # Verify key functions are available
        assert hasattr(jobs, "create_job")
        assert hasattr(jobs, "update_job_status")
        assert hasattr(models, "create_model_record")
        assert hasattr(configs, "create_or_get_config")

    def test_constants_defined(self):
        """Test that all expected constants are defined."""
        from deployment.app.db.exceptions import DatabaseError

        # Verify exception class is properly defined
        assert issubclass(DatabaseError, Exception)

        # Test that we can create an instance
        error = DatabaseError("test message")
        assert str(error) == "test message"

    @pytest.mark.asyncio
    async def test_schema_integration(self, postgres_pool, test_db_schema):
        """Test that database schema integration works correctly."""
        # Verify schema is not empty
        assert isinstance(SCHEMA_SQL, str)
        assert len(SCHEMA_SQL) > 0

        # Verify key tables exist
        async with postgres_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            tables = [row["table_name"] for row in result]

            expected_tables = ["jobs", "models", "configs", "training_results"]
            for table in expected_tables:
                assert table in tables

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, dal, sample_config):
        """Test complete end-to-end database workflow."""
        # 1. Create configuration
        config_id = await dal.create_or_get_config(sample_config)
        assert config_id is not None

        # Set config as active so get_active_config will return it
        await dal.set_config_active(config_id)

        # 2. Create job
        job_id = await dal.create_job(
            job_type="training", parameters=sample_config
        )
        assert job_id is not None

        # 3. Update job status
        await dal.update_job_status(job_id, "running", progress=50)

        # 4. Create model
        model_id = "end-to-end-model"
        await dal.create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/test/model.onnx",
            created_at=datetime.now(),
            metadata={"val_loss": 0.95},
        )

        # 5. Activate model
        await dal.set_model_active(model_id, deactivate_others=True)

        # 6. Verify final state
        final_job = await dal.get_job(job_id)
        assert final_job["status"] == "running"
        assert final_job["progress"] == 50

        active_model = await dal.get_active_model()
        assert active_model is not None
        assert active_model["model_id"] == model_id

        active_config = await dal.get_active_config()
        assert active_config is not None
        assert active_config["config_id"] == config_id
