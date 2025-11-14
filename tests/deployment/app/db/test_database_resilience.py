"""
Tests for database resilience and error handling.

This module contains tests that verify the resilience of the database module,
focusing on handling error conditions, edge cases, and recovery mechanisms:

1. Error handling and propagation
2. Handling of invalid inputs
3. Deletion operations with constraints
4. Testing non-existent resources
5. Simulation of concurrent access issues
6. Handling database connection failures

These tests ensure the system remains stable and recovers gracefully from
various error conditions and edge cases.
"""

import os
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import asyncpg
import pytest

from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.jobs import create_job, update_job_status
from deployment.app.db.queries.models import (
    create_model_record,
    delete_model_record_and_file,
    delete_models_by_ids,
    get_best_model_by_metric,
    set_model_active,
)
from deployment.app.db.queries.configs import (
    create_or_get_config,
    delete_configs_by_ids,
    get_best_config_by_metric,
    set_config_active,
)
from deployment.app.db.queries.results import create_training_result
from deployment.app.db.queries.core import execute_many, execute_query

# =============================================
# Resilience tests
# =============================================


@pytest.mark.asyncio
async def test_database_error_handling():
    """Test that DatabaseError properly captures error details"""
    # Create a DatabaseError with detailed information
    query = "SELECT * FROM nonexistent"
    params = ("param1", "param2")
    original_error = asyncpg.PostgresError("relation \"nonexistent\" does not exist")

    error = DatabaseError(
        message="Database operation failed",
        query=query,
        params=params,
        original_error=original_error,
    )

    # Verify error properties
    assert error.message == "Database operation failed"
    assert error.query == query
    assert error.params == params
    assert error.original_error == original_error
    assert str(error) == "Database operation failed"


@pytest.mark.asyncio
async def test_execute_query_connection_error(dal):
    """Test that execute_query handles connection errors"""
    async with dal._pool.acquire() as conn:
        # Create a mock connection that raises an error
        mock_conn = MagicMock(spec=asyncpg.Connection)
        mock_conn.fetchrow.side_effect = asyncpg.PostgresError("Connection failed")

        # Function should propagate the error
        with pytest.raises(DatabaseError) as exc_info:
            await execute_query("SELECT 1", mock_conn)

        # Verify error message
        assert "Database operation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_many_with_connection_error(dal):
    """Test that execute_many handles connection errors"""
    async with dal._pool.acquire() as conn:
        # Create a mock connection that raises an error
        mock_conn = MagicMock(spec=asyncpg.Connection)
        mock_conn.execute.side_effect = asyncpg.PostgresError("Connection failed")

        # Function should propagate the error
        with pytest.raises(DatabaseError) as exc_info:
            await execute_many("INSERT INTO table VALUES ($1)", [("value1",), ("value2",)], mock_conn)

        # Verify error message (check for batch operation error message)
        error_msg = str(exc_info.value)
        assert "Database operation failed" in error_msg or "Batch database operation failed" in error_msg


@pytest.mark.asyncio
async def test_create_job_duplicate_id(dal):
    """Test creating a job with a duplicate ID"""
    async with dal._pool.acquire() as conn:
        # Create first job
        job_id_1 = await create_job("training", {}, connection=conn)
        assert job_id_1 is not None

        # Try to create second job with same ID (should fail)
        # Note: In PostgreSQL, we can't easily force duplicate IDs since they're generated
        # This test verifies that unique constraint works
        # conn.execute raises asyncpg.exceptions.UniqueViolationError directly, not DatabaseError
        with pytest.raises(asyncpg.exceptions.UniqueViolationError) as exc_info:
            # Try to insert directly with same ID
            await conn.execute(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
                job_id_1, "prediction", "pending", datetime.now(), datetime.now()
            )

        # Verify error message indicates uniqueness constraint
        error_msg = str(exc_info.value).lower()
        assert (
            "unique" in error_msg 
            or "duplicate" in error_msg
            or "повторяющееся значение ключа" in error_msg
            or "нарушает ограничение уникальности" in error_msg
        )


@pytest.mark.asyncio
async def test_update_job_status_nonexistent_job(dal):
    """Test updating status for a non-existent job. Should not fail."""
    async with dal._pool.acquire() as conn:
        non_existent_id = "nonexistent-job-id"
        await update_job_status(non_existent_id, "running", connection=conn)

        # Verify no job was actually created or updated
        count = await conn.fetchval("SELECT COUNT(*) FROM jobs WHERE job_id = $1", non_existent_id)
        assert count == 0


@pytest.mark.asyncio
async def test_get_best_model_by_metric(dal, sample_config):
    """Test getting the best model based on a metric"""
    async with dal._pool.acquire() as conn:
        # Create test data
        job_id = await create_job("training", {}, connection=conn)
        model_id = str(uuid.uuid4())
        config_id = await create_or_get_config(sample_config, connection=conn)
        
        await create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/path/1",
            created_at=datetime.now(),
            connection=conn,
        )
        
        await create_training_result(
            job_id=job_id,
            model_id=model_id,
            config_id=config_id,
            metrics={"val_MIC": 10.5, "val_loss": 0.9},
            duration=100,
            connection=conn,
        )
        
        # Create a second model and result to compare against
        job2 = await create_job("training", {}, connection=conn)
        model2_id = str(uuid.uuid4())
        await create_model_record(
            model_id=model2_id,
            job_id=job2,
            model_path="/path/2",
            created_at=datetime.now(),
            connection=conn,
        )
        await create_training_result(
            job_id=job2,
            model_id=model2_id,
            config_id=config_id,
            metrics={"val_MIC": 9.8, "val_loss": 0.92},
            duration=100,
            connection=conn,
        )

        # Get best model by val_MIC (lower is better)
        best_model = await get_best_model_by_metric(
            "val_MIC", higher_is_better=False, connection=conn
        )

        assert best_model is not None
        assert best_model["model_id"] == model2_id
        assert best_model["metrics"]["val_MIC"] == 9.8

        # Get best model by non-existent metric
        with pytest.raises(ValueError) as excinfo_model:
            await get_best_model_by_metric("nonexistent_metric", connection=conn)
        assert "Invalid metric_name" in str(excinfo_model.value)


@pytest.mark.asyncio
async def test_get_best_config_by_metric(dal, sample_config):
    """Test retrieving the best config by a given metric"""
    async with dal._pool.acquire() as conn:
        config_id = await create_or_get_config(sample_config, connection=conn)

        # Ensure there are no active configs to force fallback to best by metric
        await conn.execute("UPDATE configs SET is_active = FALSE")

        # Create necessary data for the query to work
        job_id = await create_job("training", {}, connection=conn)
        model_id = str(uuid.uuid4())
        await create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/path",
            created_at=datetime.now(),
            connection=conn,
        )
        await create_training_result(
            job_id, model_id, config_id, {"val_loss": 0.88}, 120, connection=conn
        )

        best_config = await get_best_config_by_metric(
            "val_loss", higher_is_better=True, connection=conn
        )

        assert best_config is not None
        assert best_config["config_id"] == config_id


@patch("deployment.app.db.queries.models._is_path_safe", return_value=True)
@pytest.mark.asyncio
async def test_delete_model_record_and_file(mock_is_path_safe, dal, tmp_path):
    """Test deleting a model record and its associated file"""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("training", {}, connection=conn)
        model_id = str(uuid.uuid4())
        
        # Create a real model file for the record to point to
        model_path = str(tmp_path / f"{model_id}.pkl")
        with open(model_path, "wb") as f:
            f.write(b"fake model data")

        await create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path=model_path,
            created_at=datetime.now(),
            connection=conn,
        )

        assert os.path.exists(model_path)

        # Act
        deleted = await delete_model_record_and_file(model_id, connection=conn)

        # Assert
        assert deleted is True
        assert not os.path.exists(model_path)  # File should be gone
        res = await conn.fetchrow("SELECT * FROM models WHERE model_id = $1", model_id)
        assert res is None  # Record should be gone


@pytest.mark.asyncio
async def test_delete_model_nonexistent_file(dal):
    """Test that deleting a model record succeeds even if the file is already gone"""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("training", {}, connection=conn)
        model_id = str(uuid.uuid4())
        
        # Ensure the file does NOT exist
        model_path = "/path/to/nonexistent_model.pkl"
        await create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path=model_path,
            created_at=datetime.now(),
            connection=conn,
        )

        # Act
        deleted = await delete_model_record_and_file(model_id, connection=conn)

        # Assert
        assert deleted is True  # Should report success
        res = await conn.fetchrow("SELECT * FROM models WHERE model_id = $1", model_id)
        assert res is None  # Record should still be gone


@pytest.mark.asyncio
async def test_delete_models_by_ids_with_active_model(dal):
    """Test that an active model is not deleted by delete_models_by_ids"""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("training", {}, connection=conn)
        model_id = str(uuid.uuid4())
        await create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/path",
            created_at=datetime.now(),
            connection=conn,
        )
        await set_model_active(model_id, connection=conn)  # Make it active

        # Try to delete it
        result = await delete_models_by_ids([model_id], connection=conn)

        assert result["deleted_count"] == 0
        assert result["skipped_count"] == 1
        assert model_id in result["skipped_models"]

        res = await conn.fetchrow("SELECT * FROM models WHERE model_id = $1", model_id)
        assert res is not None  # Should still exist


@pytest.mark.asyncio
async def test_delete_configs_by_ids_with_active_set(dal, sample_config):
    """Test that an active config is not deleted by delete_configs_by_ids"""
    async with dal._pool.acquire() as conn:
        config_id = await create_or_get_config(sample_config, connection=conn)
        await set_config_active(config_id, connection=conn)  # Make it active

        # Try to delete it
        result = await delete_configs_by_ids([config_id], connection=conn)

        assert result["deleted_count"] == 0
        assert result["skipped_count"] == 1
        assert config_id in result["skipped_configs"]

        res = await conn.fetchrow("SELECT * FROM configs WHERE config_id = $1", config_id)
        assert res is not None  # Should still exist


@pytest.mark.asyncio
async def test_set_model_active_nonexistent_id(dal):
    """Test setting a non-existent model ID as active fails gracefully"""
    async with dal._pool.acquire() as conn:
        result = await set_model_active("non-existent-id", connection=conn)
        assert result is False


@pytest.mark.asyncio
async def test_set_config_active_nonexistent_id(dal):
    """Test setting a non-existent config ID as active fails gracefully"""
    async with dal._pool.acquire() as conn:
        result = await set_config_active("non-existent-id", connection=conn)
        assert result is False


@pytest.mark.asyncio
async def test_create_or_get_config_idempotent(dal, sample_config):
    """Test that create_or_get_config is idempotent"""
    async with dal._pool.acquire() as conn:
        config_id1 = await create_or_get_config(sample_config, connection=conn)
        assert config_id1 is not None

        # Try to create the same config again - should return the same ID
        config_id2 = await create_or_get_config(sample_config, connection=conn)
        assert config_id2 == config_id1

        # Verify only one record exists for this config
        count = await conn.fetchval("SELECT COUNT(*) FROM configs WHERE config_id = $1", config_id1)
        assert count == 1


@pytest.mark.asyncio
async def test_transaction_rollback_on_error(dal):
    """Test that a transaction is rolled back if an error occurs."""
    async with dal._pool.acquire() as conn:
        # Get initial count
        initial_count = await conn.fetchval("SELECT COUNT(*) FROM jobs")

        try:
            async with conn.transaction():
                # This one is fine
                await create_job("job1", {"p": 1}, connection=conn)
                # This one will fail due to a non-existent table, triggering a rollback
                await conn.execute("INSERT INTO non_existent_table VALUES (1)")
        except asyncpg.PostgresError:
            pass  # Expected error

        # Verify that the first job was rolled back
        final_count = await conn.fetchval("SELECT COUNT(*) FROM jobs")
        assert final_count == initial_count


@patch("deployment.app.db.queries.models._is_path_safe", return_value=True)
@pytest.mark.asyncio
async def test_delete_models_by_ids_with_path_traversal_attempt(
    mock_is_path_safe, dal, tmp_path
):
    """Test that delete_models_by_ids handles path traversal attempts for model files."""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("training", {}, connection=conn)
        model_id_1 = str(uuid.uuid4())
        await create_model_record(
            model_id=model_id_1,
            job_id=job_id,
            model_path="/path/1",
            created_at=datetime.now(),
            connection=conn,
        )

        # Create another model that points to an unsafe path
        model_id_unsafe = str(uuid.uuid4())
        unsafe_model_path = "/etc/passwd"  # An unsafe path

        await create_model_record(
            model_id=model_id_unsafe,
            job_id=job_id,
            model_path=unsafe_model_path,
            created_at=datetime.now(),
            is_active=False,
            connection=conn,
        )

        # Configure _is_path_safe to return False for the unsafe path
        def custom_is_path_safe(base_dir, path_to_check):
            if path_to_check == unsafe_model_path:
                return False
            return True  # Allow safe paths

        mock_is_path_safe.side_effect = custom_is_path_safe

        # Act: Try to delete both safe and unsafe models
        result = await delete_models_by_ids(
            [model_id_1, model_id_unsafe], connection=conn
        )

        # Assert
        assert result["deleted_count"] >= 1
        # The unsafe model should be deleted (not skipped) since it's not active
        assert model_id_unsafe not in result.get("skipped_models", [])
