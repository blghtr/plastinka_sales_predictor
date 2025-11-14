import pytest
from datetime import datetime

import asyncpg

from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.core import execute_query
from deployment.app.db.queries.jobs import create_job
from deployment.app.db.queries.models import create_model_record
from deployment.app.db.queries.configs import create_or_get_config
from deployment.app.db.queries.results import create_training_result
from deployment.app.db.utils import generate_id


@pytest.mark.asyncio
async def test_foreign_key_enforcement_on_insert_jobs_history(dal):
    """
    Test that inserting into job_status_history fails if the job_id does not exist in jobs.
    """
    async with dal._pool.acquire() as conn:
        with pytest.raises(DatabaseError) as excinfo:
            await execute_query(
                "INSERT INTO job_status_history (job_id, status, progress, status_message, updated_at) VALUES ($1, $2, $3, $4, $5)",
                conn,
                (
                    "non_existent_job_id",
                    "running",
                    50.0,
                    "Processing...",
                    datetime.fromisoformat("2023-01-01T12:00:00"),
                ),
            )
        error_msg = str(excinfo.value.original_error).lower()
        assert (
            "foreign key" in error_msg 
            or "violates foreign key" in error_msg
            or "нарушает ограничение внешнего ключа" in error_msg
            or "нарушает ограничение" in error_msg
        )


@pytest.mark.asyncio
async def test_foreign_key_enforcement_on_insert_training_results_model(dal, sample_config):
    """
    Test that inserting into training_results fails if the model_id does not exist in models.
    """
    async with dal._pool.acquire() as conn:
        # First, create a job and a config, as training_results depends on them too
        job_id = await create_job("training", {}, connection=conn)
        config_id = await create_or_get_config(sample_config, connection=conn)

        with pytest.raises(DatabaseError) as excinfo:
            await execute_query(
                "INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics, duration) VALUES ($1, $2, $3, $4, $5, $6)",
                conn,
                (generate_id(), job_id, "non_existent_model_id", config_id, "{}", 100),
            )
        error_msg = str(excinfo.value.original_error).lower()
        assert (
            "foreign key" in error_msg
            or "violates foreign key" in error_msg
            or "нарушает ограничение внешнего ключа" in error_msg
            or "нарушает ограничение" in error_msg
        )


@pytest.mark.asyncio
async def test_foreign_key_enforcement_on_insert_training_results_config(dal, sample_config):
    """
    Test that inserting into training_results fails if the config_id does not exist in configs.
    """
    async with dal._pool.acquire() as conn:
        job_id = await create_job("training", {}, connection=conn)
        model_id = str(generate_id())
        await create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/path/to/model",
            created_at=datetime.fromisoformat("2023-01-01T00:00:00"),
            connection=conn,
        )

        with pytest.raises(DatabaseError) as excinfo:
            await execute_query(
                "INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics, duration) VALUES ($1, $2, $3, $4, $5, $6)",
                conn,
                (generate_id(), job_id, model_id, "non_existent_config_id", "{}", 100),
            )
        error_msg = str(excinfo.value.original_error).lower()
        assert (
            "foreign key" in error_msg 
            or "violates foreign key" in error_msg
            or "нарушает ограничение внешнего ключа" in error_msg
            or "нарушает ограничение" in error_msg
        )


@pytest.mark.asyncio
async def test_foreign_key_cascade_delete_jobs(dal):
    """
    Test that deleting a job with history entries is RESTRICTED (not cascaded).
    The current schema for job_status_history.job_id does NOT specify ON DELETE CASCADE.
    """
    async with dal._pool.acquire() as conn:
        job_id = await create_job("test_type", {}, connection=conn)

        # Create a history entry for this job
        await execute_query(
            "INSERT INTO job_status_history (job_id, status, progress, status_message, updated_at) VALUES ($1, $2, $3, $4, $5)",
            conn,
            (job_id, "running", 50.0, "Processing...", datetime.fromisoformat("2023-01-01T12:00:00")),
        )

        # Attempt to delete the job
        with pytest.raises(DatabaseError) as excinfo:
            await execute_query(
                "DELETE FROM jobs WHERE job_id = $1", conn, (job_id,)
            )
        error_msg = str(excinfo.value.original_error).lower()
        assert (
            "foreign key" in error_msg 
            or "violates foreign key" in error_msg
            or "нарушает ограничение внешнего ключа" in error_msg
            or "нарушает ограничение" in error_msg
        )

        # Verify job and history entry still exist
        job_entry = await execute_query(
            "SELECT * FROM jobs WHERE job_id = $1", conn, (job_id,)
        )
        history_entry = await execute_query(
            "SELECT * FROM job_status_history WHERE job_id = $1",
            conn,
            (job_id,),
            fetchall=True,
        )

        assert job_entry is not None
        assert len(history_entry) == 1


@pytest.mark.asyncio
async def test_successful_insert_with_valid_foreign_keys(dal):
    """
    Test that inserts are successful when foreign keys are valid.
    """
    async with dal._pool.acquire() as conn:
        job_id = await create_job("test_type", {}, connection=conn)

        # This should succeed
        await execute_query(
            "INSERT INTO job_status_history (job_id, status, progress, status_message, updated_at) VALUES ($1, $2, $3, $4, $5)",
            conn,
            (job_id, "completed", 100.0, "Done", datetime.fromisoformat("2023-01-01T13:00:00")),
        )

        history_entry = await execute_query(
            "SELECT * FROM job_status_history WHERE job_id = $1",
            conn,
            (job_id,),
            fetchall=True,
        )
        assert len(history_entry) == 1
        assert history_entry[0]["status"] == "completed"
