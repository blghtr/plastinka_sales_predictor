import pytest
from datetime import datetime

import asyncpg

from deployment.app.db.queries.jobs import create_job


@pytest.mark.asyncio
async def test_foreign_key_constraint(dal):
    """Test that foreign key constraints are enforced."""
    async with dal._pool.acquire() as conn:
        # Insert a job with job_type
        job_id = await create_job("training", {}, connection=conn)
        now = datetime.fromisoformat("2021-01-01T00:00:00")

        # Insert a job status history entry with a valid job_id
        await conn.execute(
            "INSERT INTO job_status_history (job_id, status, progress, updated_at) VALUES ($1, $2, $3, $4)",
            job_id, "running", 50.0, now,
        )

        # Try to insert a job status history entry with an invalid job_id
        with pytest.raises(asyncpg.IntegrityConstraintViolationError) as excinfo:
            await conn.execute(
                "INSERT INTO job_status_history (job_id, status, progress, updated_at) VALUES ($1, $2, $3, $4)",
                "nonexistent_job", "running", 50.0, now,
            )

        # Check that the error message mentions foreign key constraint
        error_msg = str(excinfo.value).lower()
        assert (
            "foreign key" in error_msg 
            or "violates foreign key" in error_msg
            or "нарушает ограничение внешнего ключа" in error_msg
            or "нарушает ограничение" in error_msg
        )
