"""
Tests for core functionality in the database module.

This module contains comprehensive tests for the database operations,
covering the main database operations including:

1. Query execution
2. Job management
3. Parameter set management
4. Model management
5. Database result handling

Each test focuses on verifying the correct behavior of a specific database function
in isolation, using fixtures to set up the appropriate test environment.
"""

import json
import uuid
from datetime import date, datetime

import asyncpg
import pytest

from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.core import execute_many, execute_query
from deployment.app.db.queries.jobs import create_job, get_job, list_jobs, update_job_status
from deployment.app.db.queries.models import create_model_record, get_active_model, set_model_active
from deployment.app.db.queries.configs import create_or_get_config, get_active_config, set_config_active
from deployment.app.db.queries.results import (
    create_data_upload_result,
    create_prediction_result,
    create_training_result,
    create_tuning_result,
    get_training_results,
    get_tuning_results,
)
from deployment.app.db.queries.processing_runs import create_processing_run, update_processing_run
from deployment.app.db.queries.features import get_feature_dataframe
from deployment.app.db.queries.multiindex import get_or_create_multiindex_ids_batch, get_multiindex_mapping_by_ids
from deployment.app.db.utils import generate_id

# =============================================
# Используем фикстуры из conftest.py
# =============================================

@pytest.mark.asyncio
async def test_execute_query_select(dal):
    """Test execute_query with a SELECT statement"""
    async with dal._pool.acquire() as conn:
        # Add some test data
        now = datetime.now()
        await conn.execute(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
            "test-job-select",
            "training",
            "pending",
            now,
            now,
        )

        # Test execute_query with SELECT
        result = await execute_query(
            "SELECT * FROM jobs WHERE job_id = $1",
            conn,
            ("test-job-select",),
        )
        assert result is not None
        assert result["job_id"] == "test-job-select"
        assert result["job_type"] == "training"

        # Test execute_query with fetchall=True
        results = await execute_query("SELECT * FROM jobs WHERE job_id = $1", conn, ("test-job-select",), fetchall=True)
        assert len(results) == 1
        assert results[0]["job_id"] == "test-job-select"


@pytest.mark.asyncio
async def test_execute_query_insert(dal):
    """Test execute_query with an INSERT statement"""
    async with dal._pool.acquire() as conn:
        job_id = str(uuid.uuid4())
        now = datetime.now()

        # Use execute_query for INSERT
        await execute_query(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
            conn,
            (job_id, "training", "pending", now, now),
        )

        # Verify the record was inserted
        result = await conn.fetchrow("SELECT * FROM jobs WHERE job_id = $1", job_id)
        assert result is not None
        assert result["job_id"] == job_id


@pytest.mark.asyncio
async def test_execute_query_error(dal):
    """Test that execute_query raises DatabaseError on SQL errors"""
    async with dal._pool.acquire() as conn:
        # Execute a query with syntax error
        with pytest.raises(DatabaseError):
            await execute_query("SELECT * FROM nonexistent_table", conn)


@pytest.mark.asyncio
async def test_execute_many(dal):
    """Test execute_many with multiple parameter sets"""
    async with dal._pool.acquire() as conn:
        # Create test jobs
        now = datetime.now()
        params_list = [
            (str(uuid.uuid4()), "training", "pending", now, now),
            (str(uuid.uuid4()), "prediction", "pending", now, now),
            (str(uuid.uuid4()), "training", "pending", now, now),
        ]

        # Use execute_many
        await execute_many(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
            params_list=params_list,
            connection=conn,
    )

    # Verify all records were inserted
        count = await conn.fetchval("SELECT COUNT(*) FROM jobs WHERE job_type IN ($1, $2)", "training", "prediction")
        assert count >= len(params_list)


@pytest.mark.asyncio
async def test_execute_many_error(dal):
    """Test that execute_many raises DatabaseError on SQL errors"""
    async with dal._pool.acquire() as conn:
        # Try to execute on a non-existent table
        params_list = [("name1",), ("name2",)]

        with pytest.raises(DatabaseError) as exc_info:
            await execute_many(
                "INSERT INTO nonexistent_table (name) VALUES ($1)",
                params_list=params_list,
                connection=conn,
            )

        assert "database operation failed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_execute_many_empty_params(dal):
    """Test that execute_many handles empty params lists gracefully"""
    async with dal._pool.acquire() as conn:
        # Call with empty params list - should return without error
        result = await execute_many(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
            params_list=[],
            connection=conn,
        )
        assert result is None


@pytest.mark.asyncio
async def test_generate_id():
    """Test that generate_id produces unique IDs"""
    ids = [generate_id() for _ in range(10)]
    unique_ids = set(ids)
    assert len(ids) == len(unique_ids)


@pytest.mark.asyncio
async def test_create_job(dal):
    """Test creating a job record"""
    async with dal._pool.acquire() as conn:
        job_type = "training"
        parameters = {"batch_size": 32, "epochs": 10}
        job_id = await create_job(job_type, parameters, connection=conn)
        assert job_id is not None
        job = await conn.fetchrow("SELECT * FROM jobs WHERE job_id = $1", job_id)
        assert job["job_type"] == job_type
        # PostgreSQL JSONB returns as dict, but may be string in some cases
        import json
        job_params = job["parameters"]
        if isinstance(job_params, str):
            job_params = json.loads(job_params)
        assert job_params == parameters


@pytest.mark.asyncio
async def test_update_job_status(dal, sample_job_data):
    """Test updating job status"""
    async with dal._pool.acquire() as conn:
        job_id = await create_job(
            sample_job_data["job_type"], sample_job_data["parameters"], connection=conn
        )
        await update_job_status(
            job_id=job_id,
            status="running",
            progress=50,
            status_message="Halfway there",
            connection=conn,
        )
        job = await get_job(job_id, connection=conn)
        assert job["status"] == "running"
        assert job["progress"] == 50


@pytest.mark.asyncio
async def test_get_job_nonexistent(dal):
    """Test getting a non-existent job"""
    async with dal._pool.acquire() as conn:
        result = await get_job(str(uuid.uuid4()), connection=conn)
        assert result is None


@pytest.mark.asyncio
async def test_list_jobs(dal):
    """Test listing jobs with filters"""
    async with dal._pool.acquire() as conn:
        job_1 = await create_job("training", {"param": "value1"}, connection=conn)
        job_2 = await create_job("prediction", {"param": "value2"}, connection=conn)
        job_3 = await create_job("training", {"param": "value3"}, connection=conn)
        await update_job_status(job_1, "completed", connection=conn)
        await update_job_status(job_2, "running", connection=conn)
        await update_job_status(job_3, "running", connection=conn)
        assert len(await list_jobs(connection=conn)) == 3
        assert len(await list_jobs(job_type="training", connection=conn)) == 2
        assert len(await list_jobs(status="running", connection=conn)) == 2
        assert len(await list_jobs(job_type="training", status="running", connection=conn)) == 1


@pytest.mark.asyncio
async def test_create_model_record(dal, sample_model_data):
    """Test creating a model record"""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("training", {}, connection=conn)
        await create_model_record(
        model_id=sample_model_data["model_id"],
        job_id=job_id,
        model_path=sample_model_data["model_path"],
        created_at=sample_model_data["created_at"],
        metadata=sample_model_data["metadata"],
        is_active=True,
        connection=conn,
    )
        model = await conn.fetchrow("SELECT * FROM models WHERE model_id = $1", sample_model_data["model_id"])
    assert model["model_id"] == sample_model_data["model_id"]


@pytest.mark.asyncio
async def test_get_active_model(dal, sample_model_data):
    """Test getting the active model"""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("training", {}, connection=conn)
        await create_model_record(
        model_id=sample_model_data["model_id"],
        job_id=job_id,
        model_path=sample_model_data["model_path"],
        created_at=sample_model_data["created_at"],
        metadata=sample_model_data["metadata"],
        is_active=True,
        connection=conn,
    )
        active_model = await get_active_model(connection=conn)
    assert active_model["model_id"] == sample_model_data["model_id"]


@pytest.mark.asyncio
async def test_set_model_active(dal):
    """Test setting a model as active"""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("training", {}, connection=conn)
        model_id_1 = str(uuid.uuid4())
        model_id_2 = str(uuid.uuid4())
        await create_model_record(model_id_1, job_id, "/path/1", datetime.now(), connection=conn)
        await create_model_record(model_id_2, job_id, "/path/2", datetime.now(), connection=conn)
        await set_model_active(model_id_1, connection=conn)
        active_model = await get_active_model(connection=conn)
        assert active_model["model_id"] == model_id_1


@pytest.mark.asyncio
async def test_create_or_get_config(dal, sample_config):
    """Test creating a config record"""
    async with dal._pool.acquire() as conn:
        config_id = await create_or_get_config(
            sample_config, is_active=True, connection=conn
        )
        assert config_id is not None
        retrieved = await execute_query("SELECT * FROM configs WHERE config_id = $1", conn, (config_id,))
        import json
        config_value = retrieved["config"]
        if isinstance(config_value, str):
            config_value = json.loads(config_value)
        assert config_value == sample_config


@pytest.mark.asyncio
async def test_get_active_config(dal, sample_config):
    """Test retrieving the active config"""
    async with dal._pool.acquire() as conn:
        config_id = await create_or_get_config(
            sample_config, is_active=True, connection=conn
        )
        active_config = await get_active_config(connection=conn)
        assert active_config["config_id"] == config_id


@pytest.mark.asyncio
async def test_set_config_active(dal, sample_config):
    """Test setting a config as active"""
    async with dal._pool.acquire() as conn:
        config_id1 = await create_or_get_config(sample_config, connection=conn)
        await create_or_get_config({"a": 1}, connection=conn)
        await set_config_active(config_id1, connection=conn)
        active = await get_active_config(connection=conn)
        assert active["config_id"] == config_id1


@pytest.mark.asyncio
async def test_create_data_upload_result(dal):
    """Test creating a data upload result"""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("data_upload", {}, connection=conn)
        run_id = await create_processing_run(datetime.now(), "running", "f.csv", end_time=None, connection=conn)
        result_id = await create_data_upload_result(job_id, 100, ["sales"], run_id, conn)
        res = await execute_query("SELECT * FROM data_upload_results WHERE result_id = $1", conn, (result_id,))
        assert res["records_processed"] == 100


@pytest.mark.asyncio
async def test_create_prediction_result(dal):
    """Test creating a prediction result"""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("prediction", {}, connection=conn)
        model_id = str(uuid.uuid4())
        await create_model_record(model_id, job_id, "/path/m.onnx", datetime.now(), connection=conn)
        result_id = await create_prediction_result(job_id, model_id, "/out.csv", {"a": 1}, date(2023, 1, 1), connection=conn)
        res = await execute_query("SELECT * FROM prediction_results WHERE result_id = $1", conn, (result_id,))
        assert res["model_id"] == model_id


@pytest.mark.asyncio
async def test_get_training_results_by_id(dal):
    """Test getting a single training result by ID."""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("training", {}, connection=conn)
        model_id = str(uuid.uuid4())
        config_id = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO models (model_id, job_id, model_path, created_at) VALUES ($1, $2, $3, $4)",
            model_id, job_id, "/path", datetime.now()
        )
        await conn.execute(
            "INSERT INTO configs (config_id, config, created_at) VALUES ($1, $2, $3)",
            config_id, json.dumps({"a": 1}), datetime.now()
        )
        result_id = await create_training_result(job_id, model_id, config_id, {"m": 1}, 120, conn)
        retrieved = await get_training_results(result_id=result_id, connection=conn)
        assert retrieved["result_id"] == result_id


@pytest.mark.asyncio
async def test_get_training_results_list(dal):
    """Test getting a list of recent training results."""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("training", {}, connection=conn)
        model_id = str(uuid.uuid4())
        config_id = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO models (model_id, job_id, model_path, created_at) VALUES ($1, $2, $3, $4)",
            model_id, job_id, "/path", datetime.now()
        )
        await conn.execute(
            "INSERT INTO configs (config_id, config, created_at) VALUES ($1, $2, $3)",
            config_id, json.dumps({"a": 1}), datetime.now()
        )
        for i in range(5):
            await create_training_result(job_id, model_id, config_id, {"m": i}, 100 + i, conn)
        results = await get_training_results(connection=conn)
        assert len(results) == 5
        assert len(await get_training_results(limit=2, connection=conn)) == 2


@pytest.mark.asyncio
async def test_get_tuning_results_by_id(dal):
    """Test getting a single tuning result by ID."""
    async with dal._pool.acquire() as conn:
        job_id = await create_job("tuning", {}, connection=conn)
        config_id = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO configs (config_id, config, created_at) VALUES ($1, $2, $3)",
            config_id, json.dumps({"a": 1}), datetime.now()
        )
        result_id = await create_tuning_result(job_id, config_id, {"m": 1}, 300, connection=conn)
        retrieved = await get_tuning_results(connection=conn, result_id=result_id)
        assert retrieved["result_id"] == result_id


@pytest.mark.asyncio
async def test_get_tuning_results_list(dal):
    """Test getting a list of recent tuning results."""
    async with dal._pool.acquire() as conn:
        config_id = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO configs (config_id, config, created_at) VALUES ($1, $2, $3)",
            config_id, json.dumps({"a": 1}), datetime.now()
        )
        ids = []
        for i in range(5):
            job_id = await create_job("tuning", {}, connection=conn)
            ids.append(await create_tuning_result(job_id, config_id, {"m": i}, 300 + i, connection=conn))
        results = await get_tuning_results(connection=conn)
        assert len(results) == 5
        assert results[0]["result_id"] == ids[4]


@pytest.mark.asyncio
async def test_create_and_update_processing_run(dal):
    """Test creating and updating a processing run"""
    async with dal._pool.acquire() as conn:
        start_time = datetime.now()
        run_id = await create_processing_run(start_time, "running", "f.csv", connection=conn)
        run = await conn.fetchrow("SELECT * FROM processing_runs WHERE run_id = $1", run_id)
        assert run["status"] == "running"
        end_time = datetime.now()
        await update_processing_run(run_id, "completed", end_time, conn)
        updated_run = await conn.fetchrow("SELECT * FROM processing_runs WHERE run_id = $1", run_id)
        assert updated_run["status"] == "completed"


@pytest.mark.asyncio
async def test_get_feature_dataframe(dal):
    """Test the generic get_feature_dataframe function."""
    async with dal._pool.acquire() as conn:
        # 1. Setup data
        try:
            await conn.execute(
                "INSERT INTO dim_multiindex_mapping (multiindex_id, barcode, artist, album, cover_type, price_category, release_type, recording_decade, release_decade, style, recording_year) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
                1, '1', '1', '1', '1', '1', '1', '1', '1', '1', 1
            )
        except asyncpg.IntegrityConstraintViolationError:
            pass  # Already exists

        await conn.execute(
            "INSERT INTO fact_sales (multiindex_id, data_date, value) VALUES ($1, $2, $3)",
            1, date(2023, 1, 15), 100
        )
        await conn.execute(
            "INSERT INTO report_features (multiindex_id, data_date, availability, confidence) VALUES ($1, $2, $3, $4)",
            1, date(2023, 1, 15), 0.9, 0.8
        )

        # 2. Test fetching from fact_sales (single value column)
        sales_data = await get_feature_dataframe(
            table_name="fact_sales",
            columns=["value"],
            connection=conn,
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        assert len(sales_data) == 1
        assert sales_data[0]["multiindex_id"] == 1
        assert str(sales_data[0]["data_date"]) == "2023-01-15"
        assert sales_data[0]["value"] == 100

        # 3. Test fetching from report_features (multiple value columns)
        report_data = await get_feature_dataframe(
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
        no_data = await get_feature_dataframe(
            table_name="fact_sales",
            columns=["value"],
            connection=conn,
            start_date="2024-01-01"
        )
        assert len(no_data) == 0

        # 5. Test invalid column name (should raise ValueError)
        with pytest.raises(ValueError):
            await get_feature_dataframe(
                table_name="fact_sales",
                columns=["value; DROP TABLE users"],
            connection=conn
        )


@pytest.mark.asyncio
async def test_multiindex_batch_order_and_inverse(dal):
    """Batch function preserves order and reverse mapping is aligned."""
    async with dal._pool.acquire() as conn:
        tuples = [
            ("bc1", "art1", "alb1", "ct1", "pc1", "rt1", "rd1", "reld1", "st1", 2001),
            ("bc2", "art2", "alb2", "ct2", "pc2", "rt2", "rd2", "reld2", "st2", 2002),
            ("bc3", "art3", "alb3", "ct3", "pc3", "rt3", "rd3", "reld3", "st3", 2003),
        ]

        ids = await get_or_create_multiindex_ids_batch(tuples, connection=conn)
        assert isinstance(ids, tuple)
        assert len(ids) == len(tuples)

        # Call again to exercise existing rows path; must return same ids, same order
        ids_again = await get_or_create_multiindex_ids_batch(tuples, connection=conn)
        assert ids_again == ids

        # Reverse mapping should align with ids order
        mappings = await get_multiindex_mapping_by_ids(list(ids), connection=conn)
        assert len(mappings) == len(ids)

        # Verify each mapping corresponds to the input tuple by order
        for idx, mid in enumerate(ids):
            m = mappings[idx]
            # Ensure id matches
            assert int(m["multiindex_id"]) == int(mid)
            # Ensure all attributes match string-normalized values
            exp = tuples[idx]
            assert (
                m["barcode"], m["artist"], m["album"], m["cover_type"], m["price_category"],
                m["release_type"], m["recording_decade"], m["release_decade"], m["style"], str(m["recording_year"])
            ) == tuple(map(str, exp))


@pytest.mark.asyncio
async def test_multiindex_batch_handles_duplicates_by_position(dal):
    """If duplicates are present in input, positions must be preserved in output ids."""
    async with dal._pool.acquire() as conn:
        base = ("bcD", "artD", "albD", "ctD", "pcD", "rtD", "rdD", "reldD", "stD", 1999)
        tuples = [base, base, base]

        ids = await get_or_create_multiindex_ids_batch(tuples, connection=conn)
        assert len(ids) == 3
        # All ids should be equal and correspond to the same mapping, repeated by position
        assert ids[0] == ids[1] == ids[2]

        mappings = await get_multiindex_mapping_by_ids(list(ids), connection=conn)
        assert len(mappings) == 3
        for m in mappings:
            assert (
                m["barcode"], m["artist"], m["album"], m["cover_type"], m["price_category"],
                m["release_type"], m["recording_decade"], m["release_decade"], m["style"], str(m["recording_year"])
            ) == tuple(map(str, base))
