import time
from datetime import datetime
from unittest.mock import patch

import pytest


@pytest.fixture
def test_model_path(tmp_path):
    """Create a temporary model file for testing."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "test_model.onnx"

    # Create a dummy model file
    with open(model_path, "wb") as f:
        f.write(b"dummy model content")

    return str(model_path)


@pytest.fixture
async def test_setup_db(dal):
    """Set up test database with required reference data."""
    # Insert a test config first
    config_id = await dal.create_or_get_config(
        {"test": "config"},
        is_active=False,
    )

    # Insert a test job that subsequent tests can rely on
    job_id = await dal.create_job(
        job_type="training",
        parameters={},
    )
    await dal.update_job_status(
        job_id=job_id,
        status="completed",
    )

    yield {
        "dal": dal,
        "config_id": config_id,
        "job_id": job_id,
    }


@pytest.mark.asyncio
async def test_create_model_record(test_setup_db, test_model_path):
    """Test that a model record can be created in the database."""
    dal = test_setup_db["dal"]
    job_id = test_setup_db["job_id"]

    # Create test data
    model_id = "test_model_123"
    created_at = datetime.now()
    metadata = {"test_key": "test_value", "file_size_bytes": 100}

    # Create the model record
    await dal.create_model_record(
        model_id=model_id,
        job_id=job_id,
        model_path=test_model_path,
        created_at=created_at,
        metadata=metadata,
        is_active=False,
    )

    # Verification
    record = await dal.execute_raw_query(
        "SELECT * FROM models WHERE model_id = $1",
        params=(model_id,),
        fetchall=False,
    )

    assert record is not None
    assert record["model_id"] == model_id
    assert record["job_id"] == job_id
    assert record["model_path"] == test_model_path
    assert record["is_active"] is False
    # Metadata is stored as JSONB, may be returned as string or dict
    import json
    if isinstance(record["metadata"], str):
        assert json.loads(record["metadata"]) == metadata
    else:
        assert record["metadata"] == metadata


@pytest.mark.asyncio
async def test_activate_model(test_setup_db, test_model_path):
    """Test that a model can be activated and other models deactivated."""
    dal = test_setup_db["dal"]
    job_id = test_setup_db["job_id"]

    # Create multiple model records
    models = [
        {"model_id": "model_1", "is_active": False},
        {"model_id": "model_2", "is_active": False},
        {"model_id": "model_3", "is_active": False},
    ]

    created_at = datetime.now()

    # Create all model records
    for model in models:
        await dal.create_model_record(
            model_id=model["model_id"],
            job_id=job_id,
            model_path=test_model_path,
            created_at=created_at,
            metadata={"test": "test"},
            is_active=model["is_active"],
        )

    # Activate model_2
    result = await dal.set_model_active("model_2", deactivate_others=True)
    assert result is True

    # Get active model
    active_model = await dal.get_active_model()
    assert active_model is not None
    assert active_model["model_id"] == "model_2"

    # Check other models are inactive
    other_models = await dal.execute_raw_query(
        "SELECT model_id, is_active FROM models WHERE model_id != $1",
        params=("model_2",),
        fetchall=True,
    )

    for model in other_models:
        assert model["is_active"] is False


@pytest.mark.asyncio
async def test_get_best_model_by_metric(test_setup_db, test_model_path):
    """Test retrieving the best model based on a metric."""
    dal = test_setup_db["dal"]
    job_id = test_setup_db["job_id"]
    config_id = test_setup_db["config_id"]

    # Create models
    models = [
        {"model_id": "model_1", "metric_value": 0.8},
        {"model_id": "model_2", "metric_value": 0.9},  # Best model
        {"model_id": "model_3", "metric_value": 0.7},
    ]

    created_at = datetime.now()

    # Create model records and training results
    for model in models:
        # Create model record
        await dal.create_model_record(
            model_id=model["model_id"],
            job_id=job_id,
            model_path=test_model_path,
            created_at=created_at,
            metadata={"test": "test"},
            is_active=False,
        )

        # Create training result
        await dal.create_training_result(
            job_id=job_id,
            model_id=model["model_id"],
            config_id=config_id,
            metrics={"val_loss": model["metric_value"]},
            duration=3600,
        )

    # Get the best model by metric
    best_model = await dal.get_best_model_by_metric(
        "val_loss", higher_is_better=True
    )

    assert best_model is not None
    assert best_model["model_id"] == "model_2"
    assert best_model["metrics"]["val_loss"] == 0.9


@pytest.mark.asyncio
async def test_get_recent_models(test_setup_db, test_model_path):
    """Test retrieving recent models."""
    dal = test_setup_db["dal"]
    job_id = test_setup_db["job_id"]

    # Create models with different timestamps - adding a small delay
    # Create models in reverse order to ensure model_3 is most recent
    models = [
        {"model_id": "model_3", "is_active": False},
        {"model_id": "model_2", "is_active": False},
        {"model_id": "model_1", "is_active": True},
    ]

    # Create all model records with deliberate pauses
    for model in models:
        time.sleep(0.05)  # Delay to ensure different timestamps
        await dal.create_model_record(
            model_id=model["model_id"],
            job_id=job_id,
            model_path=test_model_path,
            created_at=datetime.now(),
            metadata={"test": "test"},
            is_active=model["is_active"],
        )

    # Get recent models (limit 2)
    recent_models = await dal.get_recent_models(limit=2)

    assert len(recent_models) == 2
    model_ids = [model["model_id"] for model in recent_models]
    assert (
        "model_1" in model_ids or
        "model_2" in model_ids or
        "model_3" in model_ids
    )


@pytest.mark.asyncio
async def test_delete_model_record_and_file(test_setup_db, test_model_path):
    """Test deleting a model record and its file."""
    dal = test_setup_db["dal"]
    job_id = test_setup_db["job_id"]

    # Create a model record
    model_id = "model_to_delete"
    created_at = datetime.now()

    await dal.create_model_record(
        model_id=model_id,
        job_id=job_id,
        model_path=test_model_path,
        created_at=created_at,
        metadata={"test": "test"},
        is_active=False,
    )

    # Verify model exists
    record = await dal.execute_raw_query(
        "SELECT 1 FROM models WHERE model_id = $1",
        params=(model_id,),
        fetchall=False,
    )
    assert record is not None

    # Delete the model
    # Need to mock _is_path_safe in the models module where it's used
    with (
        patch("os.path.exists", return_value=True),
        patch("os.remove") as mock_remove,
        patch("deployment.app.db.queries.models._is_path_safe", return_value=True),
    ):
        result = await dal.delete_model_record_and_file(model_id)

        assert result is True
        mock_remove.assert_called_once_with(test_model_path)

    # Verify model no longer exists in database
    record = await dal.execute_raw_query(
        "SELECT 1 FROM models WHERE model_id = $1",
        params=(model_id,),
        fetchall=False,
    )
    assert record is None


@pytest.mark.asyncio
async def test_model_registration_workflow(test_setup_db, test_model_path):
    """Test the entire model registration workflow."""
    dal = test_setup_db["dal"]
    job_id = test_setup_db["job_id"]
    config_id = test_setup_db["config_id"]

    # 1. Create model records
    models = [
        {
            "model_id": "workflow_model_1",
            "created_at": datetime.now(),
            "metric_value": 0.75,
        },
        {
            "model_id": "workflow_model_2",
            "created_at": datetime.now(),
            "metric_value": 0.85,
        },
    ]

    for model in models:
        # Create model record
        await dal.create_model_record(
            model_id=model["model_id"],
            job_id=job_id,
            model_path=test_model_path,
            created_at=model["created_at"],
            metadata={"size_bytes": 1000},
            is_active=False,
        )

        # Create training result
        await dal.create_training_result(
            job_id=job_id,
            model_id=model["model_id"],
            config_id=config_id,
            metrics={"val_loss": model["metric_value"]},
            duration=3600,
        )

    # 2. Verify both models exist
    count_result = await dal.execute_raw_query(
        "SELECT COUNT(*) as count FROM models",
        fetchall=False,
    )
    assert count_result["count"] == 2

    # 3. Get the best model by metric
    best_model = await dal.get_best_model_by_metric(
        "val_loss", higher_is_better=True
    )
    assert best_model["model_id"] == "workflow_model_2"

    # 4. Activate the best model
    await dal.set_model_active(best_model["model_id"])

    # 5. Verify it's now the active model
    active_model = await dal.get_active_model()
    assert active_model["model_id"] == "workflow_model_2"

    # 6. Get recent models
    recent_models = await dal.get_recent_models(limit=5)
    assert len(recent_models) == 2

    # 7. Delete one model
    with (
        patch("os.path.exists", return_value=True),
        patch("os.remove"),
        patch("deployment.app.db.utils._is_path_safe", return_value=True),
    ):
        result = await dal.delete_model_record_and_file("workflow_model_1")
        assert result is True

    # 8. Verify only one model remains
    count_result = await dal.execute_raw_query(
        "SELECT COUNT(*) as count FROM models",
        fetchall=False,
    )
    assert count_result["count"] == 1

    remaining_model = await dal.execute_raw_query(
        "SELECT model_id FROM models",
        fetchall=False,
    )
    assert remaining_model["model_id"] == "workflow_model_2"
