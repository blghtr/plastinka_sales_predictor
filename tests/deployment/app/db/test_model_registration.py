import json
import os
import sqlite3
import time
from datetime import datetime
from unittest.mock import patch

import pytest

from deployment.app.db.database import (
    create_model_record,
    delete_model_record_and_file,
    get_active_model,
    get_best_model_by_metric,
    get_recent_models,
    set_model_active,
)


# Custom dict_factory function to use in tests
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_model_registry.db"
    return str(db_path)

@pytest.fixture
def create_test_db(test_db_path):
    """
    Initialize the test database with schema and yield the connection.
    This ensures the same connection is used throughout the test.
    """
    conn = sqlite3.connect(test_db_path)
    conn.row_factory = dict_factory  # Set row_factory to get dicts
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    # Use the official schema to ensure consistency
    from deployment.app.db.schema import SCHEMA_SQL
    cursor.executescript(SCHEMA_SQL)

    # Insert a test config first
    test_config_id = "test_config_id"
    cursor.execute(
        """
        INSERT INTO configs (config_id, config, created_at, is_active)
        VALUES (?, ?, ?, ?)
        """,
        (test_config_id, json.dumps({"test": "config"}), datetime.now().isoformat(), 0)
    )

    # Insert a test job that subsequent tests can rely on
    test_job_id = "test_job_id"
    cursor.execute(
        """
        INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, progress, config_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (test_job_id, "training", "completed", datetime.now().isoformat(), datetime.now().isoformat(), 100, test_config_id)
    )

    conn.commit()

    # Yield the connection so tests can use it
    yield conn

    # Teardown: close the connection
    conn.close()

    # Cleanup the file
    for _ in range(5):
        try:
            os.remove(test_db_path)
            break
        except PermissionError:
            time.sleep(0.1)

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

def get_new_connection(db_path):
    """Helper function to get a new database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def test_create_model_record(create_test_db, test_model_path, monkeypatch):
    """Test that a model record can be created in the database."""
    conn = create_test_db  # Use the connection from the fixture

    # Mock get_db_connection to return the yielded connection
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: conn)

    # Create test data
    model_id = "test_model_123"
    job_id = "test_job_id"
    created_at = datetime.now()
    metadata = {"test_key": "test_value", "file_size_bytes": 100}

    # Create the model record using the same connection context
    create_model_record(
        model_id=model_id,
        job_id=job_id,
        model_path=test_model_path,
        created_at=created_at,
        metadata=metadata,
        is_active=False,
        connection=conn
    )

    # Verification uses the same connection
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
    record = cursor.fetchone()

    assert record is not None
    assert record["model_id"] == model_id
    assert record["job_id"] == job_id
    assert record["model_path"] == test_model_path
    assert record["is_active"] == 0
    assert json.loads(record["metadata"]) == metadata

def test_activate_model(create_test_db, test_model_path, monkeypatch):
    """Test that a model can be activated and other models deactivated."""
    conn = create_test_db  # Use the connection from the fixture

    # Mock get_db_connection to return the yielded connection
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: conn)

    # Create multiple model records
    models = [
        {"model_id": "model_1", "is_active": False},
        {"model_id": "model_2", "is_active": False},
        {"model_id": "model_3", "is_active": False}
    ]

    job_id = "test_job_id"
    created_at = datetime.now()

    # Create all model records
    for model in models:
        create_model_record(
            model_id=model["model_id"],
            job_id=job_id,
            model_path=test_model_path,
            created_at=created_at,
            metadata={"test": "test"},
            is_active=model["is_active"],
            connection=conn
        )

    # Activate model_2 with the same connection
    result = set_model_active("model_2", deactivate_others=True, connection=conn)
    assert result is True

    # Get active model with the same connection
    active_model = get_active_model(connection=conn)
    assert active_model is not None
    assert active_model["model_id"] == "model_2"

    # Check other models are inactive
    cursor = conn.cursor()
    cursor.execute("SELECT model_id, is_active FROM models WHERE model_id != ?", ("model_2",))
    other_models = cursor.fetchall()

    for model in other_models:
        assert model["is_active"] == 0

def test_get_best_model_by_metric(create_test_db, test_model_path, monkeypatch):
    """Test retrieving the best model based on a metric."""
    conn = create_test_db # Use the connection from the fixture

    # Mock get_db_connection to return the yielded connection
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: conn)

    # Use the proper dict_factory implementation
    monkeypatch.setattr("deployment.app.db.database.dict_factory", dict_factory)

    # Create models
    models = [
        {"model_id": "model_1", "metric_value": 0.8},
        {"model_id": "model_2", "metric_value": 0.9},  # Best model
        {"model_id": "model_3", "metric_value": 0.7}
    ]

    job_id = "test_job_id"
    created_at = datetime.now()

    # Create model records and training results
    for model in models:
        # Create model record
        create_model_record(
            model_id=model["model_id"],
            job_id=job_id,
            model_path=test_model_path,
            created_at=created_at,
            metadata={"test": "test"},
            is_active=False,
            connection=conn
        )

        # Create training result with the same connection
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO training_results (result_id, job_id, model_id, metrics)
            VALUES (?, ?, ?, ?)
            """,
            (
                f"result_{model['model_id']}",
                job_id,
                model["model_id"],
                json.dumps({"accuracy": model["metric_value"]})
            )
        )
        conn.commit()

    # Get the best model by accuracy with the same connection
    best_model = get_best_model_by_metric("accuracy", higher_is_better=True, connection=conn)

    assert best_model is not None
    assert best_model["model_id"] == "model_2"
    assert best_model["metrics"]["accuracy"] == 0.9

def test_get_recent_models(create_test_db, test_model_path, monkeypatch):
    """Test retrieving recent models."""
    conn = create_test_db # Use the connection from the fixture

    # Mock get_db_connection to return the yielded connection
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: conn)

    # Create models with different timestamps - adding a small delay
    now = datetime.now()

    # Create models in reverse order to ensure model_3 is most recent
    models = [
        {"model_id": "model_3", "created_at": now, "is_active": False},
        {"model_id": "model_2", "created_at": now, "is_active": False},
        {"model_id": "model_1", "created_at": now, "is_active": True}
    ]

    job_id = "test_job_id"

    # Create all model records with deliberate pauses
    for model in models:
        time.sleep(0.05)  # More significant delay to ensure different timestamps
        create_model_record(
            model_id=model["model_id"],
            job_id=job_id,
            model_path=test_model_path,
            created_at=datetime.now(),  # Use current time, not the initial time
            metadata={"test": "test"},
            is_active=model["is_active"],
            connection=conn
        )

    # Get recent models (limit 2) with the same connection
    recent_models = get_recent_models(limit=2, connection=conn)

    assert len(recent_models) == 2
    model_ids = [model['model_id'] for model in recent_models]
    assert "model_1" in model_ids, f"Expected model_1 in {model_ids}"
    assert "model_2" in model_ids, f"Expected model_2 in {model_ids}"

def test_delete_model_record_and_file(create_test_db, test_model_path, monkeypatch):
    """Test deleting a model record and its file."""
    conn = create_test_db # Use the connection from the fixture

    # Mock get_db_connection to return the yielded connection
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: conn)

    # Create a model record
    model_id = "model_to_delete"
    job_id = "test_job_id"
    created_at = datetime.now()

    create_model_record(
        model_id=model_id,
        job_id=job_id,
        model_path=test_model_path,
        created_at=created_at,
        metadata={"test": "test"},
        is_active=False,
        connection=conn
    )

    # Verify model exists with the same connection
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,))
    assert cursor.fetchone() is not None

    # Delete the model
    with patch("os.path.exists", return_value=True):
        with patch("os.remove") as mock_remove:
            result = delete_model_record_and_file(model_id, connection=conn)

            assert result is True
            mock_remove.assert_called_once_with(test_model_path)

    # Verify model no longer exists in database with the same connection
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,))
    assert cursor.fetchone() is None

def test_model_registration_workflow(create_test_db, test_model_path, monkeypatch):
    """Test the entire model registration workflow."""
    conn = create_test_db # Use the connection from the fixture

    # Mock get_db_connection to return the yielded connection
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: conn)

    # Use the proper dict_factory implementation
    monkeypatch.setattr("deployment.app.db.database.dict_factory", dict_factory)

    # 1. Create model records
    models = [
        {"model_id": "workflow_model_1", "created_at": datetime.now(), "metric_value": 0.75},
        {"model_id": "workflow_model_2", "created_at": datetime.now(), "metric_value": 0.85}
    ]

    job_id = "test_job_id"

    for model in models:
        # Create model record
        create_model_record(
            model_id=model["model_id"],
            job_id=job_id,
            model_path=test_model_path,
            created_at=model["created_at"],
            metadata={"size_bytes": 1000},
            is_active=False,
            connection=conn
        )

        # Create training result with the same connection
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO training_results (result_id, job_id, model_id, metrics)
            VALUES (?, ?, ?, ?)
            """,
            (
                f"result_{model['model_id']}",
                job_id,
                model["model_id"],
                json.dumps({"accuracy": model["metric_value"]})
            )
        )
        conn.commit()

    # 2. Verify both models exist with the same connection
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM models")
    assert cursor.fetchone()['COUNT(*)'] == 2

    # 3. Get the best model by metric with the same connection
    best_model = get_best_model_by_metric("accuracy", higher_is_better=True, connection=conn)
    assert best_model["model_id"] == "workflow_model_2"

    # 4. Activate the best model with the same connection
    set_model_active(best_model["model_id"], connection=conn)

    # 5. Verify it's now the active model with the same connection
    active_model = get_active_model(connection=conn)
    assert active_model["model_id"] == "workflow_model_2"

    # 6. Get recent models with the same connection
    recent_models = get_recent_models(limit=5, connection=conn)
    assert len(recent_models) == 2

    # 7. Delete one model
    with patch("os.path.exists", return_value=True):
        with patch("os.remove"):
            result = delete_model_record_and_file("workflow_model_1", connection=conn)
            assert result is True

    # 8. Verify only one model remains with the same connection
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM models")
    assert cursor.fetchone()['COUNT(*)'] == 1
    cursor.execute("SELECT model_id FROM models")
    assert cursor.fetchone()['model_id'] == "workflow_model_2"
