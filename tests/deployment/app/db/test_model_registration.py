import pytest
import os
import json
import sqlite3
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime
import time

from deployment.app.db.database import (
    create_model_record,
    get_active_model,
    set_model_active,
    get_best_model_by_metric,
    get_recent_models,
    delete_model_record_and_file
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
    """Initialize the test database with schema."""
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    
    # Create minimal schema needed for testing model registration
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            parameters TEXT,
            result_id TEXT,
            error_message TEXT,
            progress REAL
        );
        
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            model_path TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            metadata TEXT,
            is_active BOOLEAN DEFAULT 0,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id)
        );
        
        CREATE TABLE IF NOT EXISTS training_results (
            result_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            model_id TEXT,
            parameter_set_id TEXT,
            metrics TEXT,
            parameters TEXT,
            duration INTEGER,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        );
    """)
    
    # Insert a test job
    cursor.execute(
        """
        INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, progress)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("test_job_id", "training", "completed", datetime.now(), datetime.now(), 100)
    )
    
    conn.commit()
    conn.close()
    
    yield test_db_path
    
    # Make sure all connections are closed before attempting to remove
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
    db_path = create_test_db
    
    # Mock get_db_connection to return a new connection each time
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: get_new_connection(db_path))
    
    # Create test data
    model_id = "test_model_123"
    job_id = "test_job_id"
    created_at = datetime.now()
    metadata = {"test_key": "test_value", "file_size_bytes": 100}
    
    # Create the model record
    create_model_record(
        model_id=model_id,
        job_id=job_id,
        model_path=test_model_path,
        created_at=created_at,
        metadata=metadata,
        is_active=False
    )
    
    # Create a new connection for verification
    conn = get_new_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
    record = cursor.fetchone()
    
    assert record is not None
    assert record["model_id"] == model_id
    assert record["job_id"] == job_id
    assert record["model_path"] == test_model_path
    assert record["is_active"] == 0
    assert json.loads(record["metadata"]) == metadata
    
    conn.close()

def test_activate_model(create_test_db, test_model_path, monkeypatch):
    """Test that a model can be activated and other models deactivated."""
    db_path = create_test_db
    
    # Mock get_db_connection to return a new connection each time
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: get_new_connection(db_path))
    
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
            is_active=model["is_active"]
        )
    
    # Activate model_2 with a new connection
    result = set_model_active("model_2", deactivate_others=True)
    assert result is True
    
    # Get active model with a new connection
    conn = get_new_connection(db_path)
    active_model = get_active_model(conn)
    assert active_model is not None
    assert active_model["model_id"] == "model_2"
    
    # Check other models are inactive
    cursor = conn.cursor()
    cursor.execute("SELECT model_id, is_active FROM models WHERE model_id != ?", ("model_2",))
    other_models = cursor.fetchall()
    
    for model in other_models:
        assert model["is_active"] == 0
    
    conn.close()

def test_get_best_model_by_metric(create_test_db, test_model_path, monkeypatch):
    """Test retrieving the best model based on a metric."""
    db_path = create_test_db
    
    # Mock get_db_connection to return a new connection each time
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: get_new_connection(db_path))
    
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
            is_active=False
        )
        
        # Create training result with a new connection
        conn = get_new_connection(db_path)
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
        conn.close()
    
    # Get the best model by accuracy with a new connection
    conn = get_new_connection(db_path)
    conn.row_factory = dict_factory  # Set the row_factory on the connection as well
    best_model = get_best_model_by_metric("accuracy", higher_is_better=True, connection=conn)
    
    assert best_model is not None
    assert best_model["model_id"] == "model_2"
    # The get_best_model_by_metric function already parsed the JSON for us
    assert best_model["metrics"]["accuracy"] == 0.9
    
    conn.close()

def test_get_recent_models(create_test_db, test_model_path, monkeypatch):
    """Test retrieving recent models."""
    db_path = create_test_db
    
    # Mock get_db_connection to return a new connection each time
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: get_new_connection(db_path))
    
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
            is_active=model["is_active"]
        )
    
    # Refresh the database and wait for a moment
    time.sleep(0.1)
    
    # Get recent models (limit 2) with a new connection
    with patch("deployment.app.db.database.get_db_connection", lambda: get_new_connection(db_path)):
        recent_models = get_recent_models(limit=2)
    
    assert len(recent_models) == 2
    # We created the models in reverse order (model_3, model_2, model_1)
    # So the most recent should be model_1 and model_2
    model_ids = [model[0] for model in recent_models]
    assert "model_1" in model_ids, f"Expected model_1 in {model_ids}"
    assert "model_2" in model_ids, f"Expected model_2 in {model_ids}"

def test_delete_model_record_and_file(create_test_db, test_model_path, monkeypatch):
    """Test deleting a model record and its file."""
    db_path = create_test_db
    
    # Mock get_db_connection to return a new connection each time
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: get_new_connection(db_path))
    
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
        is_active=False
    )
    
    # Verify model exists with a new connection
    conn = get_new_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,))
    assert cursor.fetchone() is not None
    conn.close()
    
    # Delete the model
    with patch("os.path.exists", return_value=True):
        with patch("os.remove") as mock_remove:
            result = delete_model_record_and_file(model_id)
            
            assert result is True
            mock_remove.assert_called_once_with(test_model_path)
    
    # Verify model no longer exists in database with a new connection
    conn = get_new_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,))
    assert cursor.fetchone() is None
    conn.close()

def test_model_registration_workflow(create_test_db, test_model_path, monkeypatch):
    """Test the entire model registration workflow."""
    db_path = create_test_db
    
    # Mock get_db_connection to return a new connection each time
    monkeypatch.setattr("deployment.app.db.database.get_db_connection", lambda: get_new_connection(db_path))
    
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
            is_active=False
        )
        
        # Create training result with a new connection
        conn = get_new_connection(db_path)
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
        conn.close()
    
    # 2. Verify both models exist with a new connection
    conn = get_new_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM models")
    assert cursor.fetchone()[0] == 2
    conn.close()
    
    # 3. Get the best model by metric with a new connection
    conn = get_new_connection(db_path)
    conn.row_factory = dict_factory  # Set the row_factory for this connection
    best_model = get_best_model_by_metric("accuracy", higher_is_better=True, connection=conn)
    assert best_model["model_id"] == "workflow_model_2"
    conn.close()
    
    # 4. Activate the best model with a new connection
    set_model_active(best_model["model_id"])
    
    # 5. Verify it's now the active model with a new connection
    conn = get_new_connection(db_path)
    active_model = get_active_model(connection=conn)
    assert active_model["model_id"] == "workflow_model_2"
    conn.close()
    
    # 6. Get recent models with a new connection
    with patch("deployment.app.db.database.get_db_connection", lambda: get_new_connection(db_path)):
        recent_models = get_recent_models(limit=5)
    assert len(recent_models) == 2
    
    # 7. Delete one model
    with patch("os.path.exists", return_value=True):
        with patch("os.remove"):
            result = delete_model_record_and_file("workflow_model_1")
            assert result is True
    
    # 8. Verify only one model remains with a new connection
    conn = get_new_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM models")
    assert cursor.fetchone()[0] == 1
    cursor.execute("SELECT model_id FROM models")
    assert cursor.fetchone()[0] == "workflow_model_2"
    conn.close() 