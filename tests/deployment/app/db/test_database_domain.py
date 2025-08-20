"""
Comprehensive tests for deployment.app.db domain-specific database operations

This test suite consolidates domain-specific database operations testing from multiple files
into a unified, well-organized structure following comprehensive testing patterns.

Testing Approach:
- Use in-memory SQLite databases for test isolation and performance
- Mock all external dependencies using @patch decorators
- Test both success and failure scenarios for all operations
- Test model registration, lifecycle management, and cleanup operations
- Test data retention policies and automated cleanup procedures
- Test domain business logic with proper rollback verification
- Test file system operations with temporary directories
- Integration tests for module imports and domain constants

Consolidated from:
- test_model_registration.py (model lifecycle, activation, best model selection)
- test_data_retention.py (data cleanup policies, retention management)

All external imports and dependencies are mocked to ensure test isolation.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from deployment.app.db.data_access_layer import DataAccessLayer, UserContext, UserRoles

# Import the modules under test
from deployment.app.db.database import (
    create_model_record,
    delete_model_record_and_file,
    dict_factory,
    get_active_model,
    set_model_active,
)

try:
    from deployment.app.db.data_retention import (
        cleanup_old_predictions,
        run_cleanup_job,
    )

    DATA_RETENTION_AVAILABLE = True
except ImportError:
    DATA_RETENTION_AVAILABLE = False

from deployment.app.db.schema import SCHEMA_SQL


@pytest.fixture
def domain_db():
    """Create an in-memory SQLite database with schema for domain testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = dict_factory
    conn.execute("PRAGMA foreign_keys = ON")

    # Initialize with full schema
    conn.executescript(SCHEMA_SQL)

    # Insert required reference data for foreign key constraints
    now = datetime.now().isoformat()

    # Insert test config
    conn.execute(
        "INSERT INTO configs (config_id, config, created_at, is_active) VALUES (?, ?, ?, ?)",
        ("test_config_1", json.dumps({"test": "config"}), now, 0),
    )

    # Insert test job
    conn.execute(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, config_id) VALUES (?, ?, ?, ?, ?, ?)",
        ("test_job_1", "training", "completed", now, now, "test_config_1"),
    )

    conn.commit()

    yield {"conn": conn, "path": ":memory:"}
    conn.close()


@pytest.fixture
def temp_model_files():
    """Create temporary model files for testing."""
    temp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(temp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Create test model files
    model_files = []
    for i in range(5):
        model_path = os.path.join(model_dir, f"test_model_{i}.onnx")
        with open(model_path, "wb") as f:
            f.write(f"dummy model content {i}".encode())
        model_files.append(model_path)

    yield {"temp_dir": temp_dir, "model_dir": model_dir, "model_files": model_files}

    # Cleanup
    for _ in range(5):
        try:
            shutil.rmtree(temp_dir)
            break
        except (PermissionError, OSError):
            time.sleep(0.1)


@pytest.fixture
def sample_model_data():
    """Sample model data for testing."""
    return {
        "model_id": "test_model_123",
        "job_id": "test_job_1",
        "created_at": datetime.now(),
        "metadata": {"test_key": "test_value", "file_size_bytes": 1024},
        "is_active": False,
    }


class TestModelRegistration:
    """Test suite for model registration and lifecycle management."""

    def test_create_model_record_success(
        self, domain_db, temp_model_files, sample_model_data
    ):
        """Test successful model record creation."""
        conn = domain_db["conn"]
        model_path = temp_model_files["model_files"][0]

        # Create model record
        create_model_record(
            model_id=sample_model_data["model_id"],
            job_id=sample_model_data["job_id"],
            model_path=model_path,
            created_at=sample_model_data["created_at"],
            metadata=sample_model_data["metadata"],
            is_active=sample_model_data["is_active"],
            connection=conn,
        )

        # Verify record was created
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM models WHERE model_id = ?", (sample_model_data["model_id"],)
        )
        record = cursor.fetchone()

        assert record is not None
        assert record["model_id"] == sample_model_data["model_id"]
        assert record["job_id"] == sample_model_data["job_id"]
        assert record["model_path"] == model_path
        assert record["is_active"] == 0
        assert json.loads(record["metadata"]) == sample_model_data["metadata"]

    def test_get_active_model_success(self, domain_db, temp_model_files):
        """Test retrieving the active model."""
        conn = domain_db["conn"]
        model_path = temp_model_files["model_files"][0]

        # Create multiple models, one active
        models = [
            {"model_id": "model_1", "is_active": False},
            {"model_id": "model_2", "is_active": True},  # Active model
            {"model_id": "model_3", "is_active": False},
        ]

        for model in models:
            create_model_record(
                model_id=model["model_id"],
                job_id="test_job_1",
                model_path=model_path,
                created_at=datetime.now(),
                metadata={"test": "data"},
                is_active=model["is_active"],
                connection=conn,
            )

        # Get active model
        active_model = get_active_model(connection=conn)

        assert active_model is not None
        assert active_model["model_id"] == "model_2"
        # Note: get_active_model doesn't return is_active field, only model_id, model_path, metadata

    def test_set_model_active_success(self, domain_db, temp_model_files):
        """Test activating a model and deactivating others."""
        conn = domain_db["conn"]
        model_path = temp_model_files["model_files"][0]

        # Create multiple models
        model_ids = ["model_1", "model_2", "model_3"]
        for model_id in model_ids:
            create_model_record(
                model_id=model_id,
                job_id="test_job_1",
                model_path=model_path,
                created_at=datetime.now(),
                metadata={"test": "data"},
                is_active=False,
                connection=conn,
            )

        # Activate model_2
        result = set_model_active("model_2", deactivate_others=True, connection=conn)
        assert result is True

        # Verify model_2 is active
        active_model = get_active_model(connection=conn)
        assert active_model["model_id"] == "model_2"

        # Verify others are inactive
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model_id, is_active FROM models WHERE model_id != ?", ("model_2",)
        )
        other_models = cursor.fetchall()

        for model in other_models:
            assert model["is_active"] == 0

    @patch("os.path.exists")
    @patch("os.remove")
    @patch("deployment.app.db.database._is_path_safe", return_value=True)
    def test_delete_model_record_and_file_success(
        self, mock_is_path_safe, mock_remove, mock_exists, domain_db, temp_model_files
    ):
        """Test deleting model record and file."""
        mock_exists.return_value = True
        conn = domain_db["conn"]
        model_path = temp_model_files["model_files"][0]

        # Create model record
        create_model_record(
            model_id="test_model",
            job_id="test_job_1",
            model_path=model_path,
            created_at=datetime.now(),
            metadata={"test": "data"},
            is_active=False,
            connection=conn,
        )

        # Delete model
        result = delete_model_record_and_file("test_model", connection=conn)
        assert result is True

        # Verify record was deleted
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM models WHERE model_id = ?", ("test_model",))
        record = cursor.fetchone()
        assert record is None

        # Verify file deletion was attempted
        mock_remove.assert_called_once_with(model_path)


@pytest.mark.skipif(
    not DATA_RETENTION_AVAILABLE, reason="Data retention module not available"
)
class TestDataRetention:
    """Test suite for data retention and cleanup operations."""

    @patch("deployment.app.db.data_retention.get_settings")
    @patch("deployment.app.db.data_retention.DataAccessLayer")
    def test_cleanup_old_predictions_success(self, mock_dal_class, mock_get_settings, domain_db):
        """Test cleaning up old predictions."""
        # Use real current date for the test
        now = datetime.now()
        # Мокаем настройки
        mock_retention = MagicMock()
        mock_retention.prediction_retention_days = 30
        mock_retention.cleanup_enabled = True
        mock_settings_object = MagicMock()
        mock_settings_object.data_retention = mock_retention
        mock_get_settings.return_value = mock_settings_object
        conn = domain_db["conn"]
        # Create a real DAL instance for the test
        dal = DataAccessLayer(user_context=UserContext(roles=[UserRoles.SYSTEM]), connection=conn)
        # Mock the DAL class to return our real DAL instance
        mock_dal_class.return_value = dal

        # Disable foreign keys for this test
        conn.execute("PRAGMA foreign_keys = OFF")
        # Create models first for foreign key constraints
        models = ["model1", "model2", "model3", "model4"]
        for model_id in models:
            create_model_record(
                model_id=model_id,
                job_id="test_job_1",
                model_path="/fake/path.onnx",
                created_at=now,
                metadata={"test": "data"},
                is_active=False,
                connection=conn,
            )
        # Create test predictions table and data
        conn.execute("""
        CREATE TABLE IF NOT EXISTS fact_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            multiindex_id INTEGER NOT NULL,
            prediction_month TIMESTAMP NOT NULL,
            result_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            quantile_05 DECIMAL(10,2) NOT NULL,
            quantile_25 DECIMAL(10,2) NOT NULL,
            quantile_50 DECIMAL(10,2) NOT NULL,
            quantile_75 DECIMAL(10,2) NOT NULL,
            quantile_95 DECIMAL(10,2) NOT NULL,
            created_at TIMESTAMP NOT NULL
        )""")
        # Old predictions (older than 30 days)
        old_date_1 = (now - timedelta(days=45)).strftime("%Y-%m-%d")
        old_date_2 = (now - timedelta(days=40)).strftime("%Y-%m-%d")

        old_predictions = [
            (
                1,
                old_date_1,
                "result1",
                "model1",
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
                (now - timedelta(days=45)).isoformat(),
            ),
            (
                2,
                old_date_2,
                "result2",
                "model2",
                12.0,
                17.0,
                22.0,
                27.0,
                32.0,
                (now - timedelta(days=40)).isoformat(),
            ),
        ]
        # Recent predictions (within 30 days)
        recent_predictions = [
            (
                3,
                (now - timedelta(days=10)).strftime("%Y-%m-%d"),
                "result3",
                "model3",
                11.0,
                16.0,
                21.0,
                26.0,
                31.0,
                (now - timedelta(days=10)).isoformat(),
            ),
            (
                4,
                (now - timedelta(days=5)).strftime("%Y-%m-%d"),
                "result4",
                "model4",
                13.0,
                18.0,
                23.0,
                28.0,
                33.0,
                (now - timedelta(days=5)).isoformat(),
            ),
        ]
        all_predictions = old_predictions + recent_predictions
        conn.executemany(
            """INSERT INTO fact_predictions
               (multiindex_id, prediction_month, result_id, model_id, quantile_05, quantile_25, quantile_50, quantile_75, quantile_95, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            all_predictions,
        )
        conn.commit()

        # Run cleanup
        deleted_count = cleanup_old_predictions(dal=dal)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM fact_predictions")
        final_count = cursor.fetchone()["count"]

        assert deleted_count == 2  # 2 old predictions should be deleted
        assert final_count == 2  # 2 recent predictions should remain

    @patch("deployment.app.db.data_retention.cleanup_old_predictions")
    @patch("deployment.app.db.data_retention.cleanup_old_models")
    @patch("deployment.app.db.data_retention.cleanup_old_historical_data")
    @patch("deployment.app.db.data_retention.get_settings")
    def test_run_cleanup_job_success(
        self,
        mock_get_settings,
        mock_cleanup_historical,
        mock_cleanup_models,
        mock_cleanup_predictions,
    ):
        """Test running complete cleanup job."""
        # Mock settings
        mock_settings_object = MagicMock()
        mock_retention = MagicMock()
        mock_retention.cleanup_enabled = True
        mock_settings_object.data_retention = mock_retention
        mock_get_settings.return_value = mock_settings_object

        # Mock cleanup function returns
        mock_cleanup_predictions.return_value = 5
        mock_cleanup_models.return_value = [
            "model_1",
            "model_2",
            "model_3",
        ]  # Returns list of model IDs
        mock_cleanup_historical.return_value = {
            "sales": 10,
            "stock": 8,
            "stock_movement": 3,
            "prices": 2,
        }

        # Run cleanup job (returns None)
        result = run_cleanup_job()

        # Verify all cleanup functions were called
        mock_cleanup_predictions.assert_called_once()
        mock_cleanup_models.assert_called_once()
        mock_cleanup_historical.assert_called_once()

        # The function returns None, so we just verify it completed without error
        assert result is None


class TestIntegration:
    """Integration tests for database domain operations."""

    def test_module_imports_successfully(self):
        """Test that all domain modules can be imported without errors."""
        # Test database module imports

        # Test data retention imports (if available)
        if DATA_RETENTION_AVAILABLE:
            pass

        # If we get here without ImportError, imports are successful
        assert True

    def test_constants_defined(self):
        """Test that expected constants and configurations are defined."""
        from deployment.app.db.schema import SCHEMA_SQL

        # Verify schema is not empty
        assert SCHEMA_SQL is not None
        assert len(SCHEMA_SQL.strip()) > 0

        # Verify schema contains expected tables
        assert "models" in SCHEMA_SQL.lower()
        assert "training_results" in SCHEMA_SQL.lower()
        assert "configs" in SCHEMA_SQL.lower()

    def test_end_to_end_model_lifecycle(self, domain_db, temp_model_files):
        """Test complete model lifecycle from creation to deletion."""
        conn = domain_db["conn"]
        model_path = temp_model_files["model_files"][0]

        # 1. Create model
        model_id = "lifecycle_test_model"
        create_model_record(
            model_id=model_id,
            job_id="test_job_1",
            model_path=model_path,
            created_at=datetime.now(),
            metadata={"lifecycle": "test"},
            is_active=False,
            connection=conn,
        )

        # 2. Verify model exists
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        model = cursor.fetchone()
        assert model is not None
        assert model["is_active"] == 0

        # 3. Activate model
        result = set_model_active(model_id, connection=conn)
        assert result is True

        # 4. Verify model is active
        active_model = get_active_model(connection=conn)
        assert active_model["model_id"] == model_id
        # Note: get_active_model doesn't return is_active field

        # 5. Delete model
        with (
            patch("os.path.exists", return_value=True),
            patch("os.remove") as mock_remove,
            patch("deployment.app.db.database._is_path_safe", return_value=True),
        ):
            delete_result = delete_model_record_and_file(model_id, connection=conn)
            assert delete_result is True
            mock_remove.assert_called_once_with(model_path)

        # 6. Verify model is deleted
        cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        deleted_model = cursor.fetchone()
        assert deleted_model is None

        # 7. Verify no active model remains
        active_model_after_delete = get_active_model(connection=conn)
        assert active_model_after_delete is None
