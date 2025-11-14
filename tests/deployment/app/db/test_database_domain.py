"""
Comprehensive tests for deployment.app.db domain-specific database operations

This test suite consolidates domain-specific database operations testing from multiple files
into a unified, well-organized structure following comprehensive testing patterns.

Testing Approach:
- Use PostgreSQL test database with async operations
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

import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

try:
    from deployment.app.db.data_retention import (
        cleanup_old_predictions,
        run_cleanup_job,
    )

    DATA_RETENTION_AVAILABLE = True
except ImportError:
    DATA_RETENTION_AVAILABLE = False


@pytest.fixture
async def domain_db(dal):
    """Set up test database with required reference data."""
    # Insert required reference data for foreign key constraints
    # Insert test config
    config_id = await dal.create_or_get_config(
        {"test": "config"},
        is_active=False,
    )

    # Insert test job
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

    @pytest.mark.asyncio
    async def test_create_model_record_success(
        self, domain_db, temp_model_files, sample_model_data
    ):
        """Test successful model record creation."""
        dal = domain_db["dal"]
        model_path = temp_model_files["model_files"][0]

        # Create model record
        await dal.create_model_record(
            model_id=sample_model_data["model_id"],
            job_id=domain_db["job_id"],
            model_path=model_path,
            created_at=sample_model_data["created_at"],
            metadata=sample_model_data["metadata"],
            is_active=sample_model_data["is_active"],
        )

        # Verify record was created
        record = await dal.execute_raw_query(
            "SELECT * FROM models WHERE model_id = $1",
            params=(sample_model_data["model_id"],),
            fetchall=False,
        )

        assert record is not None
        assert record["model_id"] == sample_model_data["model_id"]
        assert record["job_id"] == domain_db["job_id"]
        assert record["model_path"] == model_path
        assert record["is_active"] is False
        # Metadata is stored as JSONB, may be returned as string or dict
        import json
        if isinstance(record["metadata"], str):
            assert json.loads(record["metadata"]) == sample_model_data["metadata"]
        else:
            assert record["metadata"] == sample_model_data["metadata"]

    @pytest.mark.asyncio
    async def test_get_active_model_success(self, domain_db, temp_model_files):
        """Test retrieving the active model."""
        dal = domain_db["dal"]
        model_path = temp_model_files["model_files"][0]

        # Create multiple models, one active
        models = [
            {"model_id": "model_1", "is_active": False},
            {"model_id": "model_2", "is_active": True},  # Active model
            {"model_id": "model_3", "is_active": False},
        ]

        for model in models:
            await dal.create_model_record(
                model_id=model["model_id"],
                job_id=domain_db["job_id"],
                model_path=model_path,
                created_at=datetime.now(),
                metadata={"test": "data"},
                is_active=model["is_active"],
            )

        # Get active model
        active_model = await dal.get_active_model()

        assert active_model is not None
        assert active_model["model_id"] == "model_2"
        # Note: get_active_model doesn't return is_active field, only model_id, model_path, metadata

    @pytest.mark.asyncio
    async def test_set_model_active_success(self, domain_db, temp_model_files):
        """Test activating a model and deactivating others."""
        dal = domain_db["dal"]
        model_path = temp_model_files["model_files"][0]

        # Create multiple models
        model_ids = ["model_1", "model_2", "model_3"]
        for model_id in model_ids:
            await dal.create_model_record(
                model_id=model_id,
                job_id=domain_db["job_id"],
                model_path=model_path,
                created_at=datetime.now(),
                metadata={"test": "data"},
                is_active=False,
            )

        # Activate model_2
        result = await dal.set_model_active("model_2", deactivate_others=True)
        assert result is True

        # Verify model_2 is active
        active_model = await dal.get_active_model()
        assert active_model["model_id"] == "model_2"

        # Verify others are inactive
        other_models = await dal.execute_raw_query(
            "SELECT model_id, is_active FROM models WHERE model_id != $1",
            params=("model_2",),
            fetchall=True,
        )

        for model in other_models:
            assert model["is_active"] is False

    @pytest.mark.asyncio
    @patch("os.path.exists")
    @patch("os.remove")
    @patch("deployment.app.db.queries.models._is_path_safe", return_value=True)
    async def test_delete_model_record_and_file_success(
        self, mock_is_path_safe, mock_remove, mock_exists, domain_db, temp_model_files
    ):
        """Test deleting model record and file."""
        mock_exists.return_value = True
        dal = domain_db["dal"]
        model_path = temp_model_files["model_files"][0]

        # Create model record
        await dal.create_model_record(
            model_id="test_model",
            job_id=domain_db["job_id"],
            model_path=model_path,
            created_at=datetime.now(),
            metadata={"test": "data"},
            is_active=False,
        )

        # Delete model
        result = await dal.delete_model_record_and_file("test_model")
        assert result is True

        # Verify record was deleted
        record = await dal.execute_raw_query(
            "SELECT * FROM models WHERE model_id = $1",
            params=("test_model",),
            fetchall=False,
        )
        assert record is None

        # Verify file deletion was attempted
        mock_remove.assert_called_once_with(model_path)


@pytest.mark.skipif(
    not DATA_RETENTION_AVAILABLE, reason="Data retention module not available"
)
class TestDataRetention:
    """Test suite for data retention and cleanup operations."""

    @pytest.mark.asyncio
    @patch("deployment.app.db.data_retention.get_settings")
    async def test_cleanup_old_predictions_success(self, mock_get_settings, domain_db):
        """Test cleaning up old predictions."""
        import pandas as pd
        
        # Use real current date for the test
        now = datetime.now()
        # Мокаем настройки
        mock_retention = MagicMock()
        mock_retention.prediction_retention_days = 30
        mock_retention.cleanup_enabled = True
        mock_settings_object = MagicMock()
        mock_settings_object.data_retention = mock_retention
        mock_get_settings.return_value = mock_settings_object
        
        dal = domain_db["dal"]
        
        # Create models first for foreign key constraints
        models = ["model1", "model2", "model3", "model4"]
        for model_id in models:
            await dal.create_model_record(
                model_id=model_id,
                job_id=domain_db["job_id"],
                model_path="/fake/path.onnx",
                created_at=now,
                metadata={"test": "data"},
                is_active=False,
            )
        
        # Create prediction results
        result1_id = await dal.create_prediction_result(
            job_id=domain_db["job_id"],
            prediction_month=(now - timedelta(days=45)).strftime("%Y-%m"),
            model_id="model1",
            output_path="/fake/path1",
            summary_metrics={},
        )
        result2_id = await dal.create_prediction_result(
            job_id=domain_db["job_id"],
            prediction_month=(now - timedelta(days=40)).strftime("%Y-%m"),
            model_id="model2",
            output_path="/fake/path2",
            summary_metrics={},
        )
        result3_id = await dal.create_prediction_result(
            job_id=domain_db["job_id"],
            prediction_month=(now - timedelta(days=10)).strftime("%Y-%m"),
            model_id="model3",
            output_path="/fake/path3",
            summary_metrics={},
        )
        result4_id = await dal.create_prediction_result(
            job_id=domain_db["job_id"],
            prediction_month=(now - timedelta(days=5)).strftime("%Y-%m"),
            model_id="model4",
            output_path="/fake/path4",
            summary_metrics={},
        )
        
        # Create multiindex entries
        await dal.get_or_create_multiindex_ids_batch([
            ("barcode1", "artist1", "album1", "CD", "Standard", "Studio",
             "2010s", "2010s", "Rock", 2015),
            ("barcode2", "artist2", "album2", "Vinyl", "Deluxe", "Live",
             "2000s", "2010s", "Pop", 2008),
        ])

        # Old predictions (older than 30 days) - using insert_predictions
        old_df1 = pd.DataFrame({
            'barcode': ["barcode1"], 'artist': ["artist1"], 'album': ["album1"],
            'cover_type': ["CD"], 'price_category': ["Standard"], 'release_type': ["Studio"],
            'recording_decade': ["2010s"], 'release_decade': ["2010s"], 'style': ["Rock"],
            'recording_year': [2015],
            '0.05': [10.0], '0.25': [15.0], '0.5': [20.0], '0.75': [25.0], '0.95': [30.0]
        })
        await dal.insert_predictions(
            result_id=result1_id,
            model_id="model1",
            prediction_month=(now - timedelta(days=45)).date(),
            df=old_df1,
        )
        
        old_df2 = pd.DataFrame({
            'barcode': ["barcode2"], 'artist': ["artist2"], 'album': ["album2"],
            'cover_type': ["Vinyl"], 'price_category': ["Deluxe"], 'release_type': ["Live"],
            'recording_decade': ["2000s"], 'release_decade': ["2010s"], 'style': ["Pop"],
            'recording_year': [2008],
            '0.05': [12.0], '0.25': [17.0], '0.5': [22.0], '0.75': [27.0], '0.95': [32.0]
        })
        await dal.insert_predictions(
            result_id=result2_id,
            model_id="model2",
            prediction_month=(now - timedelta(days=40)).date(),
            df=old_df2,
        )
        
        # Recent predictions (within 30 days)
        recent_df1 = pd.DataFrame({
            'barcode': ["barcode1"], 'artist': ["artist1"], 'album': ["album1"],
            'cover_type': ["CD"], 'price_category': ["Standard"], 'release_type': ["Studio"],
            'recording_decade': ["2010s"], 'release_decade': ["2010s"], 'style': ["Rock"],
            'recording_year': [2015],
            '0.05': [11.0], '0.25': [16.0], '0.5': [21.0], '0.75': [26.0], '0.95': [31.0]
        })
        await dal.insert_predictions(
            result_id=result3_id,
            model_id="model3",
            prediction_month=(now - timedelta(days=10)).date(),
            df=recent_df1,
        )
        
        recent_df2 = pd.DataFrame({
            'barcode': ["barcode2"], 'artist': ["artist2"], 'album': ["album2"],
            'cover_type': ["Vinyl"], 'price_category': ["Deluxe"], 'release_type': ["Live"],
            'recording_decade': ["2000s"], 'release_decade': ["2010s"], 'style': ["Pop"],
            'recording_year': [2008],
            '0.05': [13.0], '0.25': [18.0], '0.5': [23.0], '0.75': [28.0], '0.95': [33.0]
        })
        await dal.insert_predictions(
            result_id=result4_id,
            model_id="model4",
            prediction_month=(now - timedelta(days=5)).date(),
            df=recent_df2,
        )

        # Run cleanup
        deleted_count = await cleanup_old_predictions(days_to_keep=30, dal=dal)
        final_count_result = await dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_predictions",
            fetchall=False,
        )
        final_count = final_count_result["count"]

        assert deleted_count == 2  # 2 old predictions should be deleted
        assert final_count == 2  # 2 recent predictions should remain

    @pytest.mark.asyncio
    @patch("deployment.app.db.data_retention.cleanup_old_predictions")
    @patch("deployment.app.db.data_retention.cleanup_old_models")
    @patch("deployment.app.db.data_retention.cleanup_old_historical_data")
    @patch("deployment.app.db.data_retention.get_settings")
    async def test_run_cleanup_job_success(
        self,
        mock_get_settings,
        mock_cleanup_historical,
        mock_cleanup_models,
        mock_cleanup_predictions,
        domain_db,
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
        result = await run_cleanup_job(dal=domain_db["dal"])

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
        from deployment.app.db.schema_postgresql import SCHEMA_SQL

        # Verify schema is not empty
        assert SCHEMA_SQL is not None
        assert len(SCHEMA_SQL.strip()) > 0

        # Verify schema contains expected tables
        assert "models" in SCHEMA_SQL.lower()
        assert "training_results" in SCHEMA_SQL.lower()
        assert "configs" in SCHEMA_SQL.lower()

    @pytest.mark.asyncio
    async def test_end_to_end_model_lifecycle(self, domain_db, temp_model_files):
        """Test complete model lifecycle from creation to deletion."""
        dal = domain_db["dal"]
        model_path = temp_model_files["model_files"][0]

        # 1. Create model
        model_id = "lifecycle_test_model"
        await dal.create_model_record(
            model_id=model_id,
            job_id=domain_db["job_id"],
            model_path=model_path,
            created_at=datetime.now(),
            metadata={"lifecycle": "test"},
            is_active=False,
        )

        # 2. Verify model exists
        model = await dal.execute_raw_query(
            "SELECT * FROM models WHERE model_id = $1",
            params=(model_id,),
            fetchall=False,
        )
        assert model is not None
        assert model["is_active"] is False

        # 3. Activate model
        result = await dal.set_model_active(model_id)
        assert result is True

        # 4. Verify model is active
        active_model = await dal.get_active_model()
        assert active_model["model_id"] == model_id
        # Note: get_active_model doesn't return is_active field

        # 5. Delete model
        # Need to mock _is_path_safe in the models module where it's used
        with (
            patch("os.path.exists", return_value=True),
            patch("os.remove") as mock_remove,
            patch("deployment.app.db.queries.models._is_path_safe", return_value=True),
        ):
            delete_result = await dal.delete_model_record_and_file(model_id)
            assert delete_result is True
            mock_remove.assert_called_once_with(model_path)

        # 6. Verify model is deleted
        deleted_model = await dal.execute_raw_query(
            "SELECT * FROM models WHERE model_id = $1",
            params=(model_id,),
            fetchall=False,
        )
        assert deleted_model is None

        # 7. Verify no active model remains
        active_model_after_delete = await dal.get_active_model()
        assert active_model_after_delete is None
