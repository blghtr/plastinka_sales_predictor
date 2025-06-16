"""
Tests for save_model_file_and_db function in datasphere_service.py

This test module validates the enhanced save_model_file_and_db function that:
1. Copies model files from temporary to permanent storage
2. Creates database records with permanent paths
3. Handles various error conditions

Testing Strategy:
- Uses pyfakefs for file system mocking
- Uses in-memory SQLite for database testing
- Tests both success paths and error conditions
- Verifies file operations and database state
"""

import pytest
import os
import uuid
import sqlite3
import json
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import test utilities
from deployment.app.services.datasphere_service import save_model_file_and_db
from deployment.app.models.api_models import TrainingConfig
from deployment.app.db.schema import init_db
from deployment.app.db.database import create_model_record


@pytest.fixture
def sample_training_config():
    """Creates a sample TrainingConfig for testing."""
    return TrainingConfig(
        model_id="test_model",
        nn_model_config={
            "num_encoder_layers": 3,
            "num_decoder_layers": 2,
            "decoder_output_dim": 128,
            "temporal_width_past": 12,
            "temporal_width_future": 6,
            "temporal_hidden_size_past": 64,
            "temporal_hidden_size_future": 64,
            "temporal_decoder_hidden": 128,
            "batch_size": 32,
            "dropout": 0.2,
            "use_reversible_instance_norm": True,
            "use_layer_norm": True
        },
        optimizer_config={
            "lr": 0.001,
            "weight_decay": 0.0001
        },
        lr_shed_config={
            "T_0": 10,
            "T_mult": 2
        },
        train_ds_config={
            "alpha": 0.05,
            "span": 12
        },
        lags=12,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )


@pytest.fixture
def test_db():
    """Creates an in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    
    # Initialize database schema
    init_db(connection=conn)
    
    yield conn
    conn.close()


@pytest.fixture
def mock_settings(fs, monkeypatch):
    """Mocks application settings with fake file system paths."""
    # Create fake directories in pyfakefs
    models_dir = "/fake/models"
    fs.makedirs(models_dir, exist_ok=True)
    
    # Mock settings object
    mock_settings_obj = MagicMock()
    mock_settings_obj.models_dir = models_dir
    
    # Patch settings in the datasphere_service module
    monkeypatch.setattr('deployment.app.services.datasphere_service.settings', mock_settings_obj)
    
    return mock_settings_obj


class TestSaveModelFileAndDb:
    """Test suite for save_model_file_and_db function."""
    
    @pytest.mark.asyncio
    async def test_save_model_file_and_db_success(self, fs, test_db, mock_settings, sample_training_config, monkeypatch):
        """Test successful model file copying and database record creation."""
        # Arrange
        job_id = "test-job-123"
        ds_job_id = "ds-job-456"
        temp_model_path = "/fake/temp/model.onnx"
        metrics_data = {"mape": 15.3, "val_loss": 0.05}
        
        # Create source model file
        fs.create_file(temp_model_path, contents="fake model data")
        
        # Mock database operations
        mock_create_model_record = MagicMock()
        monkeypatch.setattr('deployment.app.services.datasphere_service.create_model_record', mock_create_model_record)
        
        # Act
        result_model_id = await save_model_file_and_db(
            job_id=job_id,
            model_path=temp_model_path,
            ds_job_id=ds_job_id,
            config=sample_training_config,
            metrics_data=metrics_data
        )
        
        # Assert
        assert result_model_id.startswith("test_model_")
        assert len(result_model_id.split("_")) == 3  # test_model_{uuid}
        
        # Verify file was copied to permanent location
        expected_permanent_path = os.path.join(mock_settings.models_dir, f"{result_model_id}.onnx")
        assert fs.exists(expected_permanent_path)
        
        # Verify file contents were preserved
        with open(expected_permanent_path, 'r') as f:
            assert f.read() == "fake model data"
        
        # Verify database record creation was called
        mock_create_model_record.assert_called_once()
        call_args = mock_create_model_record.call_args
        
        assert call_args.kwargs['model_id'] == result_model_id
        assert call_args.kwargs['job_id'] == job_id
        assert call_args.kwargs['model_path'] == expected_permanent_path
        assert 'created_at' in call_args.kwargs
        assert 'metadata' in call_args.kwargs
        
        # Verify metadata structure
        metadata = call_args.kwargs['metadata']
        assert metadata['downloaded_from_ds_job'] == ds_job_id
        assert metadata['original_temp_path'] == temp_model_path
        assert metadata['permanent_storage_path'] == expected_permanent_path
        assert metadata['config_model_id_base'] == "test_model"
        assert metadata['metrics'] == metrics_data
        assert metadata['file_size_bytes'] == len("fake model data")

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_source_not_found(self, fs, test_db, mock_settings, sample_training_config):
        """Test handling of missing source model file."""
        # Arrange
        job_id = "test-job-123"
        ds_job_id = "ds-job-456"
        nonexistent_path = "/fake/nonexistent/model.onnx"
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to save model file and create DB record"):
            await save_model_file_and_db(
                job_id=job_id,
                model_path=nonexistent_path,
                ds_job_id=ds_job_id,
                config=sample_training_config,
                metrics_data=None
            )

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_copy_failure(self, fs, test_db, mock_settings, sample_training_config, monkeypatch):
        """Test handling of file copy failure."""
        # Arrange
        job_id = "test-job-123"
        ds_job_id = "ds-job-456"
        temp_model_path = "/fake/temp/model.onnx"
        
        # Create source model file
        fs.create_file(temp_model_path, contents="fake model data")
        
        # Mock shutil.copy2 to raise an exception
        def mock_copy_failure(src, dst):
            raise OSError("Permission denied")
        
        monkeypatch.setattr('deployment.app.services.datasphere_service.shutil.copy2', mock_copy_failure)
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to save model file and create DB record"):
            await save_model_file_and_db(
                job_id=job_id,
                model_path=temp_model_path,
                ds_job_id=ds_job_id,
                config=sample_training_config,
                metrics_data=None
            )

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_copy_verification_failure(self, fs, test_db, mock_settings, sample_training_config, monkeypatch):
        """Test handling of copy verification failure."""
        # Arrange
        job_id = "test-job-123"
        ds_job_id = "ds-job-456"
        temp_model_path = "/fake/temp/model.onnx"
        
        # Create source model file
        fs.create_file(temp_model_path, contents="fake model data")
        
        # Mock copy2 to succeed but verification to fail
        original_copy2 = shutil.copy2
        def mock_copy_with_verification_failure(src, dst):
            original_copy2(src, dst)
            # Remove the file after copying to simulate verification failure
            os.remove(dst)
        
        monkeypatch.setattr('deployment.app.services.datasphere_service.shutil.copy2', mock_copy_with_verification_failure)
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Model file copy verification failed"):
            await save_model_file_and_db(
                job_id=job_id,
                model_path=temp_model_path,
                ds_job_id=ds_job_id,
                config=sample_training_config,
                metrics_data=None
            )

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_database_error(self, fs, test_db, mock_settings, sample_training_config, monkeypatch):
        """Test handling of database record creation failure."""
        # Arrange
        job_id = "test-job-123"
        ds_job_id = "ds-job-456"
        temp_model_path = "/fake/temp/model.onnx"
        
        # Create source model file
        fs.create_file(temp_model_path, contents="fake model data")
        
        # Mock create_model_record to raise an exception
        mock_create_model_record = MagicMock(side_effect=sqlite3.OperationalError("Database error"))
        monkeypatch.setattr('deployment.app.services.datasphere_service.create_model_record', mock_create_model_record)
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to save model file and create DB record"):
            await save_model_file_and_db(
                job_id=job_id,
                model_path=temp_model_path,
                ds_job_id=ds_job_id,
                config=sample_training_config,
                metrics_data=None
            )
        
        # Verify that the permanent file was created but database failed
        # (The function should still fail, but file should exist)
        result_model_id_prefix = f"{sample_training_config.model_id}_"
        model_files = [f for f in os.listdir(mock_settings.models_dir) if f.startswith(result_model_id_prefix)]
        assert len(model_files) == 1  # File was copied before DB error

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_metadata_preservation(self, fs, test_db, mock_settings, sample_training_config, monkeypatch):
        """Test that metadata is correctly preserved and structured."""
        # Arrange
        job_id = "test-job-789"
        ds_job_id = "ds-job-012"
        temp_model_path = "/fake/temp/special_model.onnx"
        complex_metrics = {
            "mape": 12.5,
            "val_loss": 0.03,
            "train_loss": 0.025,
            "epochs": 50,
            "early_stopping": True
        }
        
        # Create source model file with specific content
        model_content = "complex model binary data"
        fs.create_file(temp_model_path, contents=model_content)
        
        # Mock database operations to capture metadata
        captured_metadata = {}
        def capture_metadata(**kwargs):
            captured_metadata.update(kwargs)
        
        mock_create_model_record = MagicMock(side_effect=capture_metadata)
        monkeypatch.setattr('deployment.app.services.datasphere_service.create_model_record', mock_create_model_record)
        
        # Act
        result_model_id = await save_model_file_and_db(
            job_id=job_id,
            model_path=temp_model_path,
            ds_job_id=ds_job_id,
            config=sample_training_config,
            metrics_data=complex_metrics
        )
        
        # Assert metadata structure and content
        metadata = captured_metadata['metadata']
        
        # Basic structure
        assert 'downloaded_from_ds_job' in metadata
        assert 'original_temp_path' in metadata
        assert 'permanent_storage_path' in metadata
        assert 'config_model_id_base' in metadata
        assert 'metrics' in metadata
        assert 'file_size_bytes' in metadata
        
        # Specific values
        assert metadata['downloaded_from_ds_job'] == ds_job_id
        assert metadata['original_temp_path'] == temp_model_path
        assert metadata['permanent_storage_path'].endswith(f"{result_model_id}.onnx")
        assert metadata['config_model_id_base'] == sample_training_config.model_id
        assert metadata['metrics'] == complex_metrics
        assert metadata['file_size_bytes'] == len(model_content)
        
        # Verify paths are correctly formed
        expected_permanent_path = os.path.join(mock_settings.models_dir, f"{result_model_id}.onnx")
        assert metadata['permanent_storage_path'] == expected_permanent_path

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_no_metrics(self, fs, test_db, mock_settings, sample_training_config, monkeypatch):
        """Test handling when metrics_data is None."""
        # Arrange
        job_id = "test-job-456"
        ds_job_id = "ds-job-789"
        temp_model_path = "/fake/temp/model_no_metrics.onnx"
        
        # Create source model file
        fs.create_file(temp_model_path, contents="model without metrics")
        
        # Mock database operations
        captured_metadata = {}
        def capture_metadata(**kwargs):
            captured_metadata.update(kwargs)
        
        mock_create_model_record = MagicMock(side_effect=capture_metadata)
        monkeypatch.setattr('deployment.app.services.datasphere_service.create_model_record', mock_create_model_record)
        
        # Act
        result_model_id = await save_model_file_and_db(
            job_id=job_id,
            model_path=temp_model_path,
            ds_job_id=ds_job_id,
            config=sample_training_config,
            metrics_data=None
        )
        
        # Assert
        assert result_model_id.startswith("test_model_")
        
        # Verify that metrics is an empty dict when None is passed
        metadata = captured_metadata['metadata']
        assert metadata['metrics'] == {}
        
        # Other metadata should still be present
        assert metadata['downloaded_from_ds_job'] == ds_job_id
        assert metadata['original_temp_path'] == temp_model_path
        assert metadata['config_model_id_base'] == sample_training_config.model_id

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_unique_model_ids(self, fs, test_db, mock_settings, sample_training_config, monkeypatch):
        """Test that multiple calls generate unique model IDs."""
        # Arrange
        job_id = "test-job-multi"
        ds_job_id = "ds-job-multi"
        temp_model_path_1 = "/fake/temp/model1.onnx"
        temp_model_path_2 = "/fake/temp/model2.onnx"
        
        # Create source model files
        fs.create_file(temp_model_path_1, contents="model 1 data")
        fs.create_file(temp_model_path_2, contents="model 2 data")
        
        # Mock database operations
        mock_create_model_record = MagicMock()
        monkeypatch.setattr('deployment.app.services.datasphere_service.create_model_record', mock_create_model_record)
        
        # Act
        result_id_1 = await save_model_file_and_db(
            job_id=job_id,
            model_path=temp_model_path_1,
            ds_job_id=ds_job_id,
            config=sample_training_config,
            metrics_data=None
        )
        
        result_id_2 = await save_model_file_and_db(
            job_id=job_id,
            model_path=temp_model_path_2,
            ds_job_id=ds_job_id,
            config=sample_training_config,
            metrics_data=None
        )
        
        # Assert
        assert result_id_1 != result_id_2
        assert result_id_1.startswith("test_model_")
        assert result_id_2.startswith("test_model_")
        
        # Verify both files were created with different names
        expected_path_1 = os.path.join(mock_settings.models_dir, f"{result_id_1}.onnx")
        expected_path_2 = os.path.join(mock_settings.models_dir, f"{result_id_2}.onnx")
        
        assert fs.exists(expected_path_1)
        assert fs.exists(expected_path_2)
        assert expected_path_1 != expected_path_2 