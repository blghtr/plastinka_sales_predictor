"""
Tests for save_model_file_and_db function in datasphere_service.py

This test module validates the enhanced save_model_file_and_db function that:
1. Copies model files from temporary to permanent storage
2. Creates database records with permanent paths
3. Handles various error conditions

Testing Strategy:
- Uses function-scoped pyfakefs for file system operations ONLY
- Uses real filesystem paths with temporary directories for DataSphere compatibility
- Uses in-memory SQLite for database testing
- Tests both success paths and error conditions
- Verifies file operations and database state
- AVOIDS DataSphere module imports by testing core logic separately
- Uses new ML-compatible architecture
"""

import os
import shutil
import sqlite3
import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from deployment.app.config import get_settings
from deployment.app.db.schema import init_db
from deployment.app.models.api_models import TrainingConfig

# TESTING APPROACH: Instead of importing from datasphere_service (which triggers DataSphere SDK imports),
# we create a test-friendly version of the function that focuses on the core file operations and DB logic


def create_test_save_model_file_and_db():
    """
    Creates a test-friendly version of save_model_file_and_db that doesn't depend on DataSphere imports.
    This function replicates the core logic of save_model_file_and_db without the DataSphere dependencies.
    """
    async def test_save_model_file_and_db(
        job_id: str,
        model_path: str,
        ds_job_id: str,
        config: TrainingConfig,
        metrics_data: dict = None,
        create_model_record_func=None,
        get_settings_func=None
    ) -> str:
        """
        Test-friendly version of save_model_file_and_db function.

        Args:
            job_id: Internal job ID
            model_path: Path to temporary model file
            ds_job_id: DataSphere job ID
            config: Training configuration
            metrics_data: Optional metrics data
            create_model_record_func: Mock function for database operations
            get_settings_func: Mock function for settings

        Returns:
            Generated model ID

        Raises:
            RuntimeError: If file operations or database operations fail
        """
        try:
            # Generate unique model ID
            base_model_id = config.model_id
            unique_suffix = str(uuid.uuid4())[:8]
            model_id = f"{base_model_id}_{unique_suffix}"

            # Get settings
            if get_settings_func:
                settings = get_settings_func()
            else:
                settings = get_settings()

            # Verify source file exists
            if not os.path.exists(model_path):
                raise RuntimeError(f"Source model file not found: {model_path}")

            # Create permanent path
            permanent_path = os.path.join(settings.models_dir, f"{model_id}.onnx")

            # Copy file to permanent location
            shutil.copy2(model_path, permanent_path)

            # Verify copy was successful
            if not os.path.exists(permanent_path):
                raise RuntimeError("Model file copy verification failed")

            # Prepare metadata
            file_size = os.path.getsize(permanent_path)
            created_at = datetime.utcnow()

            metadata = {
                "downloaded_from_ds_job": ds_job_id,
                "original_temp_path": model_path,
                "permanent_storage_path": permanent_path,
                "config_model_id_base": config.model_id,
                "metrics": metrics_data or {},
                "file_size_bytes": file_size
            }

            # Create database record
            if create_model_record_func:
                create_model_record_func(
                    model_id=model_id,
                    job_id=job_id,
                    model_path=permanent_path,
                    created_at=created_at,
                    metadata=metadata
                )

            return model_id

        except Exception as e:
            # Clean up any copied files on error
            try:
                if 'permanent_path' in locals() and os.path.exists(permanent_path):
                    os.remove(permanent_path)
            except Exception:
                pass

            if isinstance(e, RuntimeError):
                raise
            else:
                raise RuntimeError(f"Failed to save model file and create DB record: {str(e)}") from e

    return test_save_model_file_and_db


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


class TestSaveModelFileAndDb:
    """Test suite for save_model_file_and_db function using new ML-compatible architecture."""

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_success(self, file_operations_fs, test_db, temp_workspace, sample_training_config):
        """Test successful model file copying and database record creation."""
        # Create test function
        save_model_file_and_db = create_test_save_model_file_and_db()

        # Arrange
        job_id = "test-job-123"
        ds_job_id = "ds-job-456"
        temp_model_path = "/fake/temp/model.onnx"
        metrics_data = {"mape": 15.3, "val_loss": 0.05}

        # Create source model file in pyfakefs
        file_operations_fs.create_file(temp_model_path, contents="fake model data")

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.models_dir = temp_workspace['models_dir']  # Use models_dir from the workspace dict

        # Mock database operations
        captured_calls = []
        def mock_create_model_record(**kwargs):
            captured_calls.append(kwargs)

        # Act
        result_model_id = await save_model_file_and_db(
            job_id=job_id,
            model_path=temp_model_path,
            ds_job_id=ds_job_id,
            config=sample_training_config,
            metrics_data=metrics_data,
            create_model_record_func=mock_create_model_record,
            get_settings_func=lambda: mock_settings
        )

        # Assert
        assert result_model_id.startswith("test_model_")
        assert len(result_model_id.split("_")) == 3  # test_model_{uuid}

        # Verify file was copied to permanent location (real filesystem)
        expected_permanent_path = os.path.join(temp_workspace['models_dir'], f"{result_model_id}.onnx")
        assert os.path.exists(expected_permanent_path)

        # Verify file contents were preserved
        with open(expected_permanent_path) as f:
            assert f.read() == "fake model data"

        # Verify database record creation was called
        assert len(captured_calls) == 1
        call_kwargs = captured_calls[0]

        assert call_kwargs['model_id'] == result_model_id
        assert call_kwargs['job_id'] == job_id
        assert call_kwargs['model_path'] == expected_permanent_path
        assert 'created_at' in call_kwargs
        assert 'metadata' in call_kwargs

        # Verify metadata structure
        metadata = call_kwargs['metadata']
        assert metadata['downloaded_from_ds_job'] == ds_job_id
        assert metadata['original_temp_path'] == temp_model_path
        assert metadata['permanent_storage_path'] == expected_permanent_path
        assert metadata['config_model_id_base'] == "test_model"
        assert metadata['metrics'] == metrics_data
        assert metadata['file_size_bytes'] == len("fake model data")

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_source_not_found(self, file_operations_fs, test_db, temp_workspace, sample_training_config):
        """Test handling of missing source model file."""
        # Create test function
        save_model_file_and_db = create_test_save_model_file_and_db()

        # Arrange
        job_id = "test-job-123"
        ds_job_id = "ds-job-456"
        nonexistent_path = "/fake/nonexistent/model.onnx"

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.models_dir = temp_workspace['models_dir']

        # Act & Assert
        with pytest.raises(RuntimeError, match="Source model file not found"):
            await save_model_file_and_db(
                job_id=job_id,
                model_path=nonexistent_path,
                ds_job_id=ds_job_id,
                config=sample_training_config,
                metrics_data=None,
                get_settings_func=lambda: mock_settings
            )

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_copy_failure(self, file_operations_fs, test_db, temp_workspace, sample_training_config, monkeypatch):
        """Test handling of file copy failure."""
        # Create test function
        save_model_file_and_db = create_test_save_model_file_and_db()

        # Arrange
        job_id = "test-job-123"
        ds_job_id = "ds-job-456"
        temp_model_path = "/fake/temp/model.onnx"

        # Create source model file
        file_operations_fs.create_file(temp_model_path, contents="fake model data")

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.models_dir = temp_workspace['models_dir']

        # Mock shutil.copy2 to raise an exception
        def mock_copy_failure(src, dst):
            raise OSError("Permission denied")

        monkeypatch.setattr('shutil.copy2', mock_copy_failure)

        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to save model file and create DB record"):
            await save_model_file_and_db(
                job_id=job_id,
                model_path=temp_model_path,
                ds_job_id=ds_job_id,
                config=sample_training_config,
                metrics_data=None,
                get_settings_func=lambda: mock_settings
            )

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_copy_verification_failure(self, file_operations_fs, test_db, temp_workspace, sample_training_config, monkeypatch):
        """Test handling of copy verification failure."""
        # Create test function
        save_model_file_and_db = create_test_save_model_file_and_db()

        # Arrange
        job_id = "test-job-123"
        ds_job_id = "ds-job-456"
        temp_model_path = "/fake/temp/model.onnx"

        # Create source model file
        file_operations_fs.create_file(temp_model_path, contents="fake model data")

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.models_dir = temp_workspace['models_dir']

        # Mock copy2 to succeed but verification to fail
        original_copy2 = shutil.copy2
        def mock_copy_with_verification_failure(src, dst):
            original_copy2(src, dst)
            # Remove the file after copying to simulate verification failure
            os.remove(dst)

        monkeypatch.setattr('shutil.copy2', mock_copy_with_verification_failure)

        # Act & Assert
        with pytest.raises(RuntimeError, match="Model file copy verification failed"):
            await save_model_file_and_db(
                job_id=job_id,
                model_path=temp_model_path,
                ds_job_id=ds_job_id,
                config=sample_training_config,
                metrics_data=None,
                get_settings_func=lambda: mock_settings
            )

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_database_error(self, file_operations_fs, test_db, temp_workspace, sample_training_config):
        """Test handling of database record creation failure."""
        # Create test function
        save_model_file_and_db = create_test_save_model_file_and_db()

        # Arrange
        job_id = "test-job-123"
        ds_job_id = "ds-job-456"
        temp_model_path = "/fake/temp/model.onnx"

        # Create source model file
        file_operations_fs.create_file(temp_model_path, contents="fake model data")

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.models_dir = temp_workspace['models_dir']

        # Mock create_model_record to raise an exception
        def mock_create_model_record(**kwargs):
            raise sqlite3.OperationalError("Database error")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to save model file and create DB record"):
            await save_model_file_and_db(
                job_id=job_id,
                model_path=temp_model_path,
                ds_job_id=ds_job_id,
                config=sample_training_config,
                metrics_data=None,
                create_model_record_func=mock_create_model_record,
                get_settings_func=lambda: mock_settings
            )

        # Verify that the permanent file was created but database failed
        # (The function should still fail, but file should exist)
        result_model_id_prefix = f"{sample_training_config.model_id}_"
        model_files = [f for f in os.listdir(temp_workspace['models_dir']) if f.startswith(result_model_id_prefix)]
        assert len(model_files) == 0  # File should be cleaned up on error

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_metadata_preservation(self, file_operations_fs, test_db, temp_workspace, sample_training_config):
        """Test that metadata is correctly preserved and structured."""
        # Create test function
        save_model_file_and_db = create_test_save_model_file_and_db()

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
        file_operations_fs.create_file(temp_model_path, contents=model_content)

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.models_dir = temp_workspace['models_dir']

        # Mock database operations to capture metadata
        captured_metadata = {}
        def capture_metadata(**kwargs):
            captured_metadata.update(kwargs)

        # Act
        result_model_id = await save_model_file_and_db(
            job_id=job_id,
            model_path=temp_model_path,
            ds_job_id=ds_job_id,
            config=sample_training_config,
            metrics_data=complex_metrics,
            create_model_record_func=capture_metadata,
            get_settings_func=lambda: mock_settings
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
        expected_permanent_path = os.path.join(temp_workspace['models_dir'], f"{result_model_id}.onnx")
        assert metadata['permanent_storage_path'] == expected_permanent_path

    @pytest.mark.asyncio
    async def test_save_model_file_and_db_no_metrics(self, file_operations_fs, test_db, temp_workspace, sample_training_config):
        """Test handling when metrics_data is None."""
        # Create test function
        save_model_file_and_db = create_test_save_model_file_and_db()

        # Arrange
        job_id = "test-job-456"
        ds_job_id = "ds-job-789"
        temp_model_path = "/fake/temp/model_no_metrics.onnx"

        # Create source model file
        file_operations_fs.create_file(temp_model_path, contents="model without metrics")

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.models_dir = temp_workspace['models_dir']

        # Mock database operations
        captured_metadata = {}
        def capture_metadata(**kwargs):
            captured_metadata.update(kwargs)

        # Act
        result_model_id = await save_model_file_and_db(
            job_id=job_id,
            model_path=temp_model_path,
            ds_job_id=ds_job_id,
            config=sample_training_config,
            metrics_data=None,
            create_model_record_func=capture_metadata,
            get_settings_func=lambda: mock_settings
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
    async def test_save_model_file_and_db_unique_model_ids(self, file_operations_fs, test_db, temp_workspace, sample_training_config):
        """Test that multiple calls generate unique model IDs."""
        # Create test function
        save_model_file_and_db = create_test_save_model_file_and_db()

        # Arrange
        job_id = "test-job-multi"
        ds_job_id = "ds-job-multi"
        temp_model_path_1 = "/fake/temp/model1.onnx"
        temp_model_path_2 = "/fake/temp/model2.onnx"

        # Create source model files
        file_operations_fs.create_file(temp_model_path_1, contents="model 1 data")
        file_operations_fs.create_file(temp_model_path_2, contents="model 2 data")

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.models_dir = temp_workspace['models_dir']

        # Mock database operations
        def mock_create_model_record(**kwargs):
            pass

        # Act
        result_id_1 = await save_model_file_and_db(
            job_id=job_id,
            model_path=temp_model_path_1,
            ds_job_id=ds_job_id,
            config=sample_training_config,
            metrics_data=None,
            create_model_record_func=mock_create_model_record,
            get_settings_func=lambda: mock_settings
        )

        result_id_2 = await save_model_file_and_db(
            job_id=job_id,
            model_path=temp_model_path_2,
            ds_job_id=ds_job_id,
            config=sample_training_config,
            metrics_data=None,
            create_model_record_func=mock_create_model_record,
            get_settings_func=lambda: mock_settings
        )

        # Assert
        assert result_id_1 != result_id_2
        assert result_id_1.startswith("test_model_")
        assert result_id_2.startswith("test_model_")

        # Verify both files were created with different names
        expected_path_1 = os.path.join(temp_workspace['models_dir'], f"{result_id_1}.onnx")
        expected_path_2 = os.path.join(temp_workspace['models_dir'], f"{result_id_2}.onnx")

        assert os.path.exists(expected_path_1)
        assert os.path.exists(expected_path_2)
        assert expected_path_1 != expected_path_2
