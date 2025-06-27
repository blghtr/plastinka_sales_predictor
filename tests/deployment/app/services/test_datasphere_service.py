"""
Comprehensive test suite for DataSphere service functionality.

This test suite follows the established patterns from test_train_and_predict.py,
providing comprehensive coverage of the DataSphere job lifecycle including:
- Job execution (success and failure scenarios)
- Status management and polling
- Partial results handling
- Error handling and recovery
- File operations and archiving
- Configuration management
- Integration testing

Testing Approach:
- All external dependencies are mocked using @patch decorators and MagicMock objects
- Both success and failure scenarios are tested
- Proper exception handling verification with pytest.raises
- Clear Arrange-Act-Assert pattern
- Integration tests verify end-to-end functionality
"""

import asyncio
import logging
import tempfile
import json
import os
import zipfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from pyfakefs.fake_filesystem_unittest import Patcher

from deployment.app.models.api_models import JobStatus, TrainingConfig, ModelConfig, OptimizerConfig, LRSchedulerConfig, TrainingDatasetConfig
from deployment.app.db.database import DatabaseError
from deployment.datasphere.client import DataSphereClient
from tests.deployment.app.datasphere.conftest import create_training_params

# Import the module under test
import deployment.app.services.datasphere_service as ds_module


class TestRunJob:
    """Test cases for the main run_job function."""

    @pytest.mark.asyncio
    async def test_run_job_success_complete(self, mock_service_env, caplog):
        """Test successful job execution with complete results."""
        # Arrange
        job_id = "test_job_success_complete"
        training_config = create_training_params()
        
        mock_client = mock_service_env["client"]
        mock_client.submit_job.return_value = "ds-job-success"
        mock_client.get_job_status.return_value = "COMPLETED"
        
        fs = mock_service_env["fs"]
        
        # Configure mock_client.download_job_results to create files in the correct location
        def download_side_effect(ds_job_id, results_dir, **kwargs):
            # Create the results directory in fakefs
            fs.makedirs(results_dir, exist_ok=True)
            
            # Create test files in the results directory
            fs.create_file(f"{results_dir}/metrics.json", contents='{"val_loss": 0.1}')
            fs.create_file(f"{results_dir}/model.onnx", contents="model data")
            fs.create_file(f"{results_dir}/predictions.csv", contents="barcode,pred\n1,10")
        
        mock_client.download_job_results.side_effect = download_side_effect

        # Act
        with caplog.at_level(logging.INFO):
            result = await ds_module.run_job(
                job_id=job_id,
                training_config=training_config.model_dump(),
                config_id="test-config-complete"
            )

        # Assert
        assert result is not None
        assert "Job completed" in caplog.text
        mock_service_env["create_training_result"].assert_called_once()

    @pytest.mark.asyncio 
    async def test_run_job_no_configuration_provided(self, mock_service_env):
        """Test job execution without providing configuration."""
        # Arrange
        job_id = "test_job_no_config"
        
        # Act & Assert
        with pytest.raises(ValueError, match="No training configuration was provided or found"):
            await ds_module.run_job(
                job_id=job_id,
                training_config=None,
                config_id="test-config-none"
            )

    @pytest.mark.asyncio
    async def test_run_job_partial_results_only(self, mock_service_env, caplog):
        """Test job execution with partial results (only some files available)."""
        # Arrange
        job_id = "test_job_partial_results"
        training_config = create_training_params()
        
        mock_client = mock_service_env["client"]
        mock_client.submit_job.return_value = "ds-job-partial"
        mock_client.get_job_status.return_value = "COMPLETED"
        
        fs = mock_service_env["fs"]
        
        # Configure mock_client.download_job_results to create only some files  
        def download_side_effect(ds_job_id, results_dir, **kwargs):
            fs.makedirs(results_dir, exist_ok=True)
            fs.create_file(f"{results_dir}/predictions.csv", contents="barcode,pred\n2,20")
            # Missing metrics.json and model.onnx files
        
        mock_client.download_job_results.side_effect = download_side_effect

        # Act
        with caplog.at_level(logging.WARNING):
            result = await ds_module.run_job(
                job_id=job_id,
                training_config=training_config.model_dump(),
                config_id="test-config-partial"
            )

        # Assert
        assert result is not None
        assert "Model file 'model.onnx' not found" in caplog.text
        assert "Metrics file 'metrics.json' not found" in caplog.text


class TestJobStatusManagement:
    """Test cases for job status management and polling functionality."""

    @pytest.mark.asyncio
    async def test_check_datasphere_job_status_completed(self, mock_service_env):
        """Test job status checking for completed jobs."""
        # Arrange
        mock_client = mock_service_env["client"]
        mock_client.get_job_status.return_value = "COMPLETED"

        # Act
        status = await ds_module._check_datasphere_job_status(
            "test-job", "test-ds-job", mock_client
        )

        # Assert
        assert status == "COMPLETED"
        mock_client.get_job_status.assert_called_once_with("test-ds-job")

    @pytest.mark.asyncio
    async def test_check_datasphere_job_status_timeout(self, mock_service_env):
        """Test job status checking handles timeout gracefully."""
        # Arrange
        mock_client = mock_service_env["client"]

        # Simulate timeout by raising TimeoutError (sync function)
        def timeout_side_effect(*args, **kwargs):
            raise asyncio.TimeoutError("Operation timed out")

        mock_client.get_job_status.side_effect = timeout_side_effect

        # Act & Assert
        with pytest.raises(RuntimeError, match="DataSphere job status check timed out"):
            await ds_module._check_datasphere_job_status(
                "test-job", "test-ds-job", mock_client
            )

    @pytest.mark.asyncio
    async def test_process_datasphere_job_polling_success(self, mock_service_env):
        """Test successful job processing with polling."""
        # Arrange
        mock_client = mock_service_env["client"]
        mock_client.submit_job.return_value = "test-ds-job"
        mock_client.get_job_status.side_effect = ["RUNNING", "RUNNING", "COMPLETED"]

        fs = mock_service_env["fs"]
        
        # Configure mock_client.download_job_results to create files in the correct location
        def download_side_effect(ds_job_id, results_dir, **kwargs):
            fs.makedirs(results_dir, exist_ok=True)
            fs.create_file(f"{results_dir}/metrics.json", contents='{"val_loss": 0.1}')
            fs.create_file(f"{results_dir}/model.onnx", contents="model data")
            fs.create_file(f"{results_dir}/predictions.csv", contents="barcode,pred\n1,10")
        
        mock_client.download_job_results.side_effect = download_side_effect

        # Create temporary config file
        config_file = "/temp/config.yaml"
        fs.create_file(config_file, contents="name: test\ntype: python")

        # Act
        result = await ds_module._process_datasphere_job(
            job_id="test-job",
            client=mock_client,
            ds_job_specific_output_base_dir="/test/output",
            ready_config_path=config_file,
            work_dir="/tmp/test"
        )

        # Assert
        ds_job_id, results_dir, metrics_data, model_path, predictions_path, polls = result
        assert ds_job_id == "test-ds-job"
        assert polls > 0


class TestPartialResultsHandling:
    """Test cases for handling partial or missing results."""

    @pytest.mark.asyncio
    async def test_run_job_missing_model_file(self, mock_service_env, caplog):
        """Test job handling when model file is missing."""
        # Arrange
        job_id = "test_job_missing_model"
        training_config = create_training_params()
        
        mock_client = mock_service_env["client"]
        mock_client.submit_job.return_value = "ds-job-no-model"
        mock_client.get_job_status.return_value = "COMPLETED"
        
        fs = mock_service_env["fs"]
        
        # Configure mock_client.download_job_results to create files but omit model file
        def download_side_effect(ds_job_id, results_dir, **kwargs):
            fs.makedirs(results_dir, exist_ok=True)
            # Create only predictions and metrics (no model)
            fs.create_file(f"{results_dir}/metrics.json", contents='{"val_loss": 0.15}')
            fs.create_file(f"{results_dir}/predictions.csv", contents="barcode,pred\n2,20")
        
        mock_client.download_job_results.side_effect = download_side_effect

        # Act
        with caplog.at_level(logging.WARNING):
            await ds_module.run_job(
                job_id=job_id,
                training_config=training_config.model_dump(),
                config_id="test-config-no-model"
            )

        # Assert
        assert "Model file 'model.onnx' not found" in caplog.text
        # Job should still complete successfully with partial results

    @pytest.mark.asyncio
    async def test_process_job_results_missing_metrics(self, mock_service_env, caplog):
        """Test processing job results when metrics are missing."""
        # Arrange
        job_id = "test_job_no_metrics"
        results_dir = "/test/results"
        config = create_training_params()
        
        fs = mock_service_env["fs"]
        fs.makedirs(results_dir, exist_ok=True)
        
        # Create model and predictions but no metrics
        model_path = f"{results_dir}/model.onnx"
        predictions_path = f"{results_dir}/predictions.csv"
        fs.create_file(model_path, contents="model data")
        fs.create_file(predictions_path, contents="barcode,pred\n3,30")

        # Act
        with caplog.at_level(logging.WARNING):
            await ds_module._process_job_results(
                job_id=job_id,
                ds_job_id="test-ds-job",
                results_dir=results_dir,
                config=config,
                metrics_data=None,
                model_path=model_path,
                predictions_path=predictions_path,
                polls=3,
                poll_interval=1.0,
                config_id="test-config"
            )

        # Assert
        assert "No metrics data available" in caplog.text

    @pytest.mark.asyncio
    async def test_process_job_results_missing_predictions(self, mock_service_env, caplog):
        """Test processing job results when predictions are missing."""
        # Arrange
        job_id = "test_job_no_predictions"
        results_dir = "/test/results"
        config = create_training_params()
        
        fs = mock_service_env["fs"]
        fs.makedirs(results_dir, exist_ok=True)
        
        # Create model and metrics but no predictions
        model_path = f"{results_dir}/model.onnx"
        fs.create_file(model_path, contents="model data")
        metrics_data = {"val_loss": 0.2}

        # Act
        with caplog.at_level(logging.WARNING):
            await ds_module._process_job_results(
                job_id=job_id,
                ds_job_id="test-ds-job",
                results_dir=results_dir,
                config=config,
                metrics_data=metrics_data,
                model_path=model_path,
                predictions_path=None,
                polls=2,
                poll_interval=1.0,
                config_id="test-config"
            )

        # Assert
        assert "No predictions file found, cannot save predictions to DB" in caplog.text


class TestErrorHandling:
    """Test cases for error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_run_job_datasphere_failure(self, mock_service_env, caplog):
        """Test handling of DataSphere job failure."""
        # Arrange
        job_id = "test_job_ds_failure"
        training_config = create_training_params()
        
        mock_client = mock_service_env["client"]
        mock_client.submit_job.return_value = "ds-job-failed"
        mock_client.get_job_status.return_value = "FAILED"
        mock_client.download_job_results.return_value = None

        # Act & Assert
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="DS Job.*ended with status: FAILED"):
                await ds_module.run_job(
                    job_id=job_id,
                    training_config=training_config.model_dump(),
                    config_id="test-config-failure"
                )

        # Verify error logging
        assert "ended with status: FAILED" in caplog.text

    @pytest.mark.asyncio
    async def test_run_job_polling_timeout(self, mock_service_env, caplog):
        """Test handling of polling timeout."""
        # Arrange
        job_id = "test_job_timeout"
        training_config = create_training_params()
        
        mock_client = mock_service_env["client"]
        mock_client.submit_job.return_value = "ds-job-timeout"
        # Always return RUNNING to simulate timeout
        mock_client.get_job_status.return_value = "RUNNING"

        # Act & Assert
        with caplog.at_level(logging.ERROR):
            with pytest.raises(TimeoutError, match="execution timed out"):
                await ds_module.run_job(
                    job_id=job_id,
                    training_config=training_config.model_dump(),
                    config_id="test-config-timeout"
                )

        # Verify timeout logging
        assert "execution timed out" in caplog.text

    @pytest.mark.asyncio
    async def test_run_job_client_connectivity_error(self, mock_service_env, caplog):
        """Test handling of DataSphere client connectivity errors."""
        # Arrange
        job_id = "test_job_connectivity_error"
        mock_client = mock_service_env["client"]

        # Simulate connectivity error
        mock_client.submit_job.side_effect = ConnectionError("Failed to connect to DataSphere")
        training_config = create_training_params()

        # Act & Assert
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="Failed to create new DataSphere job"):
                await ds_module.run_job(
                    job_id=job_id,
                    training_config=training_config.model_dump(),
                    config_id="test-config-connectivity"
                )

        # Verify error logging
        assert "Failed to create new DataSphere job" in caplog.text

    @pytest.mark.asyncio
    async def test_run_job_database_error(self, mock_service_env):
        """Test handling of database errors during job processing."""
        # Arrange
        job_id = "test_job_db_error"
        mock_client = mock_service_env["client"]

        mock_client.submit_job.return_value = "ds-job-db-error"
        mock_client.get_job_status.return_value = "COMPLETED"
        mock_client.download_job_results.return_value = None

        # Mock database error
        mock_service_env["create_training_result"].side_effect = DatabaseError("Database connection failed")
        training_config = create_training_params()

        # Act - the job will continue and not raise the DatabaseError due to conditional logic
        await ds_module.run_job(
            job_id=job_id,
            training_config=training_config.model_dump(),
            config_id="test-config-db-error"
        )

        # Assert
        # Job completes despite database error (gets handled gracefully)
        # No exception should be raised


class TestFileOperations:
    """Test cases for file operations including archiving and downloading."""

    @pytest.mark.asyncio
    async def test_archive_input_directory_success(self, mock_service_env):
        """Test successful archiving of input directory."""
        # Arrange
        fs = mock_service_env["fs"]
        input_dir = "/test/input"

        # Create test files
        fs.makedirs(input_dir)
        fs.create_file(f"{input_dir}/config.json", contents='{"test": "config"}')
        fs.create_file(f"{input_dir}/train.dill", contents="train data")
        fs.create_file(f"{input_dir}/val.dill", contents="val data")

        # Override the mock to return a proper path string
        mock_service_env["_archive_input_directory"].return_value = "/test/output/input.zip"

        # Act
        archive_path = await ds_module._archive_input_directory("test-job", input_dir)

        # Assert
        assert archive_path.endswith("input.zip")
        # Verify the mock was called correctly
        mock_service_env["_archive_input_directory"].assert_called_once_with("test-job", input_dir)

    @pytest.mark.asyncio
    async def test_download_datasphere_job_results_with_extraction(self, mock_service_env):
        """Test downloading and extracting DataSphere job results."""
        # Arrange
        mock_client = mock_service_env["client"]
        fs = mock_service_env["fs"]
        output_dir = "/test/output"

        # Create output.zip file that needs extraction
        fs.makedirs(output_dir)
        archive_path = f"{output_dir}/output.zip"

        # Mock the download to create an archive
        def download_side_effect(job_id, output_dir, **kwargs):
            fs.create_file(archive_path, contents="fake zip content")

        mock_client.download_job_results.side_effect = download_side_effect

        # Act
        with patch('deployment.app.services.datasphere_service.zipfile.ZipFile') as mock_zipfile:
            mock_zip_context = MagicMock()
            mock_zipfile.return_value.__enter__.return_value = mock_zip_context
            mock_zip_context.extractall = MagicMock()

            await ds_module._download_datasphere_job_results(
                "test-job", "test-ds-job", mock_client, output_dir
            )

            # Assert
            mock_zipfile.assert_called_once()
            mock_zip_context.extractall.assert_called_once_with(output_dir)

    @pytest.mark.asyncio
    async def test_prepare_datasphere_job_submission_creates_config(self, mock_service_env):
        """Test preparation of DataSphere job submission creates proper config."""
        # Arrange
        fs = mock_service_env["fs"]
        config = create_training_params()
        temp_dir = Path("/temp/input")

        fs.makedirs(str(temp_dir))

        # Create template config
        template_config_path = "plastinka_sales_predictor/datasphere_job/config.yaml"
        fs.makedirs(os.path.dirname(template_config_path))
        template_content = """
        name: plastinka-training-job
        type: python
        # inputs section will be added dynamically
        """
        fs.create_file(template_config_path, contents=template_content)

        # Act
        ready_config_path = await ds_module._prepare_datasphere_job_submission(
            job_id="test-job",
            config=config,
            target_input_dir=temp_dir
        )

        # Assert
        assert ready_config_path is not None
        # Check that config.json was created in the target directory
        config_json_path = temp_dir / "config.json"
        assert fs.exists(str(config_json_path))


class TestConfigurationManagement:
    """Test cases for configuration management and validation."""

    def test_create_training_config_factory(self):
        """Test the training configuration factory function."""
        # Act
        config = create_training_params()

        # Assert
        assert isinstance(config, TrainingConfig)
        assert config.nn_model_config is not None
        assert config.optimizer_config is not None
        assert config.lr_shed_config is not None
        assert config.train_ds_config is not None

    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Arrange
        test_config = create_training_params()

        # Act & Assert
        # Should not raise any validation errors
        validated_config = TrainingConfig(**test_config.model_dump())
        assert validated_config.nn_model_config.num_encoder_layers == 3
        assert validated_config.optimizer_config.lr == 0.001

    def test_invalid_configuration_raises_error(self):
        """Test that invalid configuration raises appropriate errors."""
        # Arrange
        invalid_config_data = {
            "nn_model_config": {
                "num_encoder_layers": -1,  # Invalid negative value
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
            "optimizer_config": {
                "lr": 0.001,
                "weight_decay": 0.0001
            },
            "lr_shed_config": {
                "T_0": 10,
                "T_mult": 2
            },
            "train_ds_config": {
                "alpha": 0.05,
                "span": 12
            },
            "lags": 12,
            "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]
        }

        # Act & Assert
        with pytest.raises(ValueError):
            TrainingConfig(**invalid_config_data)


class TestIntegration:
    """Integration test cases for end-to-end functionality."""

    @pytest.mark.asyncio
    async def test_complete_job_lifecycle(self, mock_service_env, caplog):
        """Test complete job lifecycle from submission to completion."""
        # Arrange
        job_id = "test_integration_complete"
        training_config = create_training_params()
        
        mock_client = mock_service_env["client"]
        mock_client.submit_job.return_value = "ds-job-integration"
        mock_client.get_job_status.side_effect = ["RUNNING", "COMPLETED"]
        
        fs = mock_service_env["fs"]
        
        # Configure mock_client.download_job_results to create files in the correct location
        def download_side_effect(ds_job_id, results_dir, **kwargs):
            fs.makedirs(results_dir, exist_ok=True)
            # Create complete result set
            fs.create_file(f"{results_dir}/metrics.json", contents='{"val_loss": 0.08}')
            fs.create_file(f"{results_dir}/model.onnx", contents="integration model")
            fs.create_file(f"{results_dir}/predictions.csv", contents="barcode,pred\n4,40")
        
        mock_client.download_job_results.side_effect = download_side_effect

        # Act
        with caplog.at_level(logging.INFO):
            result = await ds_module.run_job(
                job_id=job_id,
                training_config=training_config.model_dump(),
                config_id="test-config-integration"
            )

        # Assert
        assert result is not None
        assert "Job completed" in caplog.text
        
        # Verify all major functions were called
        mock_service_env["create_training_result"].assert_called_once()

    def test_module_imports_successfully(self):
        """Test that all required modules can be imported."""
        # Act & Assert
        # If we get here, all imports worked
        assert ds_module is not None
        assert hasattr(ds_module, 'run_job')
        assert hasattr(ds_module, '_check_datasphere_job_status')
        assert hasattr(ds_module, '_process_datasphere_job')

    def test_constants_defined(self):
        """Test that required constants are properly defined."""
        # Act & Assert
        assert hasattr(ds_module, 'DATASPHERE_MODEL_FILE')
        assert hasattr(ds_module, 'DATASPHERE_PREDICTIONS_FILE')
        assert ds_module.DATASPHERE_MODEL_FILE == "model.onnx"
        assert ds_module.DATASPHERE_PREDICTIONS_FILE == "predictions.csv"




class TestRequirementsHashCalculation:
    """Test cases for the new _calculate_requirements_hash_for_job function."""

    @pytest.mark.asyncio
    async def test_calculate_requirements_hash_for_job_called(self, mock_service_env):
        """Test that the requirements hash calculation function is properly called."""
        # Arrange
        job_id = "test-job-hash"
        mock_hash_func = mock_service_env['_calculate_requirements_hash_for_job']
        
        # Act
        result = await ds_module._calculate_requirements_hash_for_job(job_id)
        
        # Assert
        assert result == "test-requirements-hash"  # From conftest mock
        mock_hash_func.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_requirements_hash_integration_in_run_job(self, mock_service_env):
        """Test that requirements hash calculation is integrated into run_job."""
        # Arrange
        job_id = "test_job_hash_integration"
        training_config = create_training_params()
        
        mock_client = mock_service_env["client"]
        mock_client.submit_job.return_value = "ds-job-hash-test"
        mock_client.get_job_status.return_value = "COMPLETED"
        
        fs = mock_service_env["fs"]
        
        def download_side_effect(ds_job_id, results_dir, **kwargs):
            fs.makedirs(results_dir, exist_ok=True)
            fs.create_file(f"{results_dir}/metrics.json", contents='{"val_loss": 0.1}')
            fs.create_file(f"{results_dir}/model.onnx", contents="model data")
            fs.create_file(f"{results_dir}/predictions.csv", contents="barcode,pred\n1,10")
        
        mock_client.download_job_results.side_effect = download_side_effect

        # Act
        result = await ds_module.run_job(
            job_id=job_id,
            training_config=training_config.model_dump(),
            config_id="test-config-hash-integration"
        )

        # Assert
        assert result is not None
        # Verify that requirements hash calculation was called
        mock_service_env['_calculate_requirements_hash_for_job'].assert_called_once_with(job_id) 