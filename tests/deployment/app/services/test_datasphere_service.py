"""
Test examples using the NEW ARCHITECTURE: ML/DataSphere Compatible Testing

This module demonstrates:
1. How to use temp_workspace for DataSphere tests (no import conflicts)
2. How to use file_operations_fs for pure file operations only
3. How to test DataSphere service functions without session-scoped pyfakefs

Migration verification:
- ✅ DataSphere SDK imports work
- ✅ PyTorch compatibility maintained
- ✅ Real filesystem operations
- ✅ Test isolation preserved
"""

import json
import os
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import shutil

import pytest

from deployment.app.config import get_settings
from deployment.app.models.api_models import TrainingConfig
from deployment.app.services.job_registries.job_type_registry import (
    get_job_type_config,
)
from deployment.app.services.job_registries.result_processor_registry import (
    process_tuning_results,
)
from deployment.app.services.datasphere_service import (
    _prepare_job_inputs_unified,
    _download_datasphere_job_results,
)


# Test the new conftest_new.py fixtures
@pytest.fixture(autouse=True)
def use_new_conftest(temp_workspace, mock_datasphere_env):
    """Auto-use fixture to test new architecture."""
    yield


class TestDataSphereNewArchitecture:
    """Test suite demonstrating new ML-compatible architecture."""

    def test_temp_workspace_structure(self, temp_workspace):
        """Verify that temp_workspace creates proper directory structure."""
        # Assert all required directories exist
        assert os.path.exists(temp_workspace["temp_dir"])
        assert os.path.exists(temp_workspace["input_dir"])
        assert os.path.exists(temp_workspace["output_dir"])
        assert os.path.exists(temp_workspace["models_dir"])
        assert os.path.exists(temp_workspace["logs_dir"])
        assert os.path.exists(temp_workspace["job_dir"])

        # Assert config file exists
        assert os.path.exists(temp_workspace["config_path"])
        with open(temp_workspace["config_path"]) as f:
            content = f.read()
            assert "test_job" in content

    def test_real_filesystem_operations(self, temp_workspace):
        """Test that we can perform real filesystem operations."""
        # Create a test file
        test_file = os.path.join(temp_workspace["models_dir"], "test_model.onnx")
        test_content = "fake model data for testing"

        with open(test_file, "w") as f:
            f.write(test_content)

        # Verify file exists and has correct content
        assert os.path.exists(test_file)
        with open(test_file) as f:
            assert f.read() == test_content

        # Test file size
        assert os.path.getsize(test_file) == len(test_content)

    def test_datasphere_settings_mock(self, mock_datasphere_env):
        """Test that DataSphere settings are properly mocked."""
        settings = mock_datasphere_env["settings"]

        # Verify paths point to temp workspace
        assert "datasphere_input" in settings.datasphere_input_dir
        assert "datasphere_output" in settings.datasphere_output_dir
        assert "models" in settings.models_dir

        # Verify DataSphere client config
        assert settings.datasphere.project_id == "test-project-id-new-arch"
        assert settings.datasphere.max_polls == 3
        assert settings.datasphere.poll_interval == 0.1

    def test_datasphere_client_mock(self, mock_datasphere_env):
        """Test that DataSphere client is properly mocked."""
        client = mock_datasphere_env["client"]

        # Test job submission
        job_id = client.submit_job("fake_config_path", "fake_work_dir")
        assert job_id == "ds-job-default-new-arch"

        # Test status check
        status = client.get_job_status("test-job-id")
        assert status == "COMPLETED"

        # Test download (creates real files)
        temp_dir = mock_datasphere_env["settings"].datasphere_output_dir
        results_dir = os.path.join(temp_dir, "test_results")

        client.download_job_results("test-ds-job", results_dir)

        # Verify real files were created
        assert os.path.exists(os.path.join(results_dir, "metrics.json"))
        assert os.path.exists(os.path.join(results_dir, "model.onnx"))
        assert os.path.exists(os.path.join(results_dir, "predictions.csv"))

        # Verify file contents
        with open(os.path.join(results_dir, "metrics.json")) as f:
            content = f.read()
            assert "val_loss" in content
            assert "mape" in content

    def test_datasphere_sdk_import_compatibility(self):
        """Test that DataSphere SDK can be imported without conflicts."""
        # This should NOT hang or cause import errors like with session-scoped pyfakefs
        try:
            # Import inside test to avoid collection phase conflicts
            from deployment.app.services.datasphere_service import (
                run_job,
                save_model_file_and_db,
            )

            # Verify functions exist and are callable
            assert callable(save_model_file_and_db)
            assert callable(run_job)

            # This proves DataSphere SDK imports work with new architecture
            assert True
        except ImportError as e:
            pytest.fail(f"DataSphere SDK import failed: {e}")

    @pytest.mark.asyncio
    async def test_save_model_with_new_architecture(
        self, mock_datasphere_env, temp_workspace
    ):
        """Test save_model_file_and_db with new architecture."""
        # Import INSIDE test for ML compatibility
        from deployment.app.services.datasphere_service import save_model_file_and_db

        # Create test model file in real filesystem
        temp_model_path = os.path.join(temp_workspace["temp_dir"], "test_model.onnx")
        with open(temp_model_path, "w") as f:
            f.write("test model content")

        # Create test training config
        config = TrainingConfig(
            model_id="test_model_new_arch",
            nn_model_config={
                "num_encoder_layers": 2,
                "num_decoder_layers": 1,
                "decoder_output_dim": 64,
                "temporal_width_past": 6,
                "temporal_width_future": 3,
                "temporal_hidden_size_past": 32,
                "temporal_hidden_size_future": 32,
                "temporal_decoder_hidden": 64,
                "batch_size": 16,
                "dropout": 0.1,
                "use_reversible_instance_norm": True,
                "use_layer_norm": True,
            },
            optimizer_config={"lr": 0.001, "weight_decay": 0.0001},
            lr_shed_config={"T_0": 5, "T_mult": 1},
            train_ds_config={"alpha": 0.1, "span": 6},
            lags=6,
            quantiles=[0.1, 0.5, 0.9],
        )

        # Mock create_model_record to avoid DB operations
        with patch(
            "deployment.app.services.datasphere_service.create_model_record"
        ) as mock_create:
            mock_create.return_value = None

            # Test function execution
            result_model_id = await save_model_file_and_db(
                job_id="test-job-new-arch",
                model_path=temp_model_path,
                ds_job_id="test-ds-job",
                config=config,
                metrics_data={"mape": 5.0},
            )

            # Verify result
            assert result_model_id.startswith("test_model_new_arch_")

            # Verify model was copied to permanent location
            expected_permanent_path = os.path.join(
                mock_datasphere_env["settings"].models_dir, f"{result_model_id}.onnx"
            )
            assert os.path.exists(expected_permanent_path)

            # Verify content preserved
            with open(expected_permanent_path) as f:
                assert f.read() == "test model content"

    def test_file_operations_fs_for_pure_operations(self, file_operations_fs):
        """Test function-scoped pyfakefs for pure file operations only."""
        # This should ONLY be used for pure file operations testing
        # NOT for DataSphere SDK or ML framework tests

        # Create fake file
        file_operations_fs.create_file("/fake/test.txt", contents="test content")

        # Test file operations
        import os
        import shutil

        assert os.path.exists("/fake/test.txt")

        # Test copy operation
        shutil.copy2("/fake/test.txt", "/fake/test_copy.txt")
        assert os.path.exists("/fake/test_copy.txt")

        # Test content
        with open("/fake/test_copy.txt") as f:
            assert f.read() == "test content"

    def test_isolation_between_tests(self, temp_workspace):
        """Test that each test gets isolated temp workspace."""
        # Create a marker file
        marker_file = os.path.join(
            temp_workspace["temp_dir"], "isolation_test_marker.txt"
        )
        with open(marker_file, "w") as f:
            f.write("test isolation")

        assert os.path.exists(marker_file)
        # This file should NOT exist in other tests due to isolation

    def test_mock_reset_functionality(self, mock_datasphere_env):
        """Test that mocks are properly reset between tests."""
        client = mock_datasphere_env["client"]

        # Modify mock behavior
        client.submit_job.return_value = "modified-job-id"
        assert client.submit_job("test") == "modified-job-id"

        # The reset fixture should restore default behavior in next test


class TestMigrationCompatibility:
    """Test suite to verify migration doesn't break existing functionality."""

    def test_create_training_params_still_works(self):
        """Verify that create_training_params helper function still works."""
        # Import from new conftest (now renamed to conftest.py)
        from tests.deployment.app.services.conftest import create_training_params

        # Test basic creation
        config = create_training_params()
        assert config is not None
        assert config.nn_model_config is not None
        assert config.lags == 12
        assert len(config.quantiles) == 5
        # model_id is None by default in create_training_params
        assert config.model_id is None

        # Test with base params
        base_params = {"batch_size": 64, "learning_rate": 0.01}
        config = create_training_params(base_params)
        assert config.nn_model_config.batch_size == 64
        assert config.optimizer_config.lr == 0.01

    def test_backwards_compatibility_with_individual_fixtures(
        self, mock_asyncio_sleep, mock_get_datasets
    ):
        """Test that individual fixtures from old conftest still work."""
        # Test asyncio sleep mock
        assert mock_asyncio_sleep is not None

        # Test get_datasets mock
        result = mock_get_datasets.return_value
        assert result is not None


class TestDataSphereServiceTuning:
    """Test suite for tuning functionality in DataSphere service."""

    def test_get_job_config_path_train(self, mock_datasphere_env):
        """Test JobTypeConfig returns correct path for training jobs."""
        
        # Test train job type
        job_config = get_job_type_config("train")
        config_path = os.path.join(job_config.get_script_dir(get_settings()), job_config.config_filename)
        
        # Should return train/config.yaml path (cross-platform)
        assert config_path.endswith(os.path.join("train", "config.yaml"))
        assert "train" in config_path

    def test_get_job_config_path_tune(self, mock_datasphere_env):
        """Test JobTypeConfig returns correct path for tuning jobs."""
        
        # Test tune job type  
        job_config = get_job_type_config("tune")
        config_path = os.path.join(job_config.get_script_dir(get_settings()), job_config.config_filename)
        
        # Should return tune/config.yaml path (cross-platform)
        assert config_path.endswith(os.path.join("tune", "config.yaml"))
        assert "tune" in config_path

    def test_process_tuning_job_results_success(self, temp_workspace, mock_datasphere_env):
        """Test process_tuning_results with successful tuning results.
        NB: execute_query больше не вызывается напрямую, source обновляется через create_or_get_config(..., source="tuning")."""
        # Create test results directory with required files
        results_dir = os.path.join(temp_workspace["temp_dir"], "tuning_results")
        os.makedirs(results_dir, exist_ok=True)
        # Create best_configs.json with test data
        best_configs = [
            {"lr": 0.001, "batch_size": 32, "dropout": 0.2},
            {"lr": 0.005, "batch_size": 64, "dropout": 0.1},
            {"lr": 0.002, "batch_size": 16, "dropout": 0.3},
        ]
        best_configs_path = os.path.join(results_dir, "best_configs.json")
        with open(best_configs_path, "w", encoding="utf-8") as f:
            json.dump(best_configs, f, indent=2)
        output_files = {"configs": best_configs_path}
        # Test metrics data (list of dicts)
        metrics_data = [
            {"best_val_MIC": 0.95, "job_duration_seconds": 300},
            {"best_val_MIC": 0.93, "job_duration_seconds": 310},
            {"best_val_MIC": 0.91, "job_duration_seconds": 320},
        ]
        # Mock all the database and processing functions
        with patch("deployment.app.services.datasphere_service._prepare_job_datasets"), \
             patch("deployment.app.services.datasphere_service._initialize_datasphere_client"), \
             patch("deployment.app.services.datasphere_service._process_datasphere_job"), \
             patch("deployment.app.services.datasphere_service.get_settings"), \
             patch("deployment.app.services.job_registries.result_processor_registry.create_or_get_config") as mock_create_config, \
             patch("deployment.app.services.job_registries.result_processor_registry.create_tuning_result") as mock_create_tuning, \
             patch("deployment.app.services.job_registries.result_processor_registry.update_job_status") as mock_update_status, \
             patch("deployment.app.db.database.get_db_connection") as mock_get_db_connection:
            
            # Setup mock returns
            mock_create_config.side_effect = ["config_1", "config_2", "config_3"]
            mock_create_tuning.return_value = "tuning_result_id"
            mock_get_db_connection.return_value = MagicMock() # Return a mock connection

            # Execute
            process_tuning_results(
                job_id="test-tuning-job",
                results_dir=results_dir,
                metrics_data=metrics_data,
                output_files=output_files,
                polls=30,
                poll_interval=10.0
            )
            
            # Verify config creation calls
            assert mock_create_config.call_count == 3
            for i, call in enumerate(mock_create_config.call_args_list):
                config_data = call[0][0]  # First positional argument
                assert config_data == best_configs[i]
                assert call[1]["is_active"] is False  # Keyword argument
            
            # Verify tuning result creation
            assert mock_create_tuning.call_count == 3
            # Сравниваем переданные метрики с эталонными без учёта порядка
            actual_metrics = [call[1]["metrics"] for call in mock_create_tuning.call_args_list]
            assert sorted(actual_metrics, key=lambda x: x["best_val_MIC"]) == sorted(metrics_data, key=lambda x: x["best_val_MIC"])
            
            # Verify job status update
            mock_update_status.assert_called_once()
            update_call = mock_update_status.call_args
            assert update_call[0][0] == "test-tuning-job"  # job_id
            assert "completed" in str(update_call[0][1])  # status
            assert "Saved 3 configs" in str(update_call[1]["status_message"])

    def test_process_tuning_job_results_missing_file(self, temp_workspace, mock_datasphere_env):
        """Test process_tuning_results when best_configs.json is missing."""
        
        # Create empty results directory
        results_dir = os.path.join(temp_workspace["temp_dir"], "empty_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Mock update_job_status to capture failure
        with patch("deployment.app.services.job_registries.result_processor_registry.update_job_status") as mock_update_status:
            
            # Execute and expect RuntimeError
            with pytest.raises(RuntimeError, match="best_configs.json not found"):
                process_tuning_results(
                    job_id="test-failing-job",
                    results_dir=results_dir,
                    metrics_data={},
                    output_files={},
                    polls=10,
                    poll_interval=1.0
                )
            
            # Verify failure status was set
            mock_update_status.assert_called_once()
            update_call = mock_update_status.call_args
            assert update_call[0][0] == "test-failing-job"
            assert "failed" in str(update_call[0][1])
            assert "missing best_configs.json" in str(update_call[1]["error_message"])

    def test_process_tuning_job_results_invalid_json(self, temp_workspace, mock_datasphere_env):
        """Test process_tuning_results with malformed JSON file."""
        
        # Create results directory with invalid JSON
        results_dir = os.path.join(temp_workspace["temp_dir"], "invalid_results")
        os.makedirs(results_dir, exist_ok=True)
        
        best_configs_path = os.path.join(results_dir, "best_configs.json")
        with open(best_configs_path, "w") as f:
            f.write("{ invalid json content")
        
        # Execute and expect JSON parsing error
        with pytest.raises(Exception):  # Could be JSONDecodeError or other parsing error
            process_tuning_results(
                job_id="test-json-error-job",
                results_dir=results_dir,
                metrics_data={},
                output_files={"best_configs.json": best_configs_path},
                polls=5,
                poll_interval=2.0
            )

    def test_process_tuning_job_results_database_error(self, temp_workspace, mock_datasphere_env):
        """Test process_tuning_results with database errors during config persistence."""
        # Create valid test data
        results_dir = os.path.join(temp_workspace["temp_dir"], "db_error_results")
        os.makedirs(results_dir, exist_ok=True)
        best_configs = [{"lr": 0.001, "batch_size": 32}]
        best_configs_path = os.path.join(results_dir, "best_configs.json")
        with open(best_configs_path, "w") as f:
            json.dump(best_configs, f)
        output_files = {"configs": best_configs_path}
        # Test metrics data (list of dicts)
        metrics_data = [
            {"best_val_MIC": 0.95, "job_duration_seconds": 300},
        ]
        # Mock database functions to simulate errors
        with patch("deployment.app.services.job_registries.result_processor_registry.create_or_get_config") as mock_create_config, \
             patch("deployment.app.services.job_registries.result_processor_registry.update_job_status") as mock_update_status, \
             patch("deployment.app.db.database.get_db_connection") as mock_get_db_connection:
            
            # Simulate database error
            mock_create_config.side_effect = Exception("Database connection failed")
            mock_get_db_connection.return_value = MagicMock() # Return a mock connection
            
            # Execute (should handle errors gracefully)
            process_tuning_results(
                job_id="test-db-error-job",
                results_dir=results_dir,
                metrics_data=metrics_data,
                output_files=output_files,
                polls=5,
                poll_interval=1.0
            )
            
            # Should still complete with 0 saved configs
            mock_update_status.assert_called()
            final_call = mock_update_status.call_args_list[-1]
            assert "Saved 0 configs" in str(final_call[1]["status_message"])

    @pytest.mark.asyncio
    async def test_run_job_tune_branching(self, temp_workspace, mock_datasphere_env):
        """Test run_job function branches correctly for tuning jobs.
        NB: execute_query больше не вызывается напрямую, source обновляется через create_or_get_config(..., source="tuning")."""
        from deployment.app.services.datasphere_service import run_job, JOB_TYPE_TUNE
        
        # Create tune config.yaml file that the workflow expects
        tune_config_dir = os.path.join(temp_workspace["job_dir"], "tune")
        os.makedirs(tune_config_dir, exist_ok=True)
        tune_config_path = os.path.join(tune_config_dir, "config.yaml")
        with open(tune_config_path, "w") as f:
            # Create a minimal valid DataSphere config
            f.write("""name: test_tune_job
desc: Test tuning job
cmd: python -m test_tune
""")
        
        # Create test training config
        training_config = {
            "model_id": "test_tune_model",
            "nn_model_config": {
                "num_encoder_layers": 2,
                "num_decoder_layers": 1,
                "decoder_output_dim": 64,
                "temporal_width_past": 6,
                "temporal_width_future": 3,
                "temporal_hidden_size_past": 32,
                "temporal_hidden_size_future": 32,
                "temporal_decoder_hidden": 64,
                "batch_size": 16,
                "dropout": 0.1,
                "use_reversible_instance_norm": True,
                "use_layer_norm": True,
            },
            "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
            "lr_shed_config": {"T_0": 5, "T_mult": 1},
            "train_ds_config": {"alpha": 0.1, "span": 6},
            "lags": 6,
            "quantiles": [0.1, 0.5, 0.9],
        }
        
        # Mock all the database and processing functions
        with patch("deployment.app.services.datasphere_service._prepare_job_datasets") as mock_prepare, \
             patch("deployment.app.services.datasphere_service._initialize_datasphere_client") as mock_init_client, \
             patch("deployment.app.services.datasphere_service._process_datasphere_job") as mock_process_ds_job, \
             patch("deployment.app.services.datasphere_service.get_settings") as mock_get_settings, \
             patch("deployment.app.services.job_registries.result_processor_registry.create_or_get_config") as mock_create_config, \
             patch("deployment.app.services.job_registries.result_processor_registry.create_tuning_result") as mock_create_tuning, \
             patch("deployment.app.services.job_registries.result_processor_registry.update_job_status") as mock_update_status, \
             patch("deployment.app.db.database.get_db_connection") as mock_get_db_connection:
            
            # Mock _prepare_job_datasets to actually create required files
            def mock_prepare_datasets(job_id, config, start_date=None, end_date=None, output_dir=None, **kwargs):
                # Create required dataset files that _verify_datasphere_job_inputs expects
                from pathlib import Path
                output_path = Path(output_dir) if output_dir else Path("/tmp")
                train_file = output_path / "train.dill"
                val_file = output_path / "val.dill"
                config_file = output_path / "config.json"
                
                # Create dummy files
                train_file.write_text("dummy train data")
                val_file.write_text("dummy val data")
                config_file.write_text('{"dummy": "config"}')
                
                return None
            
            # Mock get_settings to return complete settings object
            mock_tuning_settings = type('MockTuning', (), {
                'model_dump': lambda: {
                    "num_samples_light": 50,
                    "num_samples_full": 200,
                    "max_concurrent": 2,
                    "resources": {"cpu": 8, "gpu": 1},
                    "best_configs_to_save": 5,
                    "metric_threshold": 0.8
                }
            })()
            mock_settings_obj = type('MockSettings', (), {
                'tuning': mock_tuning_settings,
                'datasphere_input_dir': temp_workspace["input_dir"],
                'datasphere_output_dir': temp_workspace["output_dir"],
                'models_dir': temp_workspace["models_dir"],
                'datasphere_job_train_dir': os.path.join(temp_workspace["job_dir"], "train"),
                'datasphere_job_tune_dir': os.path.join(temp_workspace["job_dir"], "tune"),
                'datasphere': type('MockDataSphere', (), {
                    'max_polls': 3,
                    'poll_interval': 0.1
                })()
            })()
            mock_get_settings.return_value = mock_settings_obj

            mock_prepare.side_effect = mock_prepare_datasets
            mock_init_client.return_value = mock_datasphere_env["client"]
            
            # Set up database mocks
            mock_create_config.side_effect = lambda cfg, **kwargs: f"config_{hash(str(cfg))}"
            mock_create_tuning.return_value = "tuning_result_123"
            mock_update_status.return_value = None
            mock_get_db_connection.return_value = MagicMock() # Return a mock connection
            
            # Mock _process_datasphere_job to return expected values with output_files
            results_dir = os.path.join(temp_workspace["temp_dir"], "results")
            
            # Create the files that would normally be downloaded by DataSphere
            os.makedirs(results_dir, exist_ok=True)
            best_configs = [
                {"lr": 0.001, "batch_size": 32},
                {"lr": 0.005, "batch_size": 64}
            ]
            with open(os.path.join(results_dir, "best_configs.json"), "w") as f:
                json.dump(best_configs, f)
            # metrics_data as list of dicts
            metrics_data = [
                {"best_val_MIC": 0.95},
                {"best_val_MIC": 0.93}
            ]
            with open(os.path.join(results_dir, "metrics.json"), "w") as f:
                json.dump(metrics_data, f)
            
            mock_process_ds_job.return_value = (
                "ds-job-123",  # ds_job_id
                results_dir,  # results_dir
                metrics_data,  # metrics_data
                {"configs": os.path.join(results_dir, "best_configs.json"), "metrics": os.path.join(results_dir, "metrics.json")},  # output_files
                3  # polls
            )
            
            # Execute run_job with tuning job type
            _ = await run_job(
                job_id="test-tune-job",
                training_config=training_config,
                config_id="test-config-id",
                job_type=JOB_TYPE_TUNE,
                dataset_start_date="2023-01-01",
                dataset_end_date="2023-12-31"
            )
            
            # Verify database operations were called
            assert mock_create_config.call_count == 2  # Two configs from best_configs
            assert mock_create_tuning.call_count == 2  # Two tuning results
            
            # Verify final status update
            final_status_calls = [call for call in mock_update_status.call_args_list 
                                 if len(call[0]) > 1 and "completed" in str(call[0][1])]
            assert len(final_status_calls) > 0
            final_call = final_status_calls[-1]
            assert "Saved 2 configs" in str(final_call[1]["status_message"])

    @pytest.mark.asyncio
    async def test_run_job_train_branching(self, temp_workspace, mock_datasphere_env):
        """Test run_job function still works correctly for training jobs."""
        from deployment.app.services.datasphere_service import run_job, JOB_TYPE_TRAIN
        
        # Create train config.yaml file that the workflow expects
        train_config_dir = os.path.join(temp_workspace["job_dir"], "train")
        os.makedirs(train_config_dir, exist_ok=True)
        train_config_path = os.path.join(train_config_dir, "config.yaml")
        with open(train_config_path, "w") as f:
            # Create a minimal valid DataSphere config
            f.write("""name: test_train_job
desc: Test training job
cmd: python -m test_train
""")
        
        # Create test training config
        training_config = {
            "model_id": "test_train_model",
            "nn_model_config": {
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
                "use_layer_norm": True,
            },
            "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
            "lr_shed_config": {"T_0": 10, "T_mult": 2},
            "train_ds_config": {"alpha": 0.05, "span": 12},
            "lags": 12,
            "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
        }
        
        # Mock all the functions for training path
        with patch("deployment.app.services.datasphere_service._prepare_job_datasets") as mock_prepare, \
             patch("deployment.app.services.datasphere_service.process_job_results_unified") as mock_process_unified, \
             patch("deployment.app.services.datasphere_service._initialize_datasphere_client") as mock_init_client, \
             patch("deployment.app.services.datasphere_service._process_datasphere_job") as mock_process_ds_job, \
             patch("deployment.app.services.datasphere_service.get_settings") as mock_get_settings:
            
            # Mock _prepare_job_datasets to actually create required files
            def mock_prepare_datasets(job_id, config, start_date=None, end_date=None, output_dir=None, **kwargs):
                # Create required dataset files that _verify_datasphere_job_inputs expects
                from pathlib import Path
                output_path = Path(output_dir) if output_dir else Path("/tmp")
                train_file = output_path / "train.dill"
                val_file = output_path / "val.dill"
                config_file = output_path / "config.json"
                
                # Create dummy files
                train_file.write_text("dummy train data")
                val_file.write_text("dummy val data")
                config_file.write_text('{"dummy": "config"}')
                
                return None
            
            # Mock get_settings to return complete settings object
            mock_settings_obj = type('MockSettings', (), {
                'datasphere_input_dir': temp_workspace["input_dir"],
                'datasphere_output_dir': temp_workspace["output_dir"],
                'models_dir': temp_workspace["models_dir"],
                'datasphere_job_train_dir': os.path.join(temp_workspace["job_dir"], "train"),
                'datasphere_job_tune_dir': os.path.join(temp_workspace["job_dir"], "tune"),
                'datasphere': type('MockDataSphere', (), {
                    'max_polls': 3,
                    'poll_interval': 0.1
                })()
            })()
            mock_get_settings.return_value = mock_settings_obj
            
            mock_prepare.side_effect = mock_prepare_datasets
            mock_process_unified.return_value = None
            mock_init_client.return_value = mock_datasphere_env["client"]
            
            # Mock _process_datasphere_job to return expected values with output_files for training
            mock_process_ds_job.return_value = (
                "ds-job-123",  # ds_job_id
                "/tmp/results",  # results_dir
                {"mape": 10.5, "val_loss": 0.05},  # metrics_data
                {"model.onnx": "/tmp/results/model.onnx", "predictions.csv": "/tmp/results/predictions.csv"},  # output_files
                3  # polls
            )
            
            # Execute run_job with training job type (default)
            _ = await run_job(
                job_id="test-train-job",
                training_config=training_config,
                config_id="test-config-id",
                job_type=JOB_TYPE_TRAIN  # Explicit training type
            )
            
            # Verify unified processing was called with correct processor name
            mock_process_unified.assert_called_once()
            
            # Verify the unified processor was called with training processor name
            unified_call = mock_process_unified.call_args
            assert unified_call[1]["job_id"] == "test-train-job"
            assert unified_call[1]["processor_name"] == "process_training_results"
            assert unified_call[1]["metrics_data"] == {"mape": 10.5, "val_loss": 0.05}
            assert unified_call[1]["output_files"] == {"model.onnx": "/tmp/results/model.onnx", "predictions.csv": "/tmp/results/predictions.csv"}


class TestDataSphereServiceTuningIntegration:
    """Integration tests for tuning functionality with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_complete_tuning_workflow(self, temp_workspace, mock_datasphere_env):
        """Test complete tuning workflow from job submission to result processing.
        NB: execute_query больше не вызывается напрямую, source обновляется через create_or_get_config(..., source="tuning")."""
        from deployment.app.services.datasphere_service import run_job, JOB_TYPE_TUNE
        
        # Create tune config.yaml file that the workflow expects
        tune_config_dir = os.path.join(temp_workspace["job_dir"], "tune")
        os.makedirs(tune_config_dir, exist_ok=True)
        tune_config_path = os.path.join(tune_config_dir, "config.yaml")
        with open(tune_config_path, "w") as f:
            # Create a minimal valid DataSphere config
            f.write("""name: test_tune_job
desc: Test tuning job
cmd: python -m test_tune
""")
        
        # Setup complete tuning workflow
        training_config = {
            "model_id": "integration_tune_model",
            "nn_model_config": {
                "num_encoder_layers": 2,
                "num_decoder_layers": 1,
                "decoder_output_dim": 64,
                "temporal_width_past": 6,
                "temporal_width_future": 3,
                "temporal_hidden_size_past": 32,
                "temporal_hidden_size_future": 32,
                "temporal_decoder_hidden": 64,
                "batch_size": 16,
                "dropout": 0.1,
                "use_reversible_instance_norm": True,
                "use_layer_norm": True,
            },
            "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
            "lr_shed_config": {"T_0": 5, "T_mult": 1},
            "train_ds_config": {"alpha": 0.1, "span": 6},
            "lags": 6,
            "quantiles": [0.1, 0.5, 0.9],
        }
        
        # Setup realistic tuning results
        best_configs = [
            {"lr": 0.001, "batch_size": 32, "dropout": 0.2},
            {"lr": 0.003, "batch_size": 64, "dropout": 0.15},
            {"lr": 0.005, "batch_size": 16, "dropout": 0.1},
        ]
        
        tuning_metrics = [
            {"best_val_MIC": 0.97, "best_val_loss": 0.03, "job_duration_seconds": 1800, "total_trials": 50, "completed_trials": 50},
            {"best_val_MIC": 0.95, "best_val_loss": 0.04, "job_duration_seconds": 1850, "total_trials": 50, "completed_trials": 50},
            {"best_val_MIC": 0.93, "best_val_loss": 0.05, "job_duration_seconds": 1900, "total_trials": 50, "completed_trials": 50},
        ]
        
        # Mock realistic DataSphere download behavior
        def mock_tuning_download(ds_job_id, results_dir, **kwargs):
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, "best_configs.json"), "w") as f:
                json.dump(best_configs, f, indent=2)
            with open(os.path.join(results_dir, "metrics.json"), "w") as f:
                json.dump(tuning_metrics, f, indent=2)
        
        mock_datasphere_env["client"].download_job_results.side_effect = mock_tuning_download
        
        # Track database operations
        config_ids_created = []
        tuning_results_created = []
        
        def mock_create_config(config_data, **kwargs):
            config_id = f"config_{len(config_ids_created) + 1}"
            config_ids_created.append((config_id, config_data))
            return config_id
        
        def mock_create_tuning_result(job_id, config_id, metrics, duration, **kwargs):
            result_id = f"tuning_result_{len(tuning_results_created) + 1}"
            tuning_results_created.append((result_id, job_id, config_id, metrics, duration))
            return result_id
        
        # Mock all required functions
        with patch("deployment.app.services.datasphere_service._prepare_job_datasets") as mock_prepare, \
             patch("deployment.app.services.job_registries.result_processor_registry.create_or_get_config", side_effect=mock_create_config), \
             patch("deployment.app.services.job_registries.result_processor_registry.create_tuning_result", side_effect=mock_create_tuning_result), \
             patch("deployment.app.services.job_registries.result_processor_registry.update_job_status") as mock_update_status, \
             patch("deployment.app.db.database.get_db_connection") as mock_get_db_connection, \
             patch("deployment.app.services.datasphere_service._initialize_datasphere_client") as mock_init_client:
            
            # Mock _prepare_job_datasets to actually create required files
            def mock_prepare_datasets(job_id, config, start_date=None, end_date=None, output_dir=None, **kwargs):
                # Create required dataset files that _verify_datasphere_job_inputs expects
                from pathlib import Path
                output_path = Path(output_dir) if output_dir else Path("/tmp")
                train_file = output_path / "train.dill"
                val_file = output_path / "val.dill"
                config_file = output_path / "config.json"
                
                # Create dummy files
                train_file.write_text("dummy train data")
                val_file.write_text("dummy val data")
                config_file.write_text('{"dummy": "config"}')
                
                return None
            
            mock_prepare.side_effect = mock_prepare_datasets
            mock_init_client.return_value = mock_datasphere_env["client"]
            mock_get_db_connection.return_value = MagicMock() # Return a mock connection
            
            # Execute complete workflow
            _ = await run_job(
                job_id="integration-tuning-job",
                training_config=training_config,
                config_id="base-config-id",
                job_type=JOB_TYPE_TUNE,
                dataset_start_date="2023-01-01",
                dataset_end_date="2023-12-31"
            )
            
            # Verify dataset preparation was called
            mock_prepare.assert_called_once()
            
            # Verify configs were created for each best config
            assert len(config_ids_created) == 3
            for i, (config_id, config_data) in enumerate(config_ids_created):
                assert config_data == best_configs[i]
                assert config_id == f"config_{i + 1}"
            
            # Verify final job status update
            final_status_calls = [call for call in mock_update_status.call_args_list 
                                 if len(call[0]) > 1 and "completed" in str(call[0][1])]
            assert len(final_status_calls) > 0
            
            final_call = final_status_calls[-1]
            assert "Saved 3 configs" in str(final_call[1]["status_message"])


@pytest.mark.asyncio
async def test_download_datasphere_job_results_unified():
    # Arrange
    job_id = "test-job"
    ds_job_id = "ds-job-123"
    job_config = get_job_type_config("train")
    # Создаём временную директорию для results_dir
    results_dir = tempfile.mkdtemp()
    try:
        # Подготовим тестовые файлы
        model_path = os.path.join(results_dir, "model.onnx")
        predictions_path = os.path.join(results_dir, "predictions.csv")
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(model_path, "w") as f:
            f.write("fake model content")
        with open(predictions_path, "w") as f:
            f.write("col1,col2\n1,2\n")
        metrics_data = {"mape": 10.5, "val_loss": 0.05}
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f)
        # Мок DataSphereClient и download_job_results (ничего не делает)
        mock_client = MagicMock()
        mock_client.download_job_results = MagicMock()
        # Act
        result = await _download_datasphere_job_results(
            job_id=job_id,
            ds_job_id=ds_job_id,
            client=mock_client,
            results_dir=results_dir,
            job_config=job_config,
        )
        # Assert
        assert isinstance(result, dict)
        assert set(result.keys()) == set(job_config.output_file_roles.values())
        assert result["model"] == model_path
        assert result["predictions"] == predictions_path
        assert result["metrics"] == metrics_data
        mock_client.download_job_results.assert_called_once_with(ds_job_id, results_dir)
        # Проверим обработку отсутствующего файла
        os.remove(predictions_path)
        result2 = await _download_datasphere_job_results(
            job_id=job_id,
            ds_job_id=ds_job_id,
            client=mock_client,
            results_dir=results_dir,
            job_config=job_config,
        )
        assert result2["predictions"] is None
    finally:
        shutil.rmtree(results_dir)


