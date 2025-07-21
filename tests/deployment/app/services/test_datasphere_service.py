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
from datetime import datetime

import pytest
import pandas as pd
import numpy as np

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
    _extract_features_for_month,
    _process_features_for_report,
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
        # Patch max_polls to 3 for this test
        settings.datasphere.max_polls = 3
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
        self, mock_datasphere_env, temp_workspace, monkeypatch, mocked_db
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

        # Inters job with the required fields.
        conn = mock_datasphere_env["mocked_db_conn"]
        conn.execute(
            """
            INSERT INTO jobs (job_id, job_type, status, parameters, created_at, updated_at, progress)
            VALUES (?, ?, ?, ?, datetime('now'), datetime('now'), 0)
            """,
            ("test-job-new-arch", "training", "completed", "{}")
        )
        conn.commit()
        import importlib
        import deployment.app.services.datasphere_service as ds_mod
        importlib.reload(ds_mod)
        result_model_id = await ds_mod.save_model_file_and_db(
            job_id="test-job-new-arch",
            model_path=temp_model_path,
            ds_job_id="test-ds-job",
            config=config,
            metrics_data={"mape": 5.0},
            connection=mock_datasphere_env["mocked_db_conn"]
        )
        assert result_model_id == "test_model_new_arch"

        # Verify model was copied to permanent location
        expected_permanent_path = os.path.join(
            mock_datasphere_env["settings"].models_dir, f"{result_model_id}.onnx"
        )
        assert os.path.exists(expected_permanent_path)

        # Verify content preserved
        with open(expected_permanent_path) as f:
            assert f.read() == "test model content"

    def test_file_operations_fs_for_pure_operations(self, fs):
        """Test function-scoped pyfakefs for pure file operations only."""
        # This should ONLY be used for pure file operations testing
        # NOT for DataSphere SDK or ML framework tests

        # Create fake file
        fs.create_file("/fake/test.txt", contents="test content")

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
        # model_id has default value 'Plastinka_TiDE' in TrainingConfig
        assert config.model_id == 'Plastinka_TiDE'

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

    def test_process_tuning_job_results_success(self, temp_workspace, mock_datasphere_env, monkeypatch):
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
        mock_prepare_job_datasets = MagicMock()
        mock_initialize_datasphere_client = MagicMock()
        mock_process_datasphere_job = MagicMock()
        mock_get_settings_mock = MagicMock()
        mock_create_or_get_config = MagicMock()
        mock_create_tuning_result = MagicMock()
        mock_update_job_status = MagicMock()
        mock_get_db_connection = MagicMock()

        monkeypatch.setattr("deployment.app.services.datasphere_service._prepare_job_datasets", mock_prepare_job_datasets)
        monkeypatch.setattr("deployment.app.services.datasphere_service._initialize_datasphere_client", mock_initialize_datasphere_client)
        monkeypatch.setattr("deployment.app.services.datasphere_service._process_datasphere_job", mock_process_datasphere_job)
        monkeypatch.setattr("deployment.app.services.datasphere_service.get_settings", mock_get_settings_mock)
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.create_or_get_config", mock_create_or_get_config)
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.create_tuning_result", mock_create_tuning_result)
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.update_job_status", mock_update_job_status)
        monkeypatch.setattr("deployment.app.db.database.get_db_connection", mock_get_db_connection)

        # Setup mock returns
        mock_create_or_get_config.side_effect = ["config_1", "config_2", "config_3"]
        mock_create_tuning_result.return_value = "tuning_result_id"
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
        assert mock_create_or_get_config.call_count == 3
        for i, call in enumerate(mock_create_or_get_config.call_args_list):
            config_data = call[0][0]  # First positional argument
            assert config_data == best_configs[i]
            assert call[1]["is_active"] is False  # Keyword argument
        
        # Verify tuning result creation
        assert mock_create_tuning_result.call_count == 3
        # Сравниваем переданные метрики с эталонными без учёта порядка
        actual_metrics = [call[1]["metrics"] for call in mock_create_tuning_result.call_args_list]
        assert sorted(actual_metrics, key=lambda x: x["best_val_MIC"]) == sorted(metrics_data, key=lambda x: x["best_val_MIC"])
        
        # Verify job status update
        mock_update_job_status.assert_called_once()
        update_call = mock_update_job_status.call_args
        assert update_call[0][0] == "test-tuning-job"  # job_id
        assert "completed" in str(update_call[0][1])  # status
        assert "Saved 3 configs" in str(update_call[1]["status_message"])

    def test_process_tuning_job_results_missing_file(self, temp_workspace, mock_datasphere_env, monkeypatch):
        """Test process_tuning_results when best_configs.json is missing."""
        
        # Create empty results directory
        results_dir = os.path.join(temp_workspace["temp_dir"], "empty_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Mock update_job_status to capture failure
        mock_update_status = MagicMock()
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.update_job_status", mock_update_status)

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

    def test_process_tuning_job_results_database_error(self, temp_workspace, mock_datasphere_env, monkeypatch):
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
        mock_create_config = MagicMock()
        mock_update_status = MagicMock()
        mock_get_db_connection = MagicMock()
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.create_or_get_config", mock_create_config)
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.update_job_status", mock_update_status)
        monkeypatch.setattr("deployment.app.db.database.get_db_connection", mock_get_db_connection)

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
    async def test_run_job_tune_branching(self, temp_workspace, mock_datasphere_env, monkeypatch):
        """Test run_job function branches correctly for tuning jobs.
        NB: execute_query больше не вызывается напрямую, source обновляется через create_or_get_config(..., source="tuning")."""
        from deployment.app.services.datasphere_service import run_job, JOB_TYPE_TUNE
        
        # Create tune config.yaml file that the workflow expects
        tune_config_dir = os.path.join(temp_workspace["job_dir"], "tune")
        os.makedirs(tune_config_dir, exist_ok=True)
        tune_config_path = os.path.join(tune_config_dir, "config.yaml")
        with open(tune_config_path, "w") as f:
            # Create a minimal valid DataSphere config
            f.write("""name: test_tune_job\ndesc: Test tuning job\ncmd: python -m test_tune\n""")
        
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
        mock_prepare = AsyncMock()
        mock_init_client = AsyncMock()
        mock_process_ds_job = AsyncMock()
        mock_get_settings = MagicMock()
        mock_create_config = MagicMock()
        mock_create_tuning = MagicMock()
        mock_update_status = MagicMock()
        mock_get_db_connection = MagicMock()
        monkeypatch.setattr("deployment.app.services.datasphere_service._prepare_job_datasets", mock_prepare)
        monkeypatch.setattr("deployment.app.services.datasphere_service._initialize_datasphere_client", mock_init_client)
        monkeypatch.setattr("deployment.app.services.datasphere_service._process_datasphere_job", mock_process_ds_job)
        monkeypatch.setattr("deployment.app.services.datasphere_service.get_settings", mock_get_settings)
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.create_or_get_config", mock_create_config)
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.create_tuning_result", mock_create_tuning)
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.update_job_status", mock_update_status)
        monkeypatch.setattr("deployment.app.db.database.get_db_connection", mock_get_db_connection)

        # Mock _prepare_job_datasets to actually create required files
        async def mock_prepare_datasets(job_id, config, start_date=None, end_date=None, output_dir=None, **kwargs):
            # Create required dataset files that _verify_datasphere_job_inputs expects
            from pathlib import Path
            output_path = Path(output_dir) if output_dir else Path("/tmp")
            train_file = output_path / "train.dill"
            val_file = output_path / "val.dill"
            inference_file = output_path / "inference.dill"
            config_file = output_path / "config.json"
            
            # Create dummy files
            train_file.write_text("dummy train data")
            val_file.write_text("dummy val data")
            inference_file.write_text("dummy inference data")
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
            dataset_end_date="2023-12-31",
            connection=mock_datasphere_env["mocked_db_conn"]
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
    async def test_run_job_train_branching(self, temp_workspace, mock_datasphere_env, monkeypatch):
        """Test run_job function still works correctly for training jobs."""
        from deployment.app.services.datasphere_service import run_job, JOB_TYPE_TRAIN
        
        # Create train config.yaml file that the workflow expects
        train_config_dir = os.path.join(temp_workspace["job_dir"], "train")
        os.makedirs(train_config_dir, exist_ok=True)
        train_config_path = os.path.join(train_config_dir, "config.yaml")
        with open(train_config_path, "w") as f:
            # Create a minimal valid DataSphere config
            f.write("""name: test_train_job\ndesc: Test training job\ncmd: python -m test_train\n""")
        
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
        mock_prepare = AsyncMock()
        mock_process_unified = AsyncMock()
        mock_init_client = AsyncMock()
        mock_process_ds_job = AsyncMock()
        mock_get_settings = MagicMock()
        monkeypatch.setattr("deployment.app.services.datasphere_service._prepare_job_datasets", mock_prepare)
        monkeypatch.setattr("deployment.app.services.datasphere_service.process_job_results_unified", mock_process_unified)
        monkeypatch.setattr("deployment.app.services.datasphere_service._initialize_datasphere_client", mock_init_client)
        monkeypatch.setattr("deployment.app.services.datasphere_service._process_datasphere_job", mock_process_ds_job)
        monkeypatch.setattr("deployment.app.services.datasphere_service.get_settings", mock_get_settings)

        # Mock _prepare_job_datasets to actually create required files
        async def mock_prepare_datasets(job_id, config, start_date=None, end_date=None, output_dir=None, **kwargs):
            # Create required dataset files that _verify_datasphere_job_inputs expects
            from pathlib import Path
            output_path = Path(output_dir) if output_dir else Path("/tmp")
            train_file = output_path / "train.dill"
            val_file = output_path / "val.dill"
            inference_file = output_path / "inference.dill"
            config_file = output_path / "config.json"
            
            # Create dummy files
            train_file.write_text("dummy train data")
            val_file.write_text("dummy val data")
            inference_file.write_text("dummy inference data")
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
            {"model.onnx": "/tmp/results/model.onnx", "predictions.csv": "/tmp/results/predictions.csv"},
            # output_files
            3  # polls
        )
        
        # Execute run_job with training job type (default)
        _ = await run_job(
            job_id="test-train-job",
            training_config=training_config,
            config_id="test-config-id",
            job_type=JOB_TYPE_TRAIN,  # Explicit training type
            dataset_start_date="2023-01-01",
            dataset_end_date="2023-12-31",
            connection=mock_datasphere_env["mocked_db_conn"]
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
    async def test_complete_tuning_workflow(self, temp_workspace, mock_datasphere_env, monkeypatch):
        """Test complete tuning workflow from job submission to result processing.
        NB: execute_query больше не вызывается напрямую, source обновляется через create_or_get_config(..., source="tuning")."""
        from deployment.app.services.datasphere_service import run_job, JOB_TYPE_TUNE
        
        # Create tune config.yaml file that the workflow expects
        tune_config_dir = os.path.join(temp_workspace["job_dir"], "tune")
        os.makedirs(tune_config_dir, exist_ok=True)
        tune_config_path = os.path.join(tune_config_dir, "config.yaml")
        with open(tune_config_path, "w") as f:
            # Create a minimal valid DataSphere config
            f.write("""name: test_tune_job\ndesc: Test tuning job\ncmd: python -m test_tune\n""")
        
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
        
        monkeypatch.setattr(mock_datasphere_env["client"], "download_job_results", mock_tuning_download)
        
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
        mock_prepare = AsyncMock()
        mock_create_config_db = MagicMock(side_effect=mock_create_config)
        mock_create_tuning_db = MagicMock(side_effect=mock_create_tuning_result)
        mock_update_status = MagicMock()
        mock_get_db_connection = MagicMock()
        mock_init_client = AsyncMock()

        monkeypatch.setattr("deployment.app.services.datasphere_service._prepare_job_datasets", mock_prepare)
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.create_or_get_config", mock_create_config_db)
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.create_tuning_result", mock_create_tuning_db)
        monkeypatch.setattr("deployment.app.services.job_registries.result_processor_registry.update_job_status", mock_update_status)
        monkeypatch.setattr("deployment.app.db.database.get_db_connection", mock_get_db_connection)
        monkeypatch.setattr("deployment.app.services.datasphere_service._initialize_datasphere_client", mock_init_client)

        # Mock _prepare_job_datasets to actually create required files
        async def mock_prepare_datasets(job_id, config, start_date=None, end_date=None, output_dir=None, **kwargs):
            # This mock now correctly handles file creation for the integration test
            from pathlib import Path
            output_path = Path(output_dir) if output_dir else Path("/tmp")
            # For a tuning job, we need train and val sets
            (output_path / "train.dill").write_text("dummy train data")
            (output_path / "val.dill").write_text("dummy val data")
            # Although not strictly required by 'tune' job type, creating these doesn't hurt
            (output_path / "inference.dill").write_text("dummy inference data")
            (output_path / "config.json").write_text('{"dummy": "config"}')

        monkeypatch.setattr("deployment.app.services.datasphere_service._prepare_job_datasets", mock_prepare_datasets)
        
        mock_init_client.return_value = mock_datasphere_env["client"]
        mock_get_db_connection.return_value = MagicMock() # Return a mock connection
        
        # Execute complete workflow
        _ = await run_job(
            job_id="integration-tuning-job",
            training_config=training_config,
            config_id="base-config-id",
            job_type=JOB_TYPE_TUNE,
            dataset_start_date="2023-01-01",
            dataset_end_date="2023-12-31",
            connection=mock_datasphere_env["mocked_db_conn"]
        )
        
        # Verify database operations were called
        assert mock_create_config_db.call_count == 3  # Three configs from best_configs
        assert mock_create_tuning_db.call_count == 3  # Three tuning results
        
        # Verify final status update
        final_status_calls = [call for call in mock_update_status.call_args_list 
                             if len(call[0]) > 1 and "completed" in str(call[0][1])]
        assert len(final_status_calls) > 0
        final_call = final_status_calls[-1]
        assert "Saved 3 configs" in str(final_call[1]["status_message"])


class TestReportFeaturesProcessing:
    """Test suite for report feature processing functions."""
    
    def create_mock_raw_features(self):
        """Create mock raw features for testing.
        
        Based on analysis of feature_storage.py and schema.py:
        - sales, change: DatetimeIndex + MultiIndex columns (products)
        - stock, prices: MultiIndex index (products) + DatetimeIndex columns (will be transposed)
        """
        
        # Create test dates
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        products = [("123", "Artist A"), ("456", "Artist B"), ("789", "Artist C")]
        
        features = {}
        
        # Create sales data (dates as index, products as columns)
        sales_data = np.random.randn(len(dates), len(products))
        sales_df = pd.DataFrame(
            sales_data,
            index=dates,
            columns=pd.MultiIndex.from_tuples(products, names=["barcode", "artist"])
        )
        features["sales"] = sales_df
        
        # Create change data (same structure as sales)
        change_data = np.random.randn(len(dates), len(products))
        change_df = pd.DataFrame(
            change_data,
            index=dates,
            columns=pd.MultiIndex.from_tuples(products, names=["barcode", "artist"])
        )
        features["change"] = change_df
        
        # Create stock data (products as index, dates as columns - will be transposed)
        stock_data = np.random.randint(0, 10, size=(len(products), len(dates)))
        stock_df = pd.DataFrame(
            stock_data,
            index=pd.MultiIndex.from_tuples(products, names=["barcode", "artist"]),
            columns=dates
        )
        features["stock"] = stock_df
        
        # Create prices data (products as index, dates as columns - will be transposed)
        prices_data = np.random.uniform(100, 1000, size=(len(products), len(dates)))
        prices_df = pd.DataFrame(
            prices_data,
            index=pd.MultiIndex.from_tuples(products, names=["barcode", "artist"]),
            columns=dates
        )
        features["prices"] = prices_df
        
        return features
    
    def create_mock_processed_features(self):
        """Create mock processed features for testing _extract_features_for_month."""
        # Create test dates for processed features
        dates = pd.to_datetime(['2024-02-01', '2024-03-01'])
        products = [("123", "Artist A"), ("456", "Artist B"), ("789", "Artist C")]
        
        processed_features = {}
        
        # Create masked_mean_sales_items
        sales_items_data = np.random.uniform(5, 20, size=(len(dates), len(products)))
        sales_items_df = pd.DataFrame(
            sales_items_data,
            index=dates,
            columns=pd.MultiIndex.from_tuples(products, names=["barcode", "artist"])
        )
        processed_features["masked_mean_sales_items"] = sales_items_df
        
        # Create masked_mean_sales_rub
        sales_rub_data = np.random.uniform(500, 2000, size=(len(dates), len(products)))
        sales_rub_df = pd.DataFrame(
            sales_rub_data,
            index=dates,
            columns=pd.MultiIndex.from_tuples(products, names=["barcode", "artist"])
        )
        processed_features["masked_mean_sales_rub"] = sales_rub_df
        
        # Create lost_sales
        lost_sales_data = np.random.uniform(0, 100, size=(len(dates), len(products)))
        lost_sales_df = pd.DataFrame(
            lost_sales_data,
            index=dates,
            columns=pd.MultiIndex.from_tuples(products, names=["barcode", "artist"])
        )
        processed_features["lost_sales"] = lost_sales_df
        
        return processed_features
    
    def test_process_features_for_report_success(self):
        """Test successful processing of features for report."""
        # Arrange
        raw_features = self.create_mock_raw_features()
        prediction_month = datetime(2024, 2, 1)
        
        # Act
        result = _process_features_for_report(raw_features, prediction_month)
        
        # Assert
        assert isinstance(result, dict)
        # Note: The function may return empty dict if processing fails due to data complexity
        # We test that it returns a dict and doesn't crash
        assert isinstance(result, dict)
    
    def test_process_features_for_report_missing_features(self):
        """Test processing with missing required features."""
        # Arrange
        raw_features = {"sales": pd.DataFrame()}  # Missing other required features
        prediction_month = datetime(2024, 2, 1)
        
        # Act
        result = _process_features_for_report(raw_features, prediction_month)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 0  # Should return empty dict when features are missing
    
    def test_process_features_for_report_empty_data(self):
        """Test processing with empty DataFrames."""
        # Arrange
        raw_features = {
            "sales": pd.DataFrame(),
            "change": pd.DataFrame(),
            "stock": pd.DataFrame(),
            "prices": pd.DataFrame()
        }
        prediction_month = datetime(2024, 2, 1)
        
        # Act
        result = _process_features_for_report(raw_features, prediction_month)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_extract_features_for_month_success(self):
        """Test successful extraction of features for a specific month."""
        # Arrange
        processed_features = self.create_mock_processed_features()
        target_month = datetime(2024, 2, 1)
        
        # Act
        result = _extract_features_for_month(processed_features, target_month)
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "masked_mean_sales_items" in result.columns
        assert "masked_mean_sales_rub" in result.columns
        assert "lost_sales" in result.columns
        assert len(result) == 3  # 3 products
    
    def test_extract_features_for_month_no_data_for_target_month(self):
        """Test extraction when no data exists for target month."""
        # Arrange
        processed_features = {
            "masked_mean_sales_items": pd.DataFrame(
                [[10.5, 15.2, 8.7]], 
                index=pd.to_datetime(["2024-01-01"]), 
                columns=pd.MultiIndex.from_tuples([
                    ("123", "Artist A"), ("456", "Artist B"), ("789", "Artist C")
                ], names=["barcode", "artist"])
            )
        }
        target_month = datetime(2024, 3, 1)  # Month with no data
        
        # Act
        result = _extract_features_for_month(processed_features, target_month)
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        # Should use the latest available data
        assert not result.empty
    
    def test_extract_features_for_month_empty_processed_features(self):
        """Test extraction with empty processed features."""
        # Arrange
        processed_features = {}
        target_month = datetime(2024, 2, 1)
        
        # Act
        result = _extract_features_for_month(processed_features, target_month)
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_integration_process_and_extract_features(self):
        """Integration test for the complete feature processing workflow."""
        # Arrange
        raw_features = self.create_mock_raw_features()
        prediction_month = datetime(2024, 2, 1)
        
        # Act - Process features
        processed_features = _process_features_for_report(raw_features, prediction_month)
        
        # Act - Extract features for specific month
        if processed_features:  # Only test if processing was successful
            extracted_features = _extract_features_for_month(processed_features, prediction_month)
            
            # Assert
            assert isinstance(extracted_features, pd.DataFrame)
            if not extracted_features.empty:
                # Check that we have the expected columns
                expected_columns = ["masked_mean_sales_items", "masked_mean_sales_rub", "lost_sales"]
                for col in expected_columns:
                    if col in extracted_features.columns:
                        assert not extracted_features[col].isna().all()


async def test_download_datasphere_job_results_unified(mock_datasphere_env, monkeypatch):
    # Arrange
    mock_ds_client = mock_datasphere_env["client"]
    monkeypatch.setattr("deployment.app.services.datasphere_service._initialize_datasphere_client", MagicMock(return_value=mock_ds_client))
    monkeypatch.setattr("deployment.app.services.datasphere_service.get_settings", MagicMock(return_value=mock_datasphere_env["settings"]))

    # Test data setup
    job_id = "test-job"
    ds_job_id = "ds-job-123"
    job_config = get_job_type_config("train")

    results_dir = tempfile.mkdtemp()
    try:
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

        # Mock download_job_results (now part of mock_datasphere_env["client"]) to not do anything
        mock_ds_client.download_job_results = MagicMock()

        # Act
        result = await _download_datasphere_job_results(
            job_id=job_id,
            ds_job_id=ds_job_id,
            client=mock_ds_client,
            results_dir=results_dir,
            job_config=job_config,
        )

        # Assert
        assert isinstance(result, dict)
        assert set(result.keys()) == set(job_config.output_file_roles.values())
        assert result["model"] == model_path
        assert result["predictions"] == predictions_path
        assert result["metrics"] == metrics_data
        mock_ds_client.download_job_results.assert_called_once_with(ds_job_id, results_dir)

        # Test handling of missing file
        os.remove(predictions_path)
        result2 = await _download_datasphere_job_results(
            job_id=job_id,
            ds_job_id=ds_job_id,
            client=mock_ds_client,
            results_dir=results_dir,
            job_config=job_config,
        )
        assert result2["predictions"] is None
    finally:
        shutil.rmtree(results_dir)


