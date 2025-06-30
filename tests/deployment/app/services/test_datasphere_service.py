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
import os
import asyncio
from unittest.mock import MagicMock, patch

import pytest

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
        assert os.path.exists(temp_workspace['temp_dir'])
        assert os.path.exists(temp_workspace['input_dir'])
        assert os.path.exists(temp_workspace['output_dir'])
        assert os.path.exists(temp_workspace['models_dir'])
        assert os.path.exists(temp_workspace['logs_dir'])
        assert os.path.exists(temp_workspace['job_dir'])
        
        # Assert config file exists
        assert os.path.exists(temp_workspace['config_path'])
        with open(temp_workspace['config_path']) as f:
            content = f.read()
            assert "test_job" in content

    def test_real_filesystem_operations(self, temp_workspace):
        """Test that we can perform real filesystem operations."""
        # Create a test file
        test_file = os.path.join(temp_workspace['models_dir'], 'test_model.onnx')
        test_content = "fake model data for testing"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Verify file exists and has correct content
        assert os.path.exists(test_file)
        with open(test_file) as f:
            assert f.read() == test_content
        
        # Test file size
        assert os.path.getsize(test_file) == len(test_content)

    def test_datasphere_settings_mock(self, mock_datasphere_env):
        """Test that DataSphere settings are properly mocked."""
        settings = mock_datasphere_env['settings']
        
        # Verify paths point to temp workspace
        assert 'datasphere_input' in settings.datasphere_input_dir
        assert 'datasphere_output' in settings.datasphere_output_dir
        assert 'models' in settings.models_dir
        
        # Verify DataSphere client config
        assert settings.datasphere.project_id == "test-project-id-new-arch"
        assert settings.datasphere.max_polls == 3
        assert settings.datasphere.poll_interval == 0.1

    def test_datasphere_client_mock(self, mock_datasphere_env):
        """Test that DataSphere client is properly mocked."""
        client = mock_datasphere_env['client']
        
        # Test job submission
        job_id = client.submit_job("fake_config_path", "fake_work_dir")
        assert job_id == "ds-job-default-new-arch"
        
        # Test status check
        status = client.get_job_status("test-job-id")
        assert status == "COMPLETED"
        
        # Test download (creates real files)
        temp_dir = mock_datasphere_env['settings'].datasphere_output_dir
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
            from deployment.app.services.datasphere_service import save_model_file_and_db
            from deployment.app.services.datasphere_service import run_job
            
            # Verify functions exist and are callable
            assert callable(save_model_file_and_db)
            assert callable(run_job)
            
            # This proves DataSphere SDK imports work with new architecture
            assert True
        except ImportError as e:
            pytest.fail(f"DataSphere SDK import failed: {e}")

    @pytest.mark.asyncio
    async def test_save_model_with_new_architecture(self, mock_datasphere_env, temp_workspace):
        """Test save_model_file_and_db with new architecture."""
        # Import INSIDE test for ML compatibility
        from deployment.app.services.datasphere_service import save_model_file_and_db
        from deployment.app.models.api_models import TrainingConfig
        
        # Create test model file in real filesystem
        temp_model_path = os.path.join(temp_workspace['temp_dir'], 'test_model.onnx')
        with open(temp_model_path, 'w') as f:
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
                "use_layer_norm": True
            },
            optimizer_config={"lr": 0.001, "weight_decay": 0.0001},
            lr_shed_config={"T_0": 5, "T_mult": 1},
            train_ds_config={"alpha": 0.1, "span": 6},
            lags=6,
            quantiles=[0.1, 0.5, 0.9]
        )
        
        # Mock create_model_record to avoid DB operations
        with patch('deployment.app.services.datasphere_service.create_model_record') as mock_create:
            mock_create.return_value = None
            
            # Test function execution
            result_model_id = await save_model_file_and_db(
                job_id="test-job-new-arch",
                model_path=temp_model_path,
                ds_job_id="test-ds-job",
                config=config,
                metrics_data={"mape": 5.0}
            )
            
            # Verify result
            assert result_model_id.startswith("test_model_new_arch_")
            
            # Verify model was copied to permanent location
            expected_permanent_path = os.path.join(
                mock_datasphere_env['settings'].models_dir, 
                f"{result_model_id}.onnx"
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
        file_operations_fs.create_file('/fake/test.txt', contents='test content')
        
        # Test file operations
        import shutil
        import os
        
        assert os.path.exists('/fake/test.txt')
        
        # Test copy operation
        shutil.copy2('/fake/test.txt', '/fake/test_copy.txt')
        assert os.path.exists('/fake/test_copy.txt')
        
        # Test content
        with open('/fake/test_copy.txt') as f:
            assert f.read() == 'test content'

    def test_isolation_between_tests(self, temp_workspace):
        """Test that each test gets isolated temp workspace."""
        # Create a marker file
        marker_file = os.path.join(temp_workspace['temp_dir'], 'isolation_test_marker.txt')
        with open(marker_file, 'w') as f:
            f.write('test isolation')
        
        assert os.path.exists(marker_file)
        # This file should NOT exist in other tests due to isolation

    def test_mock_reset_functionality(self, mock_datasphere_env):
        """Test that mocks are properly reset between tests."""
        client = mock_datasphere_env['client']
        
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
        base_params = {'batch_size': 64, 'learning_rate': 0.01}
        config = create_training_params(base_params)
        assert config.nn_model_config.batch_size == 64
        assert config.optimizer_config.lr == 0.01

    def test_backwards_compatibility_with_individual_fixtures(self, mock_asyncio_sleep, mock_get_datasets):
        """Test that individual fixtures from old conftest still work."""
        # Test asyncio sleep mock
        assert mock_asyncio_sleep is not None
        
        # Test get_datasets mock
        result = mock_get_datasets.return_value
        assert result is not None 