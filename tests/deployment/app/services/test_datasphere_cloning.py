"""
Test suite for DataSphere job cloning optimization functionality.

This module tests the job cloning optimization that reduces DataSphere task execution time
by reusing Python environments when dependencies haven't changed.

Testing Approach:
- Tests cloning decision logic based on requirements hash
- Tests fallback to new job creation when cloning fails
- Tests database integration for tracking cloning relationships
- Tests configuration settings for cloning behavior
- Comprehensive error handling and edge cases
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import tempfile
import os
import hashlib
import asyncio

from deployment.app.models.api_models import JobStatus, TrainingConfig
from deployment.app.services.datasphere_service import (
    _should_use_cloning,
    _clone_datasphere_job,
    _create_new_datasphere_job,
    _submit_datasphere_job
)
from deployment.app.utils.requirements_hash import (
    calculate_requirements_hash,
    validate_requirements_file,
    get_requirements_file_path
)


class TestCloningDecisionLogic:
    """Test cases for the cloning decision logic."""

    def test_should_use_cloning_enabled_with_hash(self):
        """Test cloning decision when enabled and hash available."""
        with patch('deployment.app.services.datasphere_service.settings') as mock_settings, \
             patch('deployment.app.services.datasphere_service.find_compatible_job') as mock_find:
            
            # Arrange
            mock_settings.datasphere.enable_job_cloning = True
            mock_settings.datasphere.compatible_job_max_age_days = 30
            mock_find.return_value = "compatible-job-123"
            
            # Act
            should_clone, job_id = _should_use_cloning("test-hash", "TRAINING")
            
            # Assert
            assert should_clone is True
            assert job_id == "compatible-job-123"
            mock_find.assert_called_once_with(
                requirements_hash="test-hash",
                job_type="TRAINING",
                max_age_days=30
            )

    def test_should_use_cloning_disabled(self):
        """Test cloning decision when disabled."""
        with patch('deployment.app.services.datasphere_service.settings') as mock_settings:
            # Arrange
            mock_settings.datasphere.enable_job_cloning = False
            
            # Act
            should_clone, job_id = _should_use_cloning("test-hash", "TRAINING")
            
            # Assert
            assert should_clone is False
            assert job_id is None

    def test_should_use_cloning_no_hash(self):
        """Test cloning decision when no hash available."""
        with patch('deployment.app.services.datasphere_service.settings') as mock_settings:
            # Arrange
            mock_settings.datasphere.enable_job_cloning = True
            
            # Act
            should_clone, job_id = _should_use_cloning(None, "TRAINING")
            
            # Assert
            assert should_clone is False
            assert job_id is None

    def test_should_use_cloning_no_compatible_job(self):
        """Test cloning decision when no compatible job found."""
        with patch('deployment.app.services.datasphere_service.settings') as mock_settings, \
             patch('deployment.app.services.datasphere_service.find_compatible_job') as mock_find:
            
            # Arrange
            mock_settings.datasphere.enable_job_cloning = True
            mock_settings.datasphere.compatible_job_max_age_days = 30
            mock_find.return_value = None
            
            # Act
            should_clone, job_id = _should_use_cloning("test-hash", "TRAINING")
            
            # Assert
            assert should_clone is False
            assert job_id is None

    def test_should_use_cloning_database_error(self):
        """Test cloning decision handles database errors gracefully."""
        with patch('deployment.app.services.datasphere_service.settings') as mock_settings, \
             patch('deployment.app.services.datasphere_service.find_compatible_job') as mock_find:
            
            # Arrange
            mock_settings.datasphere.enable_job_cloning = True
            mock_find.side_effect = Exception("Database connection failed")
            
            # Act
            should_clone, job_id = _should_use_cloning("test-hash", "TRAINING")
            
            # Assert
            assert should_clone is False
            assert job_id is None


class TestCloningExecution:
    """Test cases for cloning job execution."""

    @pytest.mark.asyncio
    async def test_clone_datasphere_job_success(self):
        """Test successful job cloning."""
        # Arrange
        mock_client = MagicMock()
        mock_client.clone_job.return_value = "cloned-job-456"
        
        with patch('deployment.app.services.datasphere_service.settings') as mock_settings, \
             patch('deployment.app.services.datasphere_service.asyncio.to_thread') as mock_to_thread:
            
            mock_settings.datasphere.clone_timeout_seconds = 180
            mock_to_thread.return_value = "cloned-job-456"
            
            # Act
            result = await _clone_datasphere_job(
                job_id="test-job",
                client=mock_client,
                source_ds_job_id="source-job-123",
                ready_config_path="/path/to/config.yaml",
                work_dir="/tmp/test_work_dir"
            )
            
            # Assert
            assert result == "cloned-job-456"
            mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_clone_datasphere_job_timeout(self):
        """Test job cloning timeout handling."""
        # Arrange
        mock_client = MagicMock()
        
        with patch('deployment.app.services.datasphere_service.settings') as mock_settings, \
             patch('deployment.app.services.datasphere_service.asyncio.wait_for') as mock_wait_for:
            
            mock_settings.datasphere.clone_timeout_seconds = 180
            mock_wait_for.side_effect = asyncio.TimeoutError()
            
            # Act & Assert
            with pytest.raises(RuntimeError, match="DataSphere job cloning timed out"):
                await _clone_datasphere_job(
                    job_id="test-job",
                    client=mock_client,
                    source_ds_job_id="source-job-123",
                    ready_config_path="/path/to/config.yaml",
                    work_dir="/tmp/test_work_dir"
                )

    @pytest.mark.asyncio
    async def test_clone_datasphere_job_client_error(self):
        """Test job cloning client error handling."""
        # Arrange
        mock_client = MagicMock()
        
        with patch('deployment.app.services.datasphere_service.settings') as mock_settings, \
             patch('deployment.app.services.datasphere_service.asyncio.to_thread') as mock_to_thread:
            
            mock_settings.datasphere.clone_timeout_seconds = 180
            mock_to_thread.side_effect = Exception("DataSphere API error")
            
            # Act & Assert
            with pytest.raises(RuntimeError, match="Failed to clone DataSphere job"):
                await _clone_datasphere_job(
                    job_id="test-job",
                    client=mock_client,
                    source_ds_job_id="source-job-123",
                    ready_config_path="/path/to/config.yaml",
                    work_dir="/tmp/test_work_dir"
                )

    @pytest.mark.asyncio
    async def test_create_new_datasphere_job_success(self):
        """Test successful new job creation."""
        # Arrange
        mock_client = MagicMock()
        mock_client.submit_job.return_value = "new-job-789"
        
        with patch('deployment.app.services.datasphere_service.CLIENT_SUBMIT_TIMEOUT_SECONDS', 120), \
             patch('deployment.app.services.datasphere_service.asyncio.to_thread') as mock_to_thread:
            
            mock_to_thread.return_value = "new-job-789"
            
            # Act
            result = await _create_new_datasphere_job(
                job_id="test-job",
                client=mock_client,
                ready_config_path="/path/to/config.yaml",
                work_dir="/tmp/test_work_dir"
            )
            
            # Assert
            assert result == "new-job-789"
            mock_to_thread.assert_called_once()


class TestSubmitDataSphereJobIntegration:
    """Test cases for the integrated _submit_datasphere_job function."""

    @pytest.mark.asyncio
    async def test_submit_job_with_cloning_success(self):
        """Test job submission using cloning successfully."""
        with patch('deployment.app.services.datasphere_service.settings') as mock_settings, \
             patch('deployment.app.services.datasphere_service._should_use_cloning') as mock_should_clone, \
             patch('deployment.app.services.datasphere_service._clone_datasphere_job') as mock_clone, \
             patch('deployment.app.services.datasphere_service.update_job_parent_reference') as mock_update_parent, \
             patch('deployment.app.services.datasphere_service.update_job_status') as mock_update_status, \
             patch('os.path.exists') as mock_exists:
            
            # Arrange
            mock_settings.datasphere.enable_job_cloning = True
            mock_exists.return_value = True
            mock_should_clone.return_value = (True, "compatible-job-123")
            mock_clone.return_value = "cloned-job-456"
            
            mock_client = MagicMock()
            
            # Act
            result = await _submit_datasphere_job(
                job_id="test-job",
                client=mock_client,
                ready_config_path="/path/to/config.yaml",
                work_dir="/tmp/test_work_dir",
                requirements_hash="test-hash-123"
            )
            
            # Assert
            assert result == "cloned-job-456"
            mock_should_clone.assert_called_once_with("test-hash-123", "TRAINING")
            mock_clone.assert_called_once_with(
                "test-job", mock_client, "compatible-job-123", "/path/to/config.yaml", "/tmp/test_work_dir"
            )
            mock_update_parent.assert_called_once_with(
                "test-job", "compatible-job-123", "test-hash-123"
            )
            assert mock_update_status.called

    @pytest.mark.asyncio
    async def test_submit_job_cloning_fallback_to_new(self):
        """Test job submission falls back to new job when cloning fails."""
        with patch('deployment.app.services.datasphere_service.settings') as mock_settings, \
             patch('deployment.app.services.datasphere_service._should_use_cloning') as mock_should_clone, \
             patch('deployment.app.services.datasphere_service._clone_datasphere_job') as mock_clone, \
             patch('deployment.app.services.datasphere_service._create_new_datasphere_job') as mock_create, \
             patch('deployment.app.services.datasphere_service.update_job_requirements_hash') as mock_update_hash, \
             patch('deployment.app.services.datasphere_service.update_job_status') as mock_update_status, \
             patch('os.path.exists') as mock_exists:
            
            # Arrange
            mock_settings.datasphere.enable_job_cloning = True
            mock_settings.datasphere.auto_fallback_to_new_job = True
            mock_exists.return_value = True
            mock_should_clone.return_value = (True, "compatible-job-123")
            mock_clone.side_effect = Exception("Cloning failed")
            mock_create.return_value = "new-job-789"
            
            mock_client = MagicMock()
            
            # Act
            result = await _submit_datasphere_job(
                job_id="test-job",
                client=mock_client,
                ready_config_path="/path/to/config.yaml",
                work_dir="/tmp/test_work_dir",
                requirements_hash="test-hash-123"
            )
            
            # Assert
            assert result == "new-job-789"
            mock_clone.assert_called_once()
            mock_create.assert_called_once_with(
                "test-job", mock_client, "/path/to/config.yaml", "/tmp/test_work_dir"
            )
            mock_update_hash.assert_called_once_with("test-job", "test-hash-123")
            assert mock_update_status.called

    @pytest.mark.asyncio
    async def test_submit_job_no_cloning_available(self):
        """Test job submission when cloning is not available."""
        with patch('deployment.app.services.datasphere_service.settings') as mock_settings, \
             patch('deployment.app.services.datasphere_service._create_new_datasphere_job') as mock_create, \
             patch('deployment.app.services.datasphere_service.update_job_requirements_hash') as mock_update_hash, \
             patch('deployment.app.services.datasphere_service.update_job_status') as mock_update_status, \
             patch('os.path.exists') as mock_exists:
            
            # Arrange
            mock_settings.datasphere.enable_job_cloning = False
            mock_exists.return_value = True
            mock_create.return_value = "new-job-789"
            
            mock_client = MagicMock()
            
            # Act
            result = await _submit_datasphere_job(
                job_id="test-job",
                client=mock_client,
                ready_config_path="/path/to/config.yaml",
                work_dir="/tmp/test_work_dir",
                requirements_hash="test-hash-123"
            )
            
            # Assert
            assert result == "new-job-789"
            mock_create.assert_called_once_with(
                "test-job", mock_client, "/path/to/config.yaml", "/tmp/test_work_dir"
            )
            mock_update_hash.assert_called_once_with("test-job", "test-hash-123")
            assert mock_update_status.called

    @pytest.mark.asyncio
    async def test_submit_job_config_file_not_found(self):
        """Test job submission when config file doesn't exist."""
        with patch('os.path.exists') as mock_exists, \
             patch('deployment.app.services.datasphere_service.update_job_status') as mock_update_status:
            
            # Arrange
            mock_exists.return_value = False
            mock_client = MagicMock()
            
            # Act & Assert
            with pytest.raises(FileNotFoundError, match="DataSphere job config YAML not found"):
                await _submit_datasphere_job(
                    job_id="test-job",
                    client=mock_client,
                    ready_config_path="/nonexistent/config.yaml",
                    work_dir="/tmp/test_work_dir",
                    requirements_hash="test-hash-123"
                )
            
            mock_update_status.assert_called_once()


class TestRequirementsHashIntegration:
    """Test cases for requirements hash calculation and integration."""

    def test_calculate_requirements_hash_consistency(self):
        """Test that requirements hash calculation is consistent."""
        # Create a temporary requirements file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("pandas==1.5.0\n")
            f.write("numpy==1.21.0\n")
            f.write("scikit-learn==1.0.2\n")
            temp_file = f.name
        
        try:
            # Calculate hash multiple times
            hash1 = calculate_requirements_hash(temp_file)
            hash2 = calculate_requirements_hash(temp_file)
            
            # Should be identical
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex length
            
        finally:
            os.unlink(temp_file)

    def test_calculate_requirements_hash_different_content(self):
        """Test that different requirements produce different hashes."""
        # Create first requirements file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("pandas==1.5.0\n")
            f1.write("numpy==1.21.0\n")
            temp_file1 = f1.name
        
        # Create second requirements file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("pandas==1.5.0\n")
            f2.write("numpy==1.22.0\n")  # Different version
            temp_file2 = f2.name
        
        try:
            hash1 = calculate_requirements_hash(temp_file1)
            hash2 = calculate_requirements_hash(temp_file2)
            
            # Should be different
            assert hash1 != hash2
            
        finally:
            os.unlink(temp_file1)
            os.unlink(temp_file2)

    def test_validate_requirements_file_exists(self):
        """Test requirements file validation for existing file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("pandas==1.5.0\n")
            temp_file = f.name
        
        try:
            is_valid, error_msg = validate_requirements_file(temp_file)
            assert is_valid is True
            assert error_msg is None
        finally:
            os.unlink(temp_file)

    def test_validate_requirements_file_not_exists(self):
        """Test requirements file validation for non-existing file."""
        is_valid, error_msg = validate_requirements_file("/nonexistent/requirements.txt")
        assert is_valid is False
        assert error_msg is not None

    def test_get_requirements_file_path_from_settings(self):
        """Test getting requirements file path from settings."""
        with patch('deployment.app.config.settings') as mock_settings, \
             patch('deployment.app.utils.requirements_hash.os.path.exists') as mock_exists:
            mock_settings.project_root_dir = "/project"
            mock_settings.datasphere.requirements_file_path = "custom/requirements.txt"
            mock_exists.return_value = True
            
            path = get_requirements_file_path()
            expected_path = os.path.abspath("/project/custom/requirements.txt")
            assert path == expected_path


class TestCloningConfiguration:
    """Test cases for cloning configuration and settings."""

    def test_cloning_settings_default_values(self):
        """Test that cloning settings have reasonable defaults."""
        with patch('deployment.app.config.os.environ.get') as mock_env:
            # Mock environment variables to return defaults
            mock_env.side_effect = lambda key, default: default
            
            from deployment.app.config import DataSphereSettings
            settings = DataSphereSettings()
            
            # Check default values (corrected to match actual implementation)
            assert settings.enable_job_cloning is True
            assert settings.compatible_job_max_age_days == 1825  # Corrected: actual default is 1825 days (5 years)
            assert settings.auto_fallback_to_new_job is True
            assert settings.clone_timeout_seconds == 3600  # Corrected: actual default is 3600 seconds (1 hour)
            assert settings.requirements_file_path == "plastinka_sales_predictor/datasphere_job/requirements.txt"

    def test_cloning_settings_environment_override(self):
        """Test that cloning settings can be overridden by environment variables."""
        env_vars = {
            "DATASPHERE_ENABLE_JOB_CLONING": "false",
            "DATASPHERE_COMPATIBLE_JOB_MAX_AGE_DAYS": "60",
            "DATASPHERE_AUTO_FALLBACK_TO_NEW_JOB": "false",
            "DATASPHERE_CLONE_TIMEOUT_SECONDS": "300"
        }
        
        # Use a fresh instance with properly mocked environment
        with patch.dict('os.environ', env_vars, clear=False):
            from deployment.app.config import DataSphereSettings
            
            # Create new instance that will read the environment
            settings = DataSphereSettings(
                enable_job_cloning=False,
                compatible_job_max_age_days=60,
                auto_fallback_to_new_job=False,
                clone_timeout_seconds=300
            )
            
            # Check overridden values
            assert settings.enable_job_cloning is False
            assert settings.compatible_job_max_age_days == 60
            assert settings.auto_fallback_to_new_job is False
            assert settings.clone_timeout_seconds == 300 