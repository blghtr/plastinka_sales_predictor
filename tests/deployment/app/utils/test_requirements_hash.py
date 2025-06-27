"""
Tests for requirements hash utilities.

Tests the functionality for calculating and managing requirements.txt hashes
for DataSphere job cloning optimization.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from deployment.app.utils.requirements_hash import (
    calculate_requirements_hash,
    get_requirements_file_path,
    requirements_changed_since_job,
    get_requirements_hash_for_current_state,
    validate_requirements_file
)


class TestCalculateRequirementsHash:
    """Test requirements hash calculation."""
    
    def test_calculate_hash_success(self):
        """Test successful hash calculation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("numpy==1.21.0\npandas==1.3.0\n")
            tmp_file.flush()
            
            # Close file explicitly for Windows compatibility
            tmp_file.close()
            
            try:
                hash_result = calculate_requirements_hash(tmp_file.name)
                
                # Should return a 64-character hex string (SHA256)
                assert isinstance(hash_result, str)
                assert len(hash_result) == 64
                assert all(c in '0123456789abcdef' for c in hash_result.lower())
            finally:
                os.unlink(tmp_file.name)
    
    def test_calculate_hash_consistent(self):
        """Test that hash calculation is consistent."""
        content = "flask==2.0.1\nrequests==2.25.1\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file1:
            tmp_file1.write(content)
            tmp_file1.flush()
            tmp_file1.close()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file2:
                tmp_file2.write(content)
                tmp_file2.flush()
                tmp_file2.close()
                
                try:
                    hash1 = calculate_requirements_hash(tmp_file1.name)
                    hash2 = calculate_requirements_hash(tmp_file2.name)
                    
                    assert hash1 == hash2, "Same content should produce same hash"
                finally:
                    os.unlink(tmp_file1.name)
                    os.unlink(tmp_file2.name)
    
    def test_calculate_hash_different_content(self):
        """Test that different content produces different hashes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file1:
            tmp_file1.write("numpy==1.21.0\n")
            tmp_file1.flush()
            tmp_file1.close()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file2:
                tmp_file2.write("numpy==1.22.0\n")  # Different version
                tmp_file2.flush()
                tmp_file2.close()
                
                try:
                    hash1 = calculate_requirements_hash(tmp_file1.name)
                    hash2 = calculate_requirements_hash(tmp_file2.name)
                    
                    assert hash1 != hash2, "Different content should produce different hashes"
                finally:
                    os.unlink(tmp_file1.name)
                    os.unlink(tmp_file2.name)
    
    def test_calculate_hash_file_not_found(self):
        """Test error handling when file doesn't exist."""
        non_existent_path = "/path/that/does/not/exist/requirements.txt"
        
        with pytest.raises(FileNotFoundError):
            calculate_requirements_hash(non_existent_path)
    
    def test_calculate_hash_directory_path(self):
        """Test error handling when path is a directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(OSError, match="Path is not a file"):
                calculate_requirements_hash(tmp_dir)
    
    def test_calculate_hash_empty_file(self):
        """Test hash calculation for empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            # Write nothing (empty file)
            tmp_file.flush()
            tmp_file.close()
            
            try:
                hash_result = calculate_requirements_hash(tmp_file.name)
                
                # Should still return a valid hash
                assert isinstance(hash_result, str)
                assert len(hash_result) == 64
            finally:
                os.unlink(tmp_file.name)


class TestGetRequirementsFilePath:
    """Test requirements file path resolution."""
    
    @patch('deployment.app.config.settings')
    def test_get_path_absolute_path(self, mock_settings):
        """Test getting absolute path from settings."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.close()  # Close file explicitly before using it
            try:
                mock_settings.datasphere.requirements_file_path = tmp_file.name
                
                result_path = get_requirements_file_path()
                assert result_path == os.path.abspath(tmp_file.name)
            finally:
                os.unlink(tmp_file.name)
    
    @patch('deployment.app.config.settings')
    def test_get_path_relative_path(self, mock_settings):
        """Test getting relative path from settings."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix='.txt', delete=False) as tmp_file:
                tmp_file.close()  # Close file explicitly before using it
                try:
                    rel_path = os.path.basename(tmp_file.name)
                    mock_settings.datasphere.requirements_file_path = rel_path
                    mock_settings.project_root_dir = tmp_dir
                    
                    result_path = get_requirements_file_path()
                    expected_path = os.path.abspath(os.path.join(tmp_dir, rel_path))
                    assert result_path == expected_path
                finally:
                    os.unlink(tmp_file.name)
    
    @patch('deployment.app.config.settings')
    def test_get_path_file_not_found(self, mock_settings):
        """Test error when configured file doesn't exist."""
        mock_settings.datasphere.requirements_file_path = "/path/that/does/not/exist.txt"
        
        with pytest.raises(OSError):
            get_requirements_file_path()
    
    @patch('deployment.app.config.settings')
    def test_get_path_settings_error(self, mock_settings):
        """Test error when settings configuration is missing."""
        # Mock settings to raise AttributeError
        mock_settings.datasphere = MagicMock()
        del mock_settings.datasphere.requirements_file_path
        
        with pytest.raises(OSError, match="Failed to get requirements file path from settings"):
            get_requirements_file_path()


class TestRequirementsChangedSinceJob:
    """Test requirements change detection."""
    
    @patch('deployment.app.db.database.get_job_requirements_hash')
    def test_requirements_unchanged(self, mock_get_hash):
        """Test when requirements haven't changed."""
        test_hash = "abc123def456"
        mock_get_hash.return_value = test_hash
        
        result = requirements_changed_since_job("job123", test_hash)
        assert result is False
    
    @patch('deployment.app.db.database.get_job_requirements_hash')
    def test_requirements_changed(self, mock_get_hash):
        """Test when requirements have changed."""
        old_hash = "abc123def456"
        new_hash = "def456ghi789"
        mock_get_hash.return_value = old_hash
        
        result = requirements_changed_since_job("job123", new_hash)
        assert result is True
    
    @patch('deployment.app.db.database.get_job_requirements_hash')
    def test_requirements_job_hash_not_found(self, mock_get_hash):
        """Test when job hash is not found."""
        mock_get_hash.return_value = None
        
        result = requirements_changed_since_job("job123", "any_hash")
        assert result is True  # Should assume changed
    
    def test_requirements_empty_job_id(self):
        """Test error handling for empty job ID."""
        with pytest.raises(ValueError, match="job_id cannot be empty"):
            requirements_changed_since_job("", "some_hash")
    
    def test_requirements_empty_hash(self):
        """Test error handling for empty hash."""
        with pytest.raises(ValueError, match="current_hash cannot be empty"):
            requirements_changed_since_job("job123", "")
    
    @patch('deployment.app.db.database.get_job_requirements_hash')
    def test_requirements_database_error(self, mock_get_hash):
        """Test error handling when database access fails."""
        mock_get_hash.side_effect = Exception("Database error")
        
        result = requirements_changed_since_job("job123", "some_hash")
        assert result is True  # Should assume changed on error


class TestGetRequirementsHashForCurrentState:
    """Test getting hash for current state."""
    
    @patch('deployment.app.utils.requirements_hash.get_requirements_file_path')
    @patch('deployment.app.utils.requirements_hash.calculate_requirements_hash')
    def test_get_current_hash_success(self, mock_calc_hash, mock_get_path):
        """Test successful current hash calculation."""
        mock_get_path.return_value = "/path/to/requirements.txt"
        mock_calc_hash.return_value = "abc123def456"
        
        result = get_requirements_hash_for_current_state()
        assert result == "abc123def456"
        
        mock_get_path.assert_called_once()
        mock_calc_hash.assert_called_once_with("/path/to/requirements.txt")
    
    @patch('deployment.app.utils.requirements_hash.get_requirements_file_path')
    def test_get_current_hash_error(self, mock_get_path):
        """Test error handling when current hash calculation fails."""
        mock_get_path.side_effect = FileNotFoundError("File not found")
        
        result = get_requirements_hash_for_current_state()
        assert result is None


class TestValidateRequirementsFile:
    """Test requirements file validation."""
    
    def test_validate_success(self):
        """Test successful validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("numpy==1.21.0\n")
            tmp_file.flush()
            tmp_file.close()
            
            try:
                is_valid, error_msg = validate_requirements_file(tmp_file.name)
                assert is_valid is True
                assert error_msg is None
            finally:
                os.unlink(tmp_file.name)
    
    def test_validate_empty_path(self):
        """Test validation with empty path."""
        is_valid, error_msg = validate_requirements_file("")
        assert is_valid is False
        assert "empty" in error_msg.lower()
    
    def test_validate_file_not_found(self):
        """Test validation when file doesn't exist."""
        non_existent_path = "/path/that/does/not/exist.txt"
        
        is_valid, error_msg = validate_requirements_file(non_existent_path)
        assert is_valid is False
        assert "does not exist" in error_msg
    
    def test_validate_directory_path(self):
        """Test validation when path is a directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            is_valid, error_msg = validate_requirements_file(tmp_dir)
            assert is_valid is False
            assert "not a file" in error_msg
    
    def test_validate_empty_file(self):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            # Write nothing (empty file)
            tmp_file.flush()
            tmp_file.close()
            
            try:
                is_valid, error_msg = validate_requirements_file(tmp_file.name)
                assert is_valid is True  # Empty file should be valid
                assert error_msg is None
            finally:
                os.unlink(tmp_file.name) 