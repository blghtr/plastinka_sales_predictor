import pytest
import os
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime

from deployment.app.services.datasphere_service import save_model_file_and_db

@pytest.fixture
def mock_model_file(tmp_path):
    """Create a temporary model file for testing."""
    model_path = tmp_path / "model.onnx"
    
    # Create a dummy model file
    with open(model_path, "wb") as f:
        f.write(b"dummy model content")
        
    return str(model_path)

@pytest.mark.asyncio
async def test_save_model_file_and_db_success(mock_model_file):
    """Test successful model saving."""
    job_id = "test_job_123"
    ds_job_id = "ds_job_456"
    
    # Mock create_model_record to avoid actual database operations
    with patch("deployment.app.services.datasphere_service.create_model_record") as mock_create_record:
        # Run the function
        model_id = await save_model_file_and_db(job_id, mock_model_file, ds_job_id)
        
        # Verify model ID format
        assert model_id.startswith("model_")
        assert len(model_id) > 6  # Should have unique suffix
        
        # Verify create_model_record was called with correct args
        mock_create_record.assert_called_once()
        args = mock_create_record.call_args.kwargs
        assert args["job_id"] == job_id
        assert args["model_path"] == mock_model_file
        assert args["is_active"] is False
        
        # Verify metadata
        assert "file_size_bytes" in args["metadata"]
        assert args["metadata"]["downloaded_from_ds_job"] == ds_job_id
        assert args["metadata"]["original_path"] == mock_model_file

@pytest.mark.asyncio
async def test_save_model_file_and_db_missing_file():
    """Test behavior when model file is missing."""
    job_id = "test_job_123"
    ds_job_id = "ds_job_456"
    non_existent_path = "/path/to/nonexistent/model.onnx"
    
    # Mock create_model_record to avoid actual database operations
    with patch("deployment.app.services.datasphere_service.create_model_record") as mock_create_record:
        with pytest.raises(FileNotFoundError):
            await save_model_file_and_db(job_id, non_existent_path, ds_job_id)
        
        # Verify create_model_record was not called
        mock_create_record.assert_not_called()

@pytest.mark.asyncio
async def test_save_model_file_and_db_generates_unique_ids():
    """Test that each call generates a unique model ID."""
    with patch("os.path.exists", return_value=True):
        with patch("os.path.getsize", return_value=1000):
            with patch("deployment.app.services.datasphere_service.create_model_record"):
                # Call the function twice
                model_id_1 = await save_model_file_and_db("job1", "model1.onnx", "ds_job1")
                model_id_2 = await save_model_file_and_db("job1", "model1.onnx", "ds_job1")
                
                # Verify the IDs are different
                assert model_id_1 != model_id_2
                assert model_id_1.startswith("model_")
                assert model_id_2.startswith("model_")

@pytest.mark.asyncio
async def test_save_model_file_and_db_handles_db_error(mock_model_file):
    """Test behavior when database operation fails."""
    job_id = "test_job_123"
    ds_job_id = "ds_job_456"
    
    # Mock create_model_record to raise an exception
    with patch("deployment.app.services.datasphere_service.create_model_record", side_effect=Exception("DB Error")):
        with pytest.raises(Exception, match="DB Error"):
            await save_model_file_and_db(job_id, mock_model_file, ds_job_id)

@pytest.mark.asyncio
async def test_save_model_file_and_db_with_model_path_verification(mock_model_file):
    """Test that the function verifies model path correctly."""
    job_id = "test_job_123"
    ds_job_id = "ds_job_456"
    
    # Mock os.path.exists to verify specific path checks
    original_exists = os.path.exists
    
    def mock_exists(path):
        # Only return True for the mock_model_file
        if path == mock_model_file:
            return True
        return original_exists(path)
    
    with patch("os.path.exists", side_effect=mock_exists):
        with patch("deployment.app.services.datasphere_service.create_model_record") as mock_create_record:
            # Should succeed because mock_model_file exists
            model_id = await save_model_file_and_db(job_id, mock_model_file, ds_job_id)
            assert model_id is not None
            
            # Verify create_model_record was called
            mock_create_record.assert_called_once()
            
            # Try with a different path that doesn't exist
            with pytest.raises(FileNotFoundError):
                await save_model_file_and_db(job_id, mock_model_file + "_nonexistent", ds_job_id) 