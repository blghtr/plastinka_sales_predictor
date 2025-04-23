"""
Integration tests for cloud function integration.
Tests the interaction between the API and cloud functions.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import pytest
import json
from datetime import datetime

from deployment.app.cloud_integration.config.cloud_config import cloud_settings
from deployment.app.cloud_integration.client.function_client import CloudFunctionClient
from deployment.app.cloud_integration.client.storage_client import CloudStorageClient


class TestCloudIntegration:
    """Integration tests for cloud function integration."""

    @pytest.fixture
    def mock_env_variables(self, monkeypatch):
        """Set up environment variables for tests."""
        monkeypatch.setenv("YANDEX_CLOUD_ACCESS_KEY", "test_access_key")
        monkeypatch.setenv("YANDEX_CLOUD_SECRET_KEY", "test_secret_key")
        monkeypatch.setenv("YANDEX_CLOUD_API_KEY", "test_api_key")
        monkeypatch.setenv("YANDEX_CLOUD_FOLDER_ID", "test_folder_id")
        monkeypatch.setenv("FASTAPI_CLOUD_CALLBACK_AUTH_TOKEN", "test_callback_token")
        monkeypatch.setenv("YANDEX_CLOUD_STORAGE_ENDPOINT", "https://test-storage.example.com")
        monkeypatch.setenv("DATABASE_PATH", "deployment/data/test_db.sqlite")
        monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000,https://app.example.com")

    def test_cloud_settings_validation(self, mock_env_variables):
        """Test that cloud settings validation works correctly."""
        # Reload cloud settings to pick up environment variables
        from importlib import reload
        from deployment.app.cloud_integration.config import cloud_config
        reload(cloud_config)
        
        # Test that validation passes with mock environment variables
        assert cloud_config.cloud_settings.validate_credentials() is True
        
        # Test environment variables are correctly loaded
        assert cloud_config.cloud_settings.storage.access_key == "test_access_key"
        assert cloud_config.cloud_settings.storage.secret_key == "test_secret_key"
        assert cloud_config.cloud_settings.functions.api_key == "test_api_key"
        assert cloud_config.cloud_settings.functions.folder_id == "test_folder_id"
        assert cloud_config.cloud_settings.callback_auth_token == "test_callback_token"
        assert cloud_config.cloud_settings.storage.endpoint_url == "https://test-storage.example.com"
    
    @patch('boto3.client')
    def test_storage_client_initialization(self, mock_boto3_client, mock_env_variables):
        """Test that storage client is correctly initialized with environment variables."""
        # Create storage client
        CloudStorageClient()
        
        # Check that boto3 client was called with correct parameters
        mock_boto3_client.assert_called_once_with(
            's3',
            endpoint_url="https://test-storage.example.com",
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
            region_name=cloud_settings.storage.region
        )
    
    @patch('requests.post')
    def test_function_client_invocation(self, mock_post, mock_env_variables):
        """Test that function client correctly invokes cloud functions."""
        # Mock response from cloud function
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "job_id": "test_job_id",
            "status": "success",
            "result": {"key": "value"}
        }
        mock_post.return_value = mock_response
        
        # Set up function client with mocked dependencies
        with patch('deployment.app.cloud_integration.client.function_client.CloudStorageClient'):
            function_client = CloudFunctionClient()
            
            # Test function invocation
            result = function_client.invoke_function(
                function_name="training",
                job_id="test_job_id",
                params={"test_param": "value"},
                input_data="test_data"
            )
            
            # Check result
            assert result["status"] == "success"
            assert result["job_id"] == "test_job_id"
            assert result["result"]["key"] == "value"
            
            # Verify that request was made with correct authentication
            args, kwargs = mock_post.call_args
            headers = kwargs.get("headers", {})
            assert "X-Api-Key" in headers
            assert headers["X-Api-Key"] == "test_api_key"
    
    @patch('deployment.app.cloud_integration.client.function_client.CloudFunctionClient.invoke_function')
    @patch('deployment.app.cloud_integration.client.function_client.CloudStorageClient')
    def test_cloud_function_env_vars(self, mock_storage_client, mock_invoke, mock_env_variables):
        """Test that environment variables are correctly passed to cloud functions."""
        # Import cloud settings to test
        from deployment.app.cloud_integration.config.cloud_config import cloud_settings
        
        # Get environment variables that would be passed to cloud functions
        env_vars = cloud_settings.get_function_env_vars
        
        # Verify environment variables
        assert env_vars["STORAGE_ENDPOINT"] == "https://test-storage.example.com"
        assert env_vars["STORAGE_ACCESS_KEY"] == "test_access_key"
        assert env_vars["STORAGE_SECRET_KEY"] == "test_secret_key"
        assert env_vars["API_KEY"] == "test_callback_token" 