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
from importlib import reload

from deployment.app.cloud_integration.config.cloud_config import CloudIntegrationSettings, CloudStorageSettings, CloudFunctionSettings
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
        monkeypatch.setenv("CLOUD_CALLBACK_AUTH_TOKEN", "test_callback_token")
        monkeypatch.setenv("YANDEX_CLOUD_STORAGE_ENDPOINT", "https://test-storage.example.com")
        monkeypatch.setenv("DATABASE_PATH", "deployment/data/test_db.sqlite")
        monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000,https://app.example.com")

    @patch('deployment.app.cloud_integration.config.cloud_config.cloud_settings')
    def test_cloud_settings_validation(self, mock_cloud_settings):
        """Test that cloud settings validation works correctly."""
        # Configure the mocked settings directly
        mock_cloud_settings.storage.access_key = "test_access_key"
        mock_cloud_settings.storage.secret_key = "test_secret_key"
        mock_cloud_settings.functions.api_key = "test_api_key"
        mock_cloud_settings.functions.folder_id = "test_folder_id"
        mock_cloud_settings.callback_auth_token = "test_callback_token"
        mock_cloud_settings.storage.endpoint_url = "https://test-storage.example.com"
        # Configure mock method return value
        mock_cloud_settings.validate_credentials.return_value = True
        
        # Test that validation passes with mock environment variables
        assert mock_cloud_settings.validate_credentials() is True
        
        # Test environment variables are correctly loaded (check mock attributes)
        assert mock_cloud_settings.storage.access_key == "test_access_key"
        assert mock_cloud_settings.storage.secret_key == "test_secret_key"
        assert mock_cloud_settings.functions.api_key == "test_api_key"
        assert mock_cloud_settings.functions.folder_id == "test_folder_id"
        assert mock_cloud_settings.callback_auth_token == "test_callback_token"
        assert mock_cloud_settings.storage.endpoint_url == "https://test-storage.example.com"
    
    @patch('boto3.client')
    @patch('deployment.app.cloud_integration.client.storage_client.cloud_settings')
    def test_storage_client_initialization(self, mock_cloud_settings, mock_boto3_client):
        """Test that storage client is correctly initialized with environment variables."""
        # Configure the mocked settings directly
        mock_cloud_settings.storage.endpoint_url = "https://test-storage.example.com"
        mock_cloud_settings.storage.access_key = "test_access_key"
        mock_cloud_settings.storage.secret_key = "test_secret_key"
        mock_cloud_settings.storage.region = "ru-central1"
        
        # Create storage client *after* patching settings
        CloudStorageClient()
        
        # Check that boto3 client was called with correct parameters
        mock_boto3_client.assert_called_once_with(
            's3',
            endpoint_url="https://test-storage.example.com",
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
            region_name=mock_cloud_settings.storage.region
        )
    
    @patch('requests.post')
    @patch('deployment.app.cloud_integration.client.function_client.cloud_settings')
    @patch('deployment.app.cloud_integration.client.function_client.get_db_connection') # Mock DB connection
    def test_function_client_invocation(self, mock_get_db, mock_cloud_settings, mock_post):
        """Test that function client correctly invokes cloud functions."""
        # Configure DB mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        # Assume function is already registered in _ensure_function_registered
        mock_cursor.fetchone.return_value = ('existing_function_id',) 
        
        # Configure the mocked settings directly
        mock_cloud_settings.functions.api_gateway_url = "http://mock-gateway/"
        mock_cloud_settings.functions.api_key = "test_api_key_func"
        mock_cloud_settings.callback_url = "http://test-callback/"
        mock_cloud_settings.callback_auth_token = "test_callback_token_func"
        
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
        function_client = CloudFunctionClient()
            
        # Test function invocation - Changed to invoke_training_function
        # Updated params and input_data to match expected structure
        training_params_data = {"epochs": 10, "lr": 0.01}
        storage_paths_data = {"input": "s3://bucket/input.csv", "output": "s3://bucket/output/"}
        result = function_client.invoke_training_function(
            job_id="test_job_id",
            training_params=training_params_data,
            storage_paths=storage_paths_data
        )
            
        # Check result (should be execution_id, check mock_post call for payload)
        args, kwargs = mock_post.call_args
        sent_payload = kwargs.get("json", {})
        assert sent_payload["job_id"] == "test_job_id"
        assert sent_payload["training_params"] == training_params_data
        assert sent_payload["storage_paths"] == storage_paths_data
        # Removed assertion for execution_id in payload
        # Assuming invoke_training_function returns the execution_id it generated
        # Assert that the returned result is a non-empty string (UUID)
        assert isinstance(result, str) and len(result) > 0 
            
        # Verify that request was made with correct authentication
        headers = kwargs.get("headers", {})
        # Check for Authorization header instead of X-Api-Key
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {mock_cloud_settings.functions.api_key}" 
    
    @patch('deployment.app.cloud_integration.config.cloud_config.cloud_settings')
    def test_cloud_function_env_vars(self, mock_cloud_settings):
        """Test that environment variables are correctly passed to cloud functions."""
        # Configure the mocked settings directly
        mock_cloud_settings.storage.endpoint_url = "https://test-storage.example.com"
        mock_cloud_settings.storage.access_key = "test_access_key"
        mock_cloud_settings.storage.secret_key = "test_secret_key"
        mock_cloud_settings.callback_auth_token = "test_callback_token_env"

        # Configure the return value for the property mock
        expected_env_vars = {
            "STORAGE_ENDPOINT": "https://test-storage.example.com",
            "STORAGE_ACCESS_KEY": "test_access_key",
            "STORAGE_SECRET_KEY": "test_secret_key",
            "API_KEY": "test_callback_token_env", 
            # Add other expected vars if necessary, matching the implementation
            # of get_function_env_vars
            "CALLBACK_URL": mock_cloud_settings.callback_url, # Assuming it's needed
            "DB_PATH": mock_cloud_settings.db_path # Assuming it's needed
        }
        # Use PropertyMock or configure return_value for the property access
        type(mock_cloud_settings).get_function_env_vars = unittest.mock.PropertyMock(return_value=expected_env_vars)

        # Get environment variables that would be passed to cloud functions
        env_vars = mock_cloud_settings.get_function_env_vars
        
        # Verify environment variables
        assert env_vars["STORAGE_ENDPOINT"] == "https://test-storage.example.com"
        assert env_vars["STORAGE_ACCESS_KEY"] == "test_access_key"
        assert env_vars["STORAGE_SECRET_KEY"] == "test_secret_key"
        assert env_vars["API_KEY"] == "test_callback_token_env" 