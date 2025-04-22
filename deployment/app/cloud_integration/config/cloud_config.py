"""
Configuration settings for cloud integration with Yandex Cloud.
Contains settings for cloud functions and storage.
"""
import os
from typing import Dict, Any
from pydantic import BaseSettings, Field


class CloudStorageSettings(BaseSettings):
    """Settings for cloud storage."""
    bucket_name: str = Field(
        default="plastinka-ml-data",
        description="Name of the cloud storage bucket for ML data"
    )
    endpoint_url: str = Field(
        default="https://storage.yandexcloud.net",
        description="Endpoint URL for the cloud storage service"
    )
    access_key: str = Field(
        default=os.environ.get("YANDEX_CLOUD_ACCESS_KEY", ""),
        description="Access key for cloud storage authentication"
    )
    secret_key: str = Field(
        default=os.environ.get("YANDEX_CLOUD_SECRET_KEY", ""),
        description="Secret key for cloud storage authentication"
    )
    region: str = Field(
        default="ru-central1",
        description="Cloud storage region"
    )
    dataset_prefix: str = Field(
        default="datasets/",
        description="Prefix for dataset objects in storage"
    )
    model_prefix: str = Field(
        default="models/",
        description="Prefix for model objects in storage"
    )
    result_prefix: str = Field(
        default="results/",
        description="Prefix for result objects in storage"
    )
    temp_prefix: str = Field(
        default="temp/",
        description="Prefix for temporary objects in storage"
    )
    presigned_url_expiration: int = Field(
        default=3600,
        description="Expiration time for presigned URLs in seconds"
    )


class CloudFunctionSettings(BaseSettings):
    """Settings for cloud functions."""
    function_service_account_id: str = Field(
        default=os.environ.get("YANDEX_CLOUD_SERVICE_ACCOUNT_ID", ""),
        description="Service account ID for cloud function authentication"
    )
    folder_id: str = Field(
        default=os.environ.get("YANDEX_CLOUD_FOLDER_ID", ""),
        description="Yandex Cloud folder ID where functions are deployed"
    )
    training_function_name: str = Field(
        default="plastinka-training-function",
        description="Name of the training function in Yandex Cloud"
    )
    prediction_function_name: str = Field(
        default="plastinka-prediction-function",
        description="Name of the prediction function in Yandex Cloud"
    )
    api_gateway_url: str = Field(
        default=os.environ.get("YANDEX_CLOUD_API_GATEWAY", ""),
        description="URL of the API Gateway for calling cloud functions"
    )
    request_timeout: int = Field(
        default=30,
        description="Timeout for function invocation requests in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed function calls"
    )
    retry_backoff_factor: float = Field(
        default=1.5,
        description="Exponential backoff factor for retries"
    )
    api_key: str = Field(
        default=os.environ.get("YANDEX_CLOUD_API_KEY", ""),
        description="API key for authentication with Yandex Cloud API"
    )


class CloudIntegrationSettings(BaseSettings):
    """Main settings container for cloud integration."""
    storage: CloudStorageSettings = CloudStorageSettings()
    functions: CloudFunctionSettings = CloudFunctionSettings()
    callback_base_url: str = Field(
        default=os.environ.get("FASTAPI_CALLBACK_BASE_URL", "http://localhost:8000"),
        description="Base URL for callbacks from cloud functions"
    )
    callback_route: str = Field(
        default="/api/v1/cloud/callback",
        description="Route for cloud function callbacks"
    )
    callback_auth_token: str = Field(
        default=os.environ.get("CLOUD_CALLBACK_AUTH_TOKEN", ""),
        description="Authentication token for cloud function callbacks"
    )
    max_upload_size: int = Field(
        default=50 * 1024 * 1024,  # 50 MB
        description="Maximum size for direct uploads in bytes"
    )
    
    @property
    def callback_url(self) -> str:
        """Get the full callback URL."""
        return f"{self.callback_base_url}{self.callback_route}"
    
    @property
    def get_function_env_vars(self) -> Dict[str, Any]:
        """Get environment variables for cloud functions."""
        return {
            "API_ENDPOINT": self.callback_base_url,
            "API_KEY": self.callback_auth_token,
            "STORAGE_BUCKET": self.storage.bucket_name,
            "STORAGE_ACCESS_KEY": self.storage.access_key,
            "STORAGE_SECRET_KEY": self.storage.secret_key,
            "LOG_LEVEL": "INFO"
        }


# Create a global instance of the settings
cloud_settings = CloudIntegrationSettings() 