"""
Configuration settings for cloud integration with Yandex Cloud.
Contains settings for cloud functions and storage.
"""
import os
import logging
from typing import Dict, Any
from pydantic import BaseSettings, Field, validator

# Import app configuration
from app.config import settings as app_settings

logger = logging.getLogger("plastinka.cloud_config")


class CloudStorageSettings(BaseSettings):
    """Settings for cloud storage."""
    bucket_name: str = Field(
        default="plastinka-ml-data",
        description="Name of the cloud storage bucket for ML data"
    )
    endpoint_url: str = Field(
        default=os.environ.get("YANDEX_CLOUD_STORAGE_ENDPOINT", "https://storage.yandexcloud.net"),
        description="Endpoint URL for the cloud storage service"
    )
    access_key: str = Field(
        default="",
        description="Access key for cloud storage authentication"
    )
    secret_key: str = Field(
        default="",
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
    
    @validator('access_key', 'secret_key', pre=True)
    def validate_credentials(cls, v, values, **kwargs):
        field_name = kwargs['field'].name
        env_var = f"YANDEX_CLOUD_{field_name.upper()}"
        value = os.environ.get(env_var, v)
        
        if not value:
            logger.warning(f"No {field_name} provided through environment variable {env_var}")
            
        return value


class CloudFunctionSettings(BaseSettings):
    """Settings for cloud functions."""
    function_service_account_id: str = Field(
        default="",
        description="Service account ID for cloud function authentication"
    )
    folder_id: str = Field(
        default="",
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
        default="",
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
        default="",
        description="API key for authentication with Yandex Cloud API"
    )
    
    @validator('function_service_account_id', 'folder_id', 'api_gateway_url', 'api_key', pre=True)
    def validate_cloud_settings(cls, v, values, **kwargs):
        field_name = kwargs['field'].name
        env_var = f"YANDEX_CLOUD_{field_name.upper()}"
        value = os.environ.get(env_var, v)
        
        if not value:
            logger.warning(f"No {field_name} provided through environment variable {env_var}")
            
        return value


class CloudIntegrationSettings(BaseSettings):
    """Main settings container for cloud integration."""
    storage: CloudStorageSettings = CloudStorageSettings()
    functions: CloudFunctionSettings = CloudFunctionSettings()
    
    # Get callback settings from app configuration
    callback_base_url: str = Field(
        default=app_settings.callback_base_url,
        description="Base URL for callbacks from cloud functions"
    )
    callback_route: str = Field(
        default=app_settings.callback_route,
        description="Route for cloud function callbacks"
    )
    callback_auth_token: str = Field(
        default=app_settings.callback_auth_token,
        description="Authentication token for cloud function callbacks"
    )
    max_upload_size: int = Field(
        default=app_settings.max_upload_size,
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
            "STORAGE_ENDPOINT": self.storage.endpoint_url,
            "STORAGE_REGION": self.storage.region,
            "LOG_LEVEL": app_settings.api.log_level
        }
    
    def validate_credentials(self):
        """Validate that all required credentials are set."""
        required_credentials = [
            ('storage.access_key', 'YANDEX_CLOUD_ACCESS_KEY'),
            ('storage.secret_key', 'YANDEX_CLOUD_SECRET_KEY'),
            ('functions.api_key', 'YANDEX_CLOUD_API_KEY'),
            ('functions.folder_id', 'YANDEX_CLOUD_FOLDER_ID'),
            ('callback_auth_token', 'CLOUD_CALLBACK_AUTH_TOKEN')
        ]
        
        missing = []
        for cred_name, env_var in required_credentials:
            parts = cred_name.split('.')
            value = self
            for part in parts:
                value = getattr(value, part)
                
            if not value:
                missing.append((cred_name, env_var))
        
        if missing:
            missing_envs = ', '.join([f"{name} ({env})" for name, env in missing])
            logger.warning(f"Missing required credentials: {missing_envs}")
            return False
        
        return True


# Create a global instance of the settings
cloud_settings = CloudIntegrationSettings()

# Check credentials on module load
if not cloud_settings.validate_credentials():
    logger.warning("Cloud integration may not function properly due to missing credentials") 