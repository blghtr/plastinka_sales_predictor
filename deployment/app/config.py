"""
Central configuration module for the Plastinka Sales Predictor API.
Loads configuration from environment variables or config files.
"""
import os
import json
import yaml
from typing import List, Dict, Any, Optional, Callable
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

def load_config_file(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        file_path: Path to the config file
        
    Returns:
        Dict containing configuration values
    """
    if not os.path.exists(file_path):
        return {}
        
    with open(file_path, 'r') as f:
        if file_path.endswith('.json'):
            return json.load(f)
        elif file_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_path}")


# Load config file once and cache it
@lru_cache()
def get_config_values() -> Dict[str, Any]:
    """Load values from the config file if specified."""
    config_file_path = os.environ.get("CONFIG_FILE_PATH")
    if not config_file_path:
        return {}
    
    try:
        return load_config_file(config_file_path)
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        return {}


# Get config for specific sections
def get_api_config() -> Dict[str, Any]:
    """Get API specific configuration."""
    return get_config_values().get("api", {})


def get_db_config() -> Dict[str, Any]:
    """Get database specific configuration."""
    return get_config_values().get("db", {})


def get_datasphere_config() -> Dict[str, Any]:
    """Get DataSphere specific configuration."""
    return get_config_values().get("datasphere", {})


def get_app_config() -> Dict[str, Any]:
    """Get application-level configuration."""
    config = get_config_values()
    return {k: v for k, v in config.items() if k not in ("api", "db", "datasphere")}


class APISettings(BaseSettings):
    """API specific settings."""
    host: str = Field(
        default="0.0.0.0",
        description="Host to run the API server on"
    )
    port: int = Field(
        default=8000,
        description="Port to run the API server on"
    )
    debug: bool = Field(
        default=False,
        description="Run in debug mode"
    )
    allowed_origins: List[str] = Field(
        default_factory=lambda: os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
        description="List of allowed origins for CORS"
    )
    api_key: str = Field(
        default="",
        description="API key for authentication"
    )
    log_level: str = Field(
        default="INFO",
        description="Log level"
    )
    
    @field_validator('allowed_origins', mode='before')
    @classmethod
    def validate_allowed_origins(cls, v):
        """Validate and parse allowed origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    # Use SettingsConfigDict for Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        extra="ignore",
        # Configure so that values from config file take precedence over environment variables
        env_nested_delimiter="__"
    )
    
    def __init__(self, **data):
        # Merge config file values with the incoming data
        config_values = get_api_config()
        merged_data = {**config_values, **data}
        super().__init__(**merged_data)


class DatabaseSettings(BaseSettings):
    """Database specific settings."""
    path: str = Field(
        default=os.environ.get("DATABASE_PATH", "deployment/data/plastinka.db"),
        description="Path to SQLite database file"
    )
    
    # Use SettingsConfigDict for Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__"
    )
    
    def __init__(self, **data):
        # Merge config file values with the incoming data
        config_values = get_db_config()
        merged_data = {**config_values, **data}
        super().__init__(**merged_data)


class TrainJobSettings(BaseSettings):
    """Settings specific to the DataSphere training job submission."""
    input_dir: str = Field(
        default=os.environ.get("DATASPHERE_TRAIN_JOB_INPUT_DIR", "deployment/data/datasphere_input"),
        description="Base directory for DataSphere training job inputs (params.json etc.)"
    )
    output_dir: str = Field(
        default=os.environ.get("DATASPHERE_TRAIN_JOB_OUTPUT_DIR", "deployment/data/datasphere_output"),
        description="Base directory for DataSphere training job outputs"
    )
    job_config_path: Optional[str] = Field(
        default=os.environ.get("DATASPHERE_TRAIN_JOB_CONFIG_PATH", None),
        description="Optional path to a base DataSphere job configuration file (might not be used by submit_job)"
    )

    model_config = SettingsConfigDict(
        env_prefix="DATASPHERE_TRAIN_JOB_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__"
    )


class DataSphereSettings(BaseSettings):
    """DataSphere specific settings."""
    # Client settings
    project_id: str = Field(
        default=os.environ.get("DATASPHERE_PROJECT_ID", ""),
        description="ID of the DataSphere project"
    )
    folder_id: str = Field(
        default=os.environ.get("DATASPHERE_FOLDER_ID", ""),
        description="Yandex Cloud folder ID"
    )
    oauth_token: Optional[str] = Field(
        default=os.environ.get("DATASPHERE_OAUTH_TOKEN", None),
        description="Yandex Cloud OAuth token (optional, uses profile/env if None)",
    )
    yc_profile: Optional[str] = Field(
        default=os.environ.get("DATASPHERE_YC_PROFILE", None),
        description="Yandex Cloud CLI profile name (optional)"
    )
    

    
    # Nested Train Job Settings
    train_job: TrainJobSettings = Field(default_factory=TrainJobSettings)
    
    # Polling configuration
    max_polls: int = Field(
        default=int(os.environ.get("DATASPHERE_MAX_POLLS", 120)),
        description="Maximum number of times to poll DataSphere job status"
    )
    poll_interval: float = Field(
        default=float(os.environ.get("DATASPHERE_POLL_INTERVAL", 5.0)),
        description="Interval in seconds between polling DataSphere job status"
    )

    # Add the new setting here
    download_diagnostics_on_success: bool = Field(
        default=os.environ.get("DATASPHERE_DOWNLOAD_DIAGNOSTICS_ON_SUCCESS", "false").lower() == "true",
        description="Whether to download logs/diagnostics for successful DataSphere jobs"
    )

    @property
    def client(self) -> Dict[str, Any]:
        """Get client configuration as a dictionary."""
        return {
            "project_id": self.project_id,
            "folder_id": self.folder_id,
            "oauth_token": self.oauth_token,
            "yc_profile": self.yc_profile
        }
    
    # Use SettingsConfigDict for Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_prefix="DATASPHERE_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__"
    )
    
    def __init__(self, **data):
        # Merge config file values with the incoming data
        config_values = get_datasphere_config()
        # Handle nested train_job settings correctly during merge
        train_job_config = config_values.pop("train_job", {})
        merged_data = {**config_values, **data}
        # Merge top-level first
        # Merge train_job settings if they exist in incoming data or config file
        if 'train_job' in merged_data or train_job_config:
            merged_data['train_job'] = TrainJobSettings(**{**train_job_config, **merged_data.get('train_job', {})})
        super().__init__(**merged_data)


class AppSettings(BaseSettings):
    """Main application settings container."""
    # Explicitly declare API and DB settings as fields
    api: APISettings = Field(default_factory=APISettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    datasphere: DataSphereSettings = Field(default_factory=DataSphereSettings)
    
    env: str = Field(
        default=os.environ.get("APP_ENV", "development"),
        description="Application environment (development, testing, production)"
    )
    callback_base_url: str = Field(
        default=os.environ.get("FASTAPI_CALLBACK_BASE_URL", "http://localhost:8000"),
        description="Base URL for cloud function callbacks"
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
        default=int(os.environ.get("MAX_UPLOAD_SIZE", 50 * 1024 * 1024)),  # 50 MB default
        description="Maximum upload size in bytes"
    )
    temp_upload_dir: str = Field(
        default=os.environ.get("TEMP_UPLOAD_DIR", "./temp_uploads"),
        description="Directory for temporary file uploads"
    )
    max_models_to_keep: int = Field(
        default=int(os.environ.get("MAX_MODELS_TO_KEEP", 5)), 
        description="Maximum number of trained model artifacts to keep locally"
    )
    
    # Model and parameters selection settings
    default_metric: str = Field(
        default=os.environ.get("DEFAULT_METRIC", "val_MIC"),
        description="Default metric name to use for selecting best models/parameters"
    )
    default_metric_higher_is_better: bool = Field(
        default=os.environ.get("DEFAULT_METRIC_HIGHER_IS_BETTER", "true").lower() == "true",
        description="Whether higher values of the default metric are better"
    )
    auto_select_best_params: bool = Field(
        default=os.environ.get("AUTO_SELECT_BEST_PARAMS", "true").lower() == "true",
        description="Whether to automatically select the best parameter set as active"
    )
    auto_select_best_model: bool = Field(
        default=os.environ.get("AUTO_SELECT_BEST_MODEL", "true").lower() == "true",
        description="Whether to automatically select the best model as active"
    )
    
    # Use SettingsConfigDict for Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__"
    )
    
    def __init__(self, **data):
        # Merge config file values with the incoming data
        config_values = get_app_config()
        merged_data = {**config_values, **data}
        
        # Let Pydantic handle field initialization
        super().__init__(**merged_data)
    
    @property
    def callback_url(self) -> str:
        """Get the full callback URL."""
        return f"{self.callback_base_url}{self.callback_route}"
    
    @property
    def is_development(self) -> bool:
        """Check if the application is running in development mode."""
        return self.env.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if the application is running in production mode."""
        return self.env.lower() == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if the application is running in testing mode."""
        return self.env.lower() == "testing"


# Create a global instance of the settings
settings = AppSettings() 

