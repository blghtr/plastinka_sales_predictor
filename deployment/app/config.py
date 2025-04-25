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


def get_app_config() -> Dict[str, Any]:
    """Get application-level configuration."""
    config = get_config_values()
    return {k: v for k, v in config.items() if k not in ("api", "db")}


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


class AppSettings(BaseSettings):
    """Main application settings container."""
    # Explicitly declare API and DB settings as fields
    api: APISettings = Field(default_factory=APISettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    
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