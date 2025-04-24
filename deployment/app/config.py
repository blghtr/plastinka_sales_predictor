"""
Central configuration module for the Plastinka Sales Predictor API.
Loads configuration from environment variables or config files.
"""
import os
import json
import yaml
from typing import List, Dict, Any, Optional
from pydantic import BaseSettings, Field, validator

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


class ConfigFileSettings(BaseSettings):
    """Configuration for loading config files."""
    config_file_path: Optional[str] = Field(
        default=os.environ.get("CONFIG_FILE_PATH", None),
        description="Path to configuration file (JSON or YAML)"
    )

    @property
    def config_values(self) -> Dict[str, Any]:
        """Load values from the config file if specified."""
        if not self.config_file_path:
            return {}
        
        try:
            return load_config_file(self.config_file_path)
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            return {}


# Global config file settings instance
config_file = ConfigFileSettings()


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
    
    @validator('allowed_origins', pre=True)
    def validate_allowed_origins(cls, v):
        """Validate and parse allowed origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    class Config:
        """Pydantic config for APISettings."""
        # Load configuration from config file if available
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                # Insert config file values before environment variables
                lambda settings: config_file.config_values.get("api", {}),
                env_settings,
                file_secret_settings,
            )


class DatabaseSettings(BaseSettings):
    """Database specific settings."""
    path: str = Field(
        default=os.environ.get("DATABASE_PATH", "deployment/data/plastinka.db"),
        description="Path to SQLite database file"
    )
    
    class Config:
        """Pydantic config for DatabaseSettings."""
        # Load configuration from config file if available
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                # Insert config file values before environment variables
                lambda settings: config_file.config_values.get("db", {}),
                env_settings,
                file_secret_settings,
            )


class AppSettings(BaseSettings):
    """Main application settings container."""
    api: APISettings = APISettings()
    db: DatabaseSettings = DatabaseSettings()
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
    
    class Config:
        """Pydantic config for AppSettings."""
        # Load configuration from config file if available
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                # Insert config file values before environment variables
                lambda settings: {k: v for k, v in config_file.config_values.items() 
                                 if k not in ("api", "db")},
                env_settings,
                file_secret_settings,
            )


# Create a global instance of the settings
settings = AppSettings() 