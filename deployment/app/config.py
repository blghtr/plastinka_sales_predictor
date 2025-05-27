"""
Central configuration module for the Plastinka Sales Predictor API.
Loads configuration from environment variables or config files.
"""
import os
import json
import yaml
import logging # Added
from pathlib import Path # Added
from typing import List, Dict, Any, Optional, Callable, Type, Tuple # Modified
from pydantic import Field, field_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import ( # Added
    InitSettingsSource,
    EnvSettingsSource,
    DotEnvSettingsSource,
    SecretsSettingsSource,
) # PydanticSettingsSource removed as it's likely internal
from functools import lru_cache

# Initialize logger for this module
logger = logging.getLogger(__name__)

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
        logger.error(f"Error loading config file: {str(e)}", exc_info=True)
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


def get_data_retention_config() -> Dict[str, Any]:
    """Get data retention specific configuration."""
    return get_config_values().get("data_retention", {})


def get_app_config() -> Dict[str, Any]:
    """Get application-level configuration."""
    config = get_config_values()
    return {k: v for k, v in config.items() if k not in ("api", "db", "datasphere", "data_retention")}


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
    x_api_key: str = Field(
        default="",
        description="API key for X-API-Key header authentication"
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
    
    _config_loader_func: Optional[Callable[[], Dict[str, Any]]] = get_api_config

    @classmethod
    def settings_customise_sources(
        cls: Type[BaseSettings],  # Changed from settings_cls to cls
        settings_cls_arg: Type[BaseSettings], # Added settings_cls_arg
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> Tuple[Any, ...]:
        """
        Customise the sources for loading settings.
        Order: __init__ args > config file dict > env vars > .env file > secrets > defaults.
        """
        config_file_data: Dict[str, Any] = {}
        # Use cls consistently
        if hasattr(cls, '_config_loader_func') and cls._config_loader_func is not None:
            loader_func = cls._config_loader_func
            loaded_data = loader_func()
            if isinstance(loaded_data, dict):
                config_file_data = loaded_data
        
        custom_dict_source = lambda: config_file_data

        return (
            init_settings,
            custom_dict_source,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    # Use SettingsConfigDict for Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        extra="ignore",
        # Values from config file (via customize_sources) take precedence over environment variables.
        env_nested_delimiter="__",
        customize_sources=settings_customise_sources,
    )


class DatabaseSettings(BaseSettings):
    """Database specific settings."""
    path: str = Field(
        default="deployment/data/plastinka.db", # Static default
        description="Path to SQLite database file"
    )
    url: str = Field(
        default="sqlite:///deployment/data/plastinka.db", # Static default
        description="SQLite database URL for SQLAlchemy"
    )

    _config_loader_func: Optional[Callable[[], Dict[str, Any]]] = get_db_config

    @classmethod
    def settings_customise_sources(
        cls: Type[BaseSettings],
        settings_cls_arg: Type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> Tuple[Any, ...]:
        """
        Customise the sources for loading settings.
        Order: __init__ args > config file dict > env vars > .env file > secrets > defaults.
        """
        config_file_data: Dict[str, Any] = {}
        if hasattr(cls, '_config_loader_func') and cls._config_loader_func is not None:
            loader_func = cls._config_loader_func
            loaded_data = loader_func()
            if isinstance(loaded_data, dict):
                config_file_data = loaded_data

        custom_dict_source = lambda: config_file_data
        
        return (
            init_settings,
            custom_dict_source,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
    
    # Use SettingsConfigDict for Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
        customize_sources=settings_customise_sources,
    )
    
    def reload(self):
        """
        Reload settings from environment variables and validate them.
        If validation fails, the original settings are retained and an error is logged.
        """
        current_path = self.path
        current_url = self.url

        try:
            # Get latest DATABASE_PATH and DATABASE_URL from environment
            # Use the current values as defaults if environment variables are not set
            # This differs slightly from the original, which had hardcoded defaults.
            # This change makes reloading more about *overriding* with env vars if present.
            env_path = os.environ.get("DATABASE_PATH")
            env_url = os.environ.get("DATABASE_URL")

            new_values = {}
            if env_path is not None:
                new_values["path"] = env_path
            if env_url is not None:
                new_values["url"] = env_url
            
            # If no new values from env, no need to reload or validate
            if not new_values:
                logger.info("DatabaseSettings.reload() called, but no relevant environment variables (DATABASE_PATH, DATABASE_URL) found to update.")
                return self

            # Create a dictionary with all current values, then update with new ones
            # This ensures all fields are present for validation if only one is being updated via env
            merged_values_for_validation = self.model_dump()
            merged_values_for_validation.update(new_values)

            # Validate the potentially new values by attempting to create a new model instance
            validated_settings = DatabaseSettings.model_validate(merged_values_for_validation)
            
            # Update the instance values from the validated settings
            # Use object.__setattr__ to bypass Pydantic's own protections if they exist here
            object.__setattr__(self, "path", validated_settings.path)
            object.__setattr__(self, "url", validated_settings.url)
            logger.info(f"DatabaseSettings reloaded and validated. Path: '{self.path}', URL: '{self.url}'")

        except ValidationError as e:
            logger.error(f"Failed to reload and validate DatabaseSettings: {e}. "
                         f"Original settings remain (Path: '{current_path}', URL: '{current_url}').")
            # Optionally, re-raise the error or handle it as per application requirements
            # For now, we log and keep original settings.
        
        return self


class TrainJobSettings(BaseSettings):
    """Settings specific to the DataSphere training job submission."""
    input_dir: str = Field(
        default="deployment/data/datasphere_input", # Static default
        description="Base directory for DataSphere training job inputs (params.json etc.)"
    )
    output_dir: str = Field(
        default="deployment/data/datasphere_output", # Static default
        description="Base directory for DataSphere training job outputs"
    )
    job_config_path: Optional[str] = Field(
        default=None, # Static default
        description="Optional path to a base DataSphere job configuration file (might not be used by submit_job)"
    )

    @classmethod
    def settings_customise_sources(
        cls: Type[BaseSettings],
        settings_cls_arg: Type[BaseSettings], # Parameter to accept Pydantic's settings_cls kwarg
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> Tuple[Any, ...]:
        # For TrainJobSettings, we don't have a dedicated file section,
        # so we just return the standard sources in default order.
        # Environment variables should be picked up by env_settings.
        return (
            init_settings,
            env_settings, # Ensure env_settings is explicitly included
            dotenv_settings,
            file_secret_settings,
        )

    model_config = SettingsConfigDict(
        env_prefix="DATASPHERE_TRAIN_JOB_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
        customize_sources=settings_customise_sources, # Added customize_sources
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
    
    _config_loader_func: Optional[Callable[[], Dict[str, Any]]] = get_datasphere_config

    @classmethod
    def settings_customise_sources(
        cls: Type[BaseSettings],
        settings_cls_arg: Type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> Tuple[Any, ...]:
        """
        Customise the sources for loading settings.
        Order: __init__ args > config file dict > env vars > .env file > secrets > defaults.
        """
        config_file_data: Dict[str, Any] = {}
        if hasattr(cls, '_config_loader_func') and cls._config_loader_func is not None:
            loader_func = cls._config_loader_func
            loaded_data = loader_func()
            if isinstance(loaded_data, dict):
                config_file_data = loaded_data

        custom_dict_source = lambda: config_file_data

        return (
            init_settings,
            custom_dict_source,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    # Use SettingsConfigDict for Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_prefix="DATASPHERE_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
        customize_sources=settings_customise_sources,
    )


class DataRetentionSettings(BaseSettings):
    """Data retention specific settings."""
    # Retention periods (in days)
    sales_retention_days: int = Field(
        default=int(os.environ.get("RETENTION_SALES_DAYS", 730)),  # ~2 years
        description="Retention period for sales data in days"
    )
    stock_retention_days: int = Field(
        default=int(os.environ.get("RETENTION_STOCK_DAYS", 730)),  # ~2 years
        description="Retention period for stock data in days"
    )
    prediction_retention_days: int = Field(
        default=int(os.environ.get("RETENTION_PREDICTION_DAYS", 365)),  # ~1 year
        description="Retention period for prediction data in days"
    )
    
    # Model management
    models_to_keep: int = Field(
        default=int(os.environ.get("RETENTION_MODELS_TO_KEEP", 5)),
        description="Number of models to keep per parameter set"
    )
    inactive_model_retention_days: int = Field(
        default=int(os.environ.get("RETENTION_INACTIVE_MODEL_DAYS", 90)),  # ~3 months
        description="Retention period for inactive models in days"
    )
    
    # Execution settings
    cleanup_enabled: bool = Field(
        default=os.environ.get("RETENTION_CLEANUP_ENABLED", "true").lower() == "true",
        description="Enable automatic data cleanup"
    )
    cleanup_schedule: str = Field(
        default=os.environ.get("RETENTION_CLEANUP_SCHEDULE", "0 3 * * 0"),  # 3:00 AM every Sunday
        description="Cleanup schedule in cron format"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="RETENTION_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__"
    )
    
    _config_loader_func: Optional[Callable[[], Dict[str, Any]]] = get_data_retention_config

    @classmethod
    def settings_customise_sources(
        cls: Type[BaseSettings],
        settings_cls_arg: Type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> Tuple[Any, ...]:
        """
        Customise the sources for loading settings.
        Order: __init__ args > config file dict > env vars > .env file > secrets > defaults.
        """
        config_file_data: Dict[str, Any] = {}
        if hasattr(cls, '_config_loader_func') and cls._config_loader_func is not None:
            loader_func = cls._config_loader_func
            loaded_data = loader_func()
            if isinstance(loaded_data, dict):
                config_file_data = loaded_data
        
        custom_dict_source = lambda: config_file_data

        return (
            init_settings,
            custom_dict_source,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
    
    # model_config is inherited or defined in DataRetentionSettings if needed
    # For this example, assuming it's similar to others if specific env_prefix is used.
    # If DataRetentionSettings needs its own customize_sources, it should be defined here.
    # For now, let's assume the above method is part of this class.
    model_config = SettingsConfigDict(
        env_prefix="RETENTION_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
        customize_sources=settings_customise_sources, # Added
    )


class AppSettings(BaseSettings):
    """Main application settings container."""
    # Explicitly declare API and DB settings as fields
    api: APISettings = Field(default_factory=APISettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    datasphere: DataSphereSettings = Field(default_factory=DataSphereSettings)
    data_retention: DataRetentionSettings = Field(default_factory=DataRetentionSettings)
    
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
    project_root_dir: str = Field(
        default=os.environ.get("PROJECT_ROOT_DIR", str(Path(__file__).resolve().parents[2])),
        description="Absolute path to the project root directory."
    )
    
    # Use SettingsConfigDict for Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__"
    )
    
    _config_loader_func: Optional[Callable[[], Dict[str, Any]]] = get_app_config

    @classmethod
    def settings_customise_sources(
        cls: Type[BaseSettings],
        settings_cls_arg: Type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> Tuple[Any, ...]:
        """
        Customise the sources for loading settings.
        Order: __init__ args > config file dict > env vars > .env file > secrets > defaults.
        """
        config_file_data: Dict[str, Any] = {}
        if hasattr(cls, '_config_loader_func') and cls._config_loader_func is not None:
            loader_func = cls._config_loader_func
            loaded_data = loader_func() # For AppSettings, this loads its direct fields
            if isinstance(loaded_data, dict):
                config_file_data = loaded_data
        
        custom_dict_source = lambda: config_file_data
        
        # Nested settings (api, db, etc.) will handle their own sources
        # when Pydantic initializes them.
        return (
            init_settings,
            custom_dict_source,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    # model_config is inherited or defined in AppSettings if needed
    model_config = SettingsConfigDict(
        env_file=".env", # AppSettings might not have its own prefix for direct fields
        extra="ignore",
        env_nested_delimiter="__", # For nested models if their env vars are prefixed by AppSettings field names
        customize_sources=settings_customise_sources, # Added
    )

    @field_validator('project_root_dir', mode='before')
    @classmethod
    def _resolve_project_root(cls, v):
        if v:
            # Ensure the path is absolute and resolved
            return str(Path(v).resolve())
        # Fallback if environment variable is empty or not set, re-apply default logic
        return str(Path(__file__).resolve().parents[2])
    
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

