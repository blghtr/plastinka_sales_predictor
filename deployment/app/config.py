"""
Central configuration module for the Plastinka Sales Predictor API.
Loads configuration from environment variables or config files.
"""

import functools
import json
import logging  # Added
import os
import tempfile  # Added for test environment fallback
from collections.abc import Callable  # Modified
from functools import lru_cache
from pathlib import Path  # Added
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import (  # Added
    DotEnvSettingsSource,
    EnvSettingsSource,
    InitSettingsSource,
    SecretsSettingsSource,
)  # PydanticSettingsSource removed as it's likely internal

# Initialize logger for this module
logger = logging.getLogger(__name__)


def _get_default_data_root_dir() -> str:
    """
    Get the default data root directory with fallback logic for test environments.

    Returns:
        str: Path to the default data root directory

    Raises:
        RuntimeError: If home directory cannot be determined and fallback fails
    """
    try:
        # Primary approach: use user home directory
        # For Windows, prefer APPDATA, for Linux/macOS, prefer ~/.local/share
        if os.name == 'nt':  # Windows
            app_data = os.environ.get('APPDATA')
            if app_data:
                return str(Path(app_data) / "PlastinkaSalesPredictor")
            else:
                # Fallback if APPDATA not set
                return str(Path("~/.plastinka_sales_predictor").expanduser())
        else:  # Linux/macOS or other POSIX-like
            return str(Path("~/.local/share/plastinka_sales_predictor").expanduser())
    except (RuntimeError, OSError) as e:
        logger.warning(
            f"Could not determine appropriate data root directory ({e}), using temporary directory fallback"
        )
        try:
            # Fallback for test environments: use system temp directory
            temp_root = tempfile.gettempdir()
            fallback_path = os.path.join(temp_root, "plastinka_sales_predictor_test")
            logger.info(f"Using fallback data root directory: {fallback_path}")
            return fallback_path
        except Exception as fallback_error:
            # Final fallback: current working directory
            fallback_path = os.path.join(os.getcwd(), ".plastinka_data")
            logger.warning(
                f"Temp directory fallback failed ({fallback_error}), using current directory: {fallback_path}"
            )
            return fallback_path


def load_config_file(file_path: str) -> dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.

    Args:
        file_path: Path to the config file

    Returns:
        Dict containing configuration values
    """
    if not os.path.exists(file_path):
        return {}

    with open(file_path) as f:
        if file_path.endswith(".json"):
            return json.load(f)
        elif file_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_path}")


# Load config file once and cache it
@lru_cache
def get_config_values() -> dict[str, Any]:
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
def get_api_config() -> dict[str, Any]:
    """Get API specific configuration."""
    return get_config_values().get("api", {})


def get_db_config() -> dict[str, Any]:
    """Get database specific configuration."""
    return get_config_values().get("db", {})


def get_datasphere_config() -> dict[str, Any]:
    """Get DataSphere specific configuration."""
    return get_config_values().get("datasphere", {})


def get_data_retention_config() -> dict[str, Any]:
    """Get data retention specific configuration."""
    return get_config_values().get("data_retention", {})


def get_app_config() -> dict[str, Any]:
    """Get application-level configuration."""
    config = get_config_values()
    return {
        k: v
        for k, v in config.items()
        if k not in ("api", "db", "datasphere", "data_retention")
    }


# Общий валидатор для создания директорий
def ensure_directory_exists(path_value: str) -> str:
    """
    Проверяет существование директории и создает её при необходимости.

    Args:
        path_value: Путь к директории

    Returns:
        Тот же путь, после создания директории
    """
    if not path_value:
        return path_value

    # Если это путь к файлу, создаем родительскую директорию
    path = Path(path_value)
    if "." in path.name and not path.name.endswith(("/", "\\")):
        # Вероятно это файл - создаем родительскую директорию
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Это директория - создаем её
        path.mkdir(parents=True, exist_ok=True)

    return path_value


class APISettings(BaseSettings):
    """API specific settings."""

    host: str = Field(default="0.0.0.0", description="Host to run the API server on")
    port: int = Field(default=8000, description="Port to run the API server on")
    debug: bool = Field(default=False, description="Run in debug mode")
    admin_user: str = Field(default="admin", description="Username for the admin user")
    admin_api_key_hash: str = Field(
        default="",
        description="Hashed admin API key for Bearer authentication (set via API_ADMIN_API_KEY_HASH).",
    )
    x_api_key_hash: str = Field(
        default="", description="Hashed API key for X-API-Key header authentication (set via API_X_API_KEY_HASH)."
    )
    log_level: str = Field(default="INFO", description="Log level")
    allowed_origins: list[str] = Field(
        default=[],
        description="A list of origins that should be permitted to make cross-origin requests.",
    )
    docs_security_enabled: bool = Field(
        default=True,
        description="Enable security for API documentation endpoints (/docs, /redoc). Defaults to True in production, False in development.",
    )

    @field_validator("docs_security_enabled", mode="before")
    @classmethod
    def set_default_docs_security(cls, v: bool | None, values: dict[str, Any]) -> bool:
        """Set default value for docs_security_enabled based on environment."""
        if v is not None:
            return v

        # Fallback to environment variable if not explicitly provided
        env_val = os.getenv("API_DOCS_SECURITY_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes")

        # Default based on app environment
        app_env = values.get("env", "production")  # Default to production if env not set
        is_dev = app_env.lower() == "development"

        logger.info(f"docs_security_enabled not set, defaulting to {not is_dev} for env: {app_env}")
        return not is_dev

    _config_loader_func: Callable[[], dict[str, Any]] | None = get_api_config

    @classmethod
    def settings_customise_sources(
        cls: type[BaseSettings],  # Changed from settings_cls to cls
        settings_cls_arg: type[BaseSettings],  # Added settings_cls_arg
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> tuple[Any, ...]:
        """
        Customise the sources for loading settings.
        Order: __init__ args > config file dict > env vars > .env file > secrets > defaults.
        """
        config_file_data: dict[str, Any] = {}
        # Use cls consistently
        if hasattr(cls, "_config_loader_func") and cls._config_loader_func is not None:
            loader_func = cls._config_loader_func
            loaded_data = loader_func()
            if isinstance(loaded_data, dict):
                config_file_data = loaded_data

        def custom_dict_source():
            return config_file_data

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

    # These paths will be computed automatically from data_root_dir in AppSettings
    # Removing static defaults that reference deployment/data
    filename: str = Field(
        default="plastinka.db",
        description="SQLite database filename (path will be computed from data_root_dir)",
    )

    database_busy_timeout: int = Field(
        default=5000, description="SQLite busy timeout in milliseconds"
    )

    # Database directory creation is handled in AppSettings computed properties

    _config_loader_func: Callable[[], dict[str, Any]] | None = get_db_config

    @classmethod
    def settings_customise_sources(
        cls: type[BaseSettings],
        settings_cls_arg: type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> tuple[Any, ...]:
        """
        Customise the sources for loading settings.
        Order: __init__ args > config file dict > env vars > .env file > secrets > defaults.
        """
        config_file_data: dict[str, Any] = {}
        if hasattr(cls, "_config_loader_func") and cls._config_loader_func is not None:
            loader_func = cls._config_loader_func
            loaded_data = loader_func()
            if isinstance(loaded_data, dict):
                config_file_data = loaded_data

        def custom_dict_source():
            return config_file_data

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

    # Removed reload method - paths are now computed automatically in AppSettings


class DataSphereSettings(BaseSettings):
    """DataSphere specific settings."""

    # Client settings
    project_id: str = Field(default="", description="ID of the DataSphere project")
    folder_id: str = Field(default="", description="Yandex Cloud folder ID")

    oauth_token: str = Field(
        default="",
        description="Yandex Cloud OAuth token for user authentication.",
    )

    yc_profile: str = Field(
        default="datasphere-prod",
        description="Yandex Cloud CLI profile name for service account authentication.",
    )

    auth_method: str = Field(
        default="auto",
        description="Authentication method: 'auto', 'yc_profile', or 'oauth_token'.",
    )

    # Polling configuration
    max_polls: int = Field(
        default=72, description="Maximum number of times to poll DataSphere job status"
    )
    poll_interval: float = Field(
        default=300.0,
        description="Interval in seconds between polling DataSphere job status",
    )

    # Add the new setting here
    download_diagnostics_on_success: bool = Field(
        default=False,
        description="Whether to download logs/diagnostics for successful DataSphere jobs",
    )

    client_init_timeout_seconds: int = Field(
        default=60,
        description="Timeout for DataSphere client initialization in seconds",
    )
    client_submit_timeout_seconds: int = Field(
        default=300,
        description="Timeout for DataSphere client job submission in seconds",
    )
    client_status_timeout_seconds: int = Field(
        default=30,
        description="Timeout for DataSphere client job status checks in seconds",
    )
    client_download_timeout_seconds: int = Field(
        default=300,
        description="Timeout for DataSphere client results download in seconds",
    )
    client_cancel_timeout_seconds: int = Field(
        default=60,
        description="Timeout for DataSphere client job cancellation in seconds",
    )

    @field_validator("auth_method", mode="before")
    @classmethod
    def validate_auth_method(cls, v: str):
        allowed = {"auto", "yc_profile", "oauth_token"}
        if v not in allowed:
            raise ValueError(f"Invalid auth_method '{v}'. Allowed values: {allowed}")
        return v

    @field_validator("oauth_token", mode="before")
    @classmethod
    def validate_oauth_token(cls, v: str):
        # Check if YC_OAUTH_TOKEN environment variable is set
        import os
        yc_token = os.environ.get("YC_OAUTH_TOKEN")
        if yc_token and not v:
            return yc_token
        return v

    @property
    def client(self) -> dict[str, Any]:
        """Get client configuration as a dictionary."""
        return {
            "project_id": self.project_id,
            "folder_id": self.folder_id,
            "oauth_token": self.oauth_token,
            "yc_profile": self.yc_profile,
            "auth_method": self.auth_method,
        }

    @property
    def api_client(self) -> dict[str, Any]:
        """Get client configuration as a dictionary (alias for client property)."""
        return self.client

    _config_loader_func: Callable[[], dict[str, Any]] | None = get_datasphere_config

    @classmethod
    def settings_customise_sources(
        cls: type[BaseSettings],
        settings_cls_arg: type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> tuple[Any, ...]:
        """
        Customise the sources for loading settings.
        Order: __init__ args > config file dict > env vars > .env file > secrets > defaults.
        """
        config_file_data: dict[str, Any] = {}
        if hasattr(cls, "_config_loader_func") and cls._config_loader_func is not None:
            loader_func = cls._config_loader_func
            loaded_data = loader_func()
            if isinstance(loaded_data, dict):
                config_file_data = loaded_data

        def custom_dict_source():
            return config_file_data

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
        default=730,  # ~2 years
        description="Retention period for sales data in days",
    )
    stock_retention_days: int = Field(
        default=730,  # ~2 years
        description="Retention period for stock data in days",
    )
    prediction_retention_days: int = Field(
        default=365,  # ~1 year
        description="Retention period for prediction data in days",
    )

    # Model management
    models_to_keep: int = Field(
        default=5, description="Number of models to keep per parameter set"
    )
    inactive_model_retention_days: int = Field(
        default=90,  # ~3 months
        description="Retention period for inactive models in days",
    )
    backup_retention_days: int = Field(
        default=30,  # 30 days
        description="Retention period for database backup files in days",
    )

    # Execution settings
    cleanup_enabled: bool = Field(
        default=True, description="Enable automatic data cleanup"
    )
    cleanup_schedule: str = Field(
        default="0 3 * * 0",  # 3:00 AM every Sunday
        description="Cleanup schedule in cron format",
    )

    model_config = SettingsConfigDict(
        env_prefix="RETENTION_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
    )

    _config_loader_func: Callable[[], dict[str, Any]] | None = get_data_retention_config

    @classmethod
    def settings_customise_sources(
        cls: type[BaseSettings],
        settings_cls_arg: type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> tuple[Any, ...]:
        """
        Customise the sources for loading settings.
        Order: __init__ args > config file dict > env vars > .env file > secrets > defaults.
        """
        config_file_data: dict[str, Any] = {}
        if hasattr(cls, "_config_loader_func") and cls._config_loader_func is not None:
            loader_func = cls._config_loader_func
            loaded_data = loader_func()
            if isinstance(loaded_data, dict):
                config_file_data = loaded_data

        def custom_dict_source():
            return config_file_data

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
        customize_sources=settings_customise_sources,  # Added
    )


class TuningSettings(BaseSettings):
    """Hyperparameter tuning specific settings."""

    num_samples_lite: int = Field(50, description="Samples for lite mode")
    num_samples_full: int = Field(200, description="Samples for full mode")
    max_concurrent: int = Field(16, description="Max concurrent Ray trials")
    resources: dict[str, int] = Field({"cpu": 32}, description="Resource allocation per trial")
    best_configs_to_save: int = Field(5, description="How many best configs to persist after tuning")
    seed_configs_limit: int = Field(5, description="How many historical configs to pass to tuning as starters")
    metric_threshold: float = Field(1.2, description="Minimum metric for initial configs selection")

    # NEW: default tuning mode (overridable per-job via API) – 'lite' or 'full'
    mode: str = Field("lite", description="Default tuning mode: lite or full")

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, v: str):
        allowed = {"lite", "full"}
        if v not in allowed:
            raise ValueError(f"Invalid tuning mode '{v}'. Allowed values: {allowed}")
        return v

    model_config = SettingsConfigDict(
        env_prefix="TUNING_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
    )


class AppSettings(BaseSettings):
    """Main application settings container."""

    # Explicitly declare API and DB settings as fields
    api: APISettings = Field(default_factory=APISettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    datasphere: DataSphereSettings = Field(default_factory=DataSphereSettings)
    data_retention: DataRetentionSettings = Field(default_factory=DataRetentionSettings)
    tuning: TuningSettings = Field(default_factory=TuningSettings)

    env: str = Field(
        default="development",
        description="Application environment (development, testing, production)",
    )
    max_upload_size: int = Field(
        default=50 * 1024 * 1024,  # 50 MB default
        description="Maximum upload size in bytes",
    )

    # Data processing behavior
    fill_missing_days: bool = Field(
        default=True,
        description="If true, patch trailing missing days at end of month during data processing",
    )

    # SQLite configuration
    sqlite_max_variables: int = Field(
        default=900,
        description="Maximum number of variables in SQLite queries (safe limit below 999)",
    )

    # Refractory period configuration
    job_refractory_seconds_default: int = Field(
        default=300,
        description="Default refractory period in seconds between submissions of the same job type",
    )
    job_refractory_seconds_map: dict[str, int] = Field(
        default_factory=dict,
        description="Optional per-job-type overrides for refractory period (e.g., {'training': 600})",
    )

    price_category_bins: list[float] = Field(
        default=[
            0.0,
            689.999,
            2490.0,
            3990.0,
            5590.0,
            7390.0,
            8590.0,
            11990.0,
            np.inf,
        ],
        description="Static bins for price categorization as a sequence of scalars.",
    )

    @property
    def price_category_interval_index(self) -> pd.IntervalIndex:
        """Get price category bins as a pandas IntervalIndex."""
        import pandas as pd
        return pd.IntervalIndex.from_breaks(self.price_category_bins, closed="right")

    @property
    def temp_upload_dir(self) -> str:
        """Directory for temporary file uploads."""
        path = os.path.join(self.data_root_dir, "temp_uploads")
        ensure_directory_exists(path)
        return path

    @property
    def file_storage_dir(self) -> str:
        """Base directory for storing uploaded files (models, reports, etc.)."""
        path = os.path.join(self.data_root_dir, "uploads")
        ensure_directory_exists(path)
        return path

    # Model and parameters selection settings
    default_metric: str = Field(
        default="val_MIWS_MIC_Ratio",
        description="Default metric name to use for selecting best models/parameters",
    )
    default_metric_higher_is_better: bool = Field(
        default=False,
        description="Whether higher values of the default metric are better",
    )
    auto_select_best_configs: bool = Field(
        default=True,
        description="Whether to automatically select the best parameter set as active",
    )
    auto_select_best_model: bool = Field(
        default=True,
        description="Whether to automatically select the best model as active",
    )
    metric_thesh_for_health_check: float = Field(
        default=1.2,
        description="Minimum value of the primary metric for the active model to be considered healthy.",
    )
    project_root_dir: str = Field(
        default=str(Path(__file__).resolve().parents[2]),
        description="Absolute path to the project root directory.",
    )

    # New smart data directory configuration
    data_root_dir: str = Field(
        default=_get_default_data_root_dir(),
        description="Root directory for all application data, models, logs, etc.",
    )

    # Smart computed properties for all data paths
    @property
    def database_path(self) -> str:
        """Compute database file path from data_root_dir."""
        path = os.path.join(self.data_root_dir, "database", self.db.filename)
        ensure_directory_exists(path)  # Create parent directory
        return path

    @property
    def database_url(self) -> str:
        """Compute database URL from database_path."""
        return f"sqlite:///{self.database_path}"

    @property
    def models_dir(self) -> str:
        """Directory for storing model files."""
        path = os.path.join(self.data_root_dir, "models")
        ensure_directory_exists(path)
        return path

    @property
    def datasphere_input_dir(self) -> str:
        """Directory for DataSphere job inputs."""
        path = os.path.join(self.data_root_dir, "datasphere_input")
        ensure_directory_exists(path)
        return path

    @property
    def datasphere_output_dir(self) -> str:
        """Directory for DataSphere job outputs."""
        path = os.path.join(self.data_root_dir, "datasphere_output")
        ensure_directory_exists(path)
        return path

    @property
    def predictions_dir(self) -> str:
        """Directory for prediction outputs."""
        path = os.path.join(self.data_root_dir, "predictions")
        ensure_directory_exists(path)
        return path

    @property
    def reports_dir(self) -> str:
        """Directory for generated reports."""
        path = os.path.join(self.data_root_dir, "reports")
        ensure_directory_exists(path)
        return path

    @property
    def logs_dir(self) -> str:
        """Directory for application logs."""
        path = os.path.join(self.data_root_dir, "logs")
        ensure_directory_exists(path)
        return path

    @property
    def database_backup_dir(self) -> str:
        """Directory for database backups."""
        path = os.path.join(self.data_root_dir, "backups", "database")
        ensure_directory_exists(path)
        return path

    @property
    def datasphere_jobs_base_dir(self) -> str:
        """Directory containing DataSphere job scripts."""
        return os.path.join(self.project_root_dir, "plastinka_sales_predictor", "datasphere_jobs")

    @property
    def datasphere_train_dir(self) -> str:
        """Directory for DataSphere train job scripts."""
        return os.path.join(self.datasphere_jobs_base_dir, "train")

    @property
    def datasphere_tune_dir(self) -> str:
        """Directory for DataSphere tune job scripts."""
        return os.path.join(self.datasphere_jobs_base_dir, "tune")

    @property
    def datasphere_job_config_path(self) -> str:
        """Path to DataSphere job configuration file."""
        return os.path.join(self.datasphere_jobs_base_dir, "config.yaml")



    # Валидатор для создания базовой директории данных
    @field_validator("data_root_dir", mode="after")
    @classmethod
    def ensure_data_root_exists(cls, v: str) -> str:
        """Проверяет существование корневой директории данных и создает её при необходимости."""
        return ensure_directory_exists(v)

    # Use SettingsConfigDict for Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_nested_delimiter="__"
    )

    _config_loader_func: Callable[[], dict[str, Any]] | None = get_app_config

    @classmethod
    def settings_customise_sources(
        cls: type[BaseSettings],
        settings_cls_arg: type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> tuple[Any, ...]:
        """
        Customise the sources for loading settings.
        Order: __init__ args > config file dict > env vars > .env file > secrets > defaults.
        """
        config_file_data: dict[str, Any] = {}
        if hasattr(cls, "_config_loader_func") and cls._config_loader_func is not None:
            loader_func = cls._config_loader_func
            loaded_data = loader_func()  # For AppSettings, this loads its direct fields
            if isinstance(loaded_data, dict):
                config_file_data = loaded_data

        def custom_dict_source():
            return config_file_data

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
        env_file=".env",  # AppSettings might not have its own prefix for direct fields
        extra="ignore",
        env_nested_delimiter="__",  # For nested models if their env vars are prefixed by AppSettings field names
        customize_sources=settings_customise_sources,  # Added
    )

    @field_validator("project_root_dir", mode="before")
    @classmethod
    def _resolve_project_root(cls, v):
        if v:
            # Ensure the path is absolute and resolved
            return str(Path(v).resolve())
        # Fallback if environment variable is empty or not set, re-apply default logic
        return str(Path(__file__).resolve().parents[2])

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

    @property
    def model_storage_dir(self) -> str:
        """Directory for storing model files (alias for models_dir)."""
        return self.models_dir

    @property
    def datasphere_job_train_dir(self) -> str:
        """Alias for datasphere_train_dir for backward compatibility."""
        return self.datasphere_train_dir

    @property
    def datasphere_job_tune_dir(self) -> str:
        """Alias for datasphere_tune_dir for backward compatibility."""
        return self.datasphere_tune_dir

    @property
    def datasphere_job_dir(self) -> str:
        """Alias for datasphere_jobs_base_dir for backward compatibility."""
        return self.datasphere_jobs_base_dir

    def get_job_refractory_seconds(self, job_type: str) -> int:
        """Return refractory seconds for a given job type using overrides or default."""
        try:
            # Normalize key to match how job_type is stored (enum values are lower_snake)
            key = (job_type or "").lower()
            return int(self.job_refractory_seconds_map.get(key, self.job_refractory_seconds_default))
        except Exception:
            return self.job_refractory_seconds_default


# Create a global instance of the settings
# settings = AppSettings()


@functools.lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Get the application settings.
    The settings are loaded once and cached.
    """
    return AppSettings()
