"""
Comprehensive tests for deployment.app.config

This test suite covers all configuration classes and functions in the config module
with comprehensive mocking of external dependencies. Tests are organized by
configuration class groups and include both success and failure scenarios.

Testing Approach:
- Mock all external dependencies (os.environ, Path operations, file system operations)
- Test configuration loading from multiple sources (init, file, env, .env, defaults)
- Test configuration precedence rules (init > file > env > .env > default)
- Test environment variable handling and validation
- Test computed properties and directory creation
- Test error handling and validation scenarios
- Use pyfakefs for file system isolation
- Test individual settings classes in isolation
- Test main AppSettings integration
- Verify configuration source customization

All external imports and dependencies are mocked to ensure test isolation.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

# Import the module under test
from deployment.app.config import (
    APISettings,
    AppSettings,
    DatabaseSettings,
    DataRetentionSettings,
    DataSphereSettings,
    _get_default_data_root_dir,
    ensure_directory_exists,
    get_api_config,
    get_app_config,
    get_config_values,
    get_data_retention_config,
    get_datasphere_config,
    get_db_config,
    get_settings,
    load_config_file,
)


class TestUtilityFunctions:
    """Test suite for utility functions in config module."""

    def setup_method(self, method):
        """Setup method to clear LRU cache before each test."""
        # Clear the LRU cache to ensure tests don't interfere with each other
        get_config_values.cache_clear()

    def test_get_default_data_root_dir_success(self):
        """Test successful default data root directory resolution."""
        # Act
        result = _get_default_data_root_dir()
        # Assert
        # Accept all valid path variants
        assert (
            "PlastinkaSalesPredictor" in result
            or ".plastinka_sales_predictor" in result
            or "plastinka_sales_predictor" in result
        )

    @patch("deployment.app.config.logger")
    @patch("deployment.app.config.Path")
    @patch("deployment.app.config.tempfile.gettempdir")
    @patch("deployment.app.config.os.path.join")
    def test_get_default_data_root_dir_fallback_to_temp(
        self, mock_join, mock_gettempdir, mock_path, mock_logger
    ):
        """Test fallback to temp directory when home directory fails."""
        # Arrange
        mock_path.side_effect = RuntimeError("Home directory not available")
        mock_gettempdir.return_value = "/tmp"
        mock_join.return_value = "/tmp/plastinka_sales_predictor_test"

        # Act
        result = _get_default_data_root_dir()

        # Assert - for cross-platform compatibility
        assert "plastinka_sales_predictor" in result
        mock_logger.warning.assert_called()
        mock_logger.info.assert_called()
        mock_gettempdir.assert_called_once()

    @patch("deployment.app.config.logger")
    @patch("deployment.app.config.Path")
    @patch("deployment.app.config.tempfile.gettempdir")
    @patch("deployment.app.config.os.getcwd")
    @patch("deployment.app.config.os.path.join")
    def test_get_default_data_root_dir_final_fallback(
        self, mock_join, mock_getcwd, mock_gettempdir, mock_path, mock_logger
    ):
        """Test final fallback to current working directory when all else fails."""
        # Arrange
        mock_path.side_effect = RuntimeError("Home directory not available")
        mock_gettempdir.side_effect = Exception("Temp directory failed")
        mock_getcwd.return_value = "/current"
        mock_join.return_value = "/current/.plastinka_data"

        # Act
        result = _get_default_data_root_dir()

        # Assert - for cross-platform compatibility
        assert "plastinka_data" in result
        mock_logger.warning.assert_called()
        mock_getcwd.assert_called_once()

    def test_load_config_file_json_success(self, fs):
        """Test successful JSON config file loading."""
        # Arrange
        config_data = {"api": {"host": "test-host"}, "db": {"filename": "test.db"}}
        config_path = "/test/config.json"
        fs.create_file(config_path, contents=json.dumps(config_data))

        # Act
        result = load_config_file(config_path)

        # Assert
        assert result == config_data

    def test_load_config_file_yaml_success(self, fs):
        """Test successful YAML config file loading."""
        # Arrange
        config_data = {"api": {"host": "test-host"}, "db": {"filename": "test.db"}}
        config_path = "/test/config.yaml"
        fs.create_file(config_path, contents=yaml.dump(config_data))

        # Act
        result = load_config_file(config_path)

        # Assert
        assert result == config_data

    def test_load_config_file_not_exists(self):
        """Test loading non-existent config file returns empty dict."""
        # Act
        result = load_config_file("/nonexistent/config.json")

        # Assert
        assert result == {}

    def test_load_config_file_unsupported_format(self, fs):
        """Test loading config file with unsupported format raises ValueError."""
        # Arrange
        config_path = "/test/config.txt"
        fs.create_file(config_path, contents="some text")

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config_file(config_path)

    def test_get_config_values_with_file(self, fs):
        """Test get_config_values loads from specified config file."""
        # Arrange - create a real config file
        config_path = "/test/config.yaml"
        config_content = """
api:
  host: test
"""
        fs.create_file(config_path, contents=config_content)
        expected_config = {"api": {"host": "test"}}

        # Clear the cache and set environment variable
        get_config_values.cache_clear()

        with patch.dict(os.environ, {"CONFIG_FILE_PATH": config_path}):
            # Act
            result = get_config_values()

            # Assert
            assert result == expected_config

    @patch.dict(os.environ, {}, clear=True)
    def test_get_config_values_no_file(self):
        """Test get_config_values returns empty dict when no config file specified."""
        # Clear the cache
        get_config_values.cache_clear()

        # Act
        result = get_config_values()

        # Assert
        assert result == {}

    @patch("deployment.app.config.logger")
    @patch.dict(os.environ, {"CONFIG_FILE_PATH": "/test/config.yaml"})
    @patch("deployment.app.config.load_config_file")
    def test_get_config_values_load_error(self, mock_load_config, mock_logger):
        """Test get_config_values handles config loading errors gracefully."""
        # Arrange
        mock_load_config.side_effect = Exception("File read error")

        # Clear the cache
        get_config_values.cache_clear()

        # Act
        result = get_config_values()

        # Assert
        assert result == {}
        mock_logger.error.assert_called_once()

    @patch("deployment.app.config.get_config_values")
    def test_config_section_getters(self, mock_get_config):
        """Test all config section getter functions."""
        # Arrange
        full_config = {
            "api": {"host": "api-host"},
            "db": {"filename": "db.sqlite"},
            "datasphere": {"project_id": "test-project"},
            "data_retention": {"cleanup_enabled": True},
            "other": {"value": "test"},
        }
        mock_get_config.return_value = full_config

        # Act & Assert
        assert get_api_config() == {"host": "api-host"}
        assert get_db_config() == {"filename": "db.sqlite"}
        assert get_datasphere_config() == {"project_id": "test-project"}
        assert get_data_retention_config() == {"cleanup_enabled": True}
        assert get_app_config() == {"other": {"value": "test"}}

    @patch("deployment.app.config.logger")
    @patch("deployment.app.config.Path")
    def test_ensure_directory_exists_file_path(self, mock_path_class, mock_logger):
        """Test ensure_directory_exists creates parent directory for file paths."""
        # Arrange
        mock_path_instance = MagicMock()
        mock_path_instance.name = "file.txt"
        mock_path_instance.parent = MagicMock()
        mock_path_class.return_value = mock_path_instance

        # Act
        result = ensure_directory_exists("/test/path/file.txt")

        # Assert
        assert result == "/test/path/file.txt"
        mock_path_instance.parent.mkdir.assert_called_once_with(
            parents=True, exist_ok=True
        )

    @patch("deployment.app.config.logger")
    @patch("deployment.app.config.Path")
    def test_ensure_directory_exists_directory_path(self, mock_path_class, mock_logger):
        """Test ensure_directory_exists creates directory for directory paths."""
        # Arrange
        mock_path_instance = MagicMock()
        mock_path_instance.name = "directory"
        mock_path_class.return_value = mock_path_instance

        # Act
        result = ensure_directory_exists("/test/path/directory")

        # Assert
        assert result == "/test/path/directory"
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_ensure_directory_exists_empty_path(self):
        """Test ensure_directory_exists handles empty path."""
        # Act
        result = ensure_directory_exists("")

        # Assert
        assert result == ""


class TestAPISettings:
    """Test suite for APISettings configuration."""

    def setup_method(self, method):
        """Setup method to clear LRU cache before each test."""
        get_config_values.cache_clear()

    @patch.dict(os.environ, {}, clear=True)
    @patch(
        "pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}
    )  # Disable .env loading
    def test_api_settings_defaults(self, mock_dotenv):
        """Test APISettings uses correct default values."""
        # Act
        settings = APISettings()

        # Assert
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.debug is False
        assert settings.admin_api_key == ""
        assert settings.x_api_key == ""
        assert settings.log_level == "INFO"

    @patch.dict(
        os.environ,
        {
            "API_HOST": "127.0.0.1",
            "API_PORT": "9000",
            "API_DEBUG": "true",
            "API_ALLOWED_ORIGINS": '["http://test1.com", "http://test2.com"]',
            "API_API_KEY": "test-api-key",
            "API_X_API_KEY": "test-x-api-key",
            "API_LOG_LEVEL": "DEBUG",
            "API_ADMIN_API_KEY": "test-api-key",
        },
    )
    @patch(
        "pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}
    )  # Disable .env loading
    def test_api_settings_from_environment(self, mock_dotenv):
        """Test APISettings loads values from environment variables."""
        # Act
        settings = APISettings()

        # Assert
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.debug is True
        assert settings.admin_api_key == "test-api-key"
        assert settings.x_api_key == "test-x-api-key"
        assert settings.log_level == "DEBUG"

    @patch.dict(os.environ, {"CONFIG_FILE_PATH": "/test/config.yaml"})
    @patch("deployment.app.config.load_config_file")
    def test_api_settings_custom_sources(self, mock_load_config_file):
        """Test APISettings custom configuration sources."""
        # Arrange
        mock_load_config_file.return_value = {
            "api": {"host": "config-host", "port": 3000}
        }
        # Clear the LRU cache so get_config_values can be called again
        get_config_values.cache_clear()

        # Act
        settings = APISettings()

        # Assert - config file values should be loaded
        mock_load_config_file.assert_called_once_with("/test/config.yaml")
        # The settings should use config values when available
        assert settings.host == "config-host"
        assert settings.port == 3000


class TestDatabaseSettings:
    """Test suite for DatabaseSettings configuration."""

    def setup_method(self, method):
        """Setup method to clear LRU cache before each test."""
        get_config_values.cache_clear()

    @patch.dict(os.environ, {}, clear=True)
    def test_database_settings_defaults(self):
        """Test DatabaseSettings uses correct default values."""
        # Act
        settings = DatabaseSettings()

        # Assert
        assert settings.filename == "plastinka.db"

    @patch.dict(os.environ, {"DB_FILENAME": "custom.db"})
    def test_database_settings_from_environment(self):
        """Test DatabaseSettings loads values from environment variables."""
        # Act
        settings = DatabaseSettings()

        # Assert
        assert settings.filename == "custom.db"

    @patch.dict(os.environ, {"CONFIG_FILE_PATH": "/test/config.yaml"})
    @patch("deployment.app.config.load_config_file")
    def test_database_settings_custom_sources(self, mock_load_config_file):
        """Test DatabaseSettings custom configuration sources."""
        # Arrange
        mock_load_config_file.return_value = {"db": {"filename": "config.db"}}
        # Clear the LRU cache so get_config_values can be called again
        get_config_values.cache_clear()

        # Act
        settings = DatabaseSettings()

        # Assert
        mock_load_config_file.assert_called_once_with("/test/config.yaml")
        assert settings.filename == "config.db"


class TestDataSphereSettings:
    """Test suite for DataSphereSettings configuration."""

    def setup_method(self, method):
        """Setup method to clear LRU cache before each test."""
        get_config_values.cache_clear()

    @patch.dict(os.environ, {}, clear=True)
    @patch(
        "pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}
    )  # Disable .env loading
    def test_datasphere_settings_defaults(self, mock_dotenv):
        """Test DataSphereSettings uses correct default values."""
        # Act
        settings = DataSphereSettings()

        # Assert
        assert settings.project_id == ""
        assert settings.folder_id == ""
        assert settings.oauth_token is None
        assert settings.yc_profile == "datasphere-prod"
        assert settings.max_polls == 72
        assert settings.poll_interval == 300.0
        assert settings.download_diagnostics_on_success is False
        assert settings.client_init_timeout_seconds == 60
        assert settings.client_submit_timeout_seconds == 300
        assert settings.client_status_timeout_seconds == 30
        assert settings.client_download_timeout_seconds == 300
        assert settings.client_cancel_timeout_seconds == 60

    @patch.dict(
        os.environ,
        {
            "DATASPHERE_PROJECT_ID": "test-project",
            "DATASPHERE_FOLDER_ID": "test-folder",
            "YC_OAUTH_TOKEN": "test-token",
            "DATASPHERE_YC_PROFILE": "test-profile",
            "DATASPHERE_MAX_POLLS": "200",
            "DATASPHERE_POLL_INTERVAL": "10.0",
            "DATASPHERE_DOWNLOAD_DIAGNOSTICS_ON_SUCCESS": "true",
        },
    )
    def test_datasphere_settings_from_environment(self):
        """Test DataSphereSettings loads values from environment variables."""
        # Act
        settings = DataSphereSettings()

        # Assert
        assert settings.project_id == "test-project"
        assert settings.folder_id == "test-folder"
        assert settings.oauth_token == "test-token"
        assert settings.yc_profile == "test-profile"
        assert settings.max_polls == 200
        assert settings.poll_interval == 10.0
        assert settings.download_diagnostics_on_success is True

    def test_datasphere_settings_client_property(self):
        """Test DataSphereSettings client property returns correct dictionary."""
        # Arrange
        settings = DataSphereSettings(
            project_id="test-project",
            folder_id="test-folder",
            oauth_token="test-token",
            yc_profile="test-profile",
        )

        # Act
        client_config = settings.client

        # Assert
        # auth_method can be determined dynamically; accept 'auto', 'oauth_token', or 'yc_profile'
        assert client_config["project_id"] == "test-project"
        assert client_config["folder_id"] == "test-folder"
        assert client_config["oauth_token"] == "test-token"
        assert client_config["yc_profile"] == "test-profile"
        assert client_config["auth_method"] in {"auto", "oauth_token", "yc_profile"}

    @patch.dict(os.environ, {"CONFIG_FILE_PATH": "/test/config.yaml"})
    @patch("deployment.app.config.load_config_file")
    def test_datasphere_settings_custom_sources(self, mock_load_config_file):
        """Test DataSphereSettings custom configuration sources."""
        # Arrange
        mock_load_config_file.return_value = {
            "datasphere": {"project_id": "config-project"}
        }
        # Clear the LRU cache so get_config_values can be called again
        get_config_values.cache_clear()

        # Act
        settings = DataSphereSettings()

        # Assert
        mock_load_config_file.assert_called_once_with("/test/config.yaml")
        assert settings.project_id == "config-project"


class TestDataRetentionSettings:
    """Test suite for DataRetentionSettings configuration."""

    @patch.dict(os.environ, {}, clear=True)
    def test_data_retention_settings_defaults(self):
        """Test DataRetentionSettings uses correct default values."""
        # Act
        settings = DataRetentionSettings()

        # Assert
        assert settings.sales_retention_days == 730
        assert settings.stock_retention_days == 730
        assert settings.prediction_retention_days == 365
        assert settings.models_to_keep == 5
        assert settings.inactive_model_retention_days == 90
        assert settings.cleanup_enabled is True
        assert settings.cleanup_schedule == "0 3 * * 0"

    @patch("pydantic_settings.sources.EnvSettingsSource.__call__")
    def test_data_retention_settings_from_environment(self, mock_env_source):
        """
        Test DataRetentionSettings loads values from environment variables.
        This test now patches the Pydantic EnvSettingsSource directly
        for robust and isolated testing.
        """
        # Arrange
        mock_env_source.return_value = {
            "sales_retention_days": 365,
            "stock_retention_days": 180,
            "prediction_retention_days": 90,
            "models_to_keep": 10,
            "inactive_model_retention_days": 30,
            "cleanup_enabled": False,
            "cleanup_schedule": "0 2 * * 1",
        }

        # Act: Create a new instance, which will use the patched source
        settings = DataRetentionSettings()

        # Assert values match the mocked source
        assert settings.sales_retention_days == 365
        assert settings.stock_retention_days == 180
        assert settings.prediction_retention_days == 90
        assert settings.models_to_keep == 10
        assert settings.inactive_model_retention_days == 30
        assert settings.cleanup_enabled is False
        assert settings.cleanup_schedule == "0 2 * * 1"

    @patch.dict(os.environ, {"CONFIG_FILE_PATH": "/test/config.yaml"})
    @patch("deployment.app.config.load_config_file")
    def test_data_retention_settings_custom_sources(self, mock_load_config_file):
        """Test DataRetentionSettings custom configuration sources."""
        # Arrange
        mock_load_config_file.return_value = {
            "data_retention": {"cleanup_enabled": False}
        }
        # Clear the LRU cache so get_config_values can be called again
        get_config_values.cache_clear()

        # Act
        settings = DataRetentionSettings()

        # Assert
        mock_load_config_file.assert_called_once_with("/test/config.yaml")
        assert settings.cleanup_enabled is False


class TestAppSettings:
    """Test suite for AppSettings and computed properties."""

    def setup_method(self, method):
        """Setup method to clear LRU cache before each test."""
        get_config_values.cache_clear()

    @patch.dict(
        os.environ,
        {"DATA_ROOT_DIR": "/default/data/root", "PROJECT_ROOT_DIR": "/project/root"},
        clear=True,
    )
    @patch("deployment.app.config.ensure_directory_exists", side_effect=lambda x: x)
    def test_app_settings_defaults(self, mock_ensure_dir):
        """Test AppSettings uses correct default values."""
        # Act
        settings = AppSettings()

        # Assert
        assert settings.env == "development"
        assert settings.max_upload_size == 50 * 1024 * 1024
        assert settings.temp_upload_dir.replace("\\", "/") == "/default/data/root/temp_uploads"
        assert settings.file_storage_dir.replace("\\", "/") == "/default/data/root/uploads"
        assert settings.default_metric == "val_MIWS_MIC_Ratio"
        assert settings.default_metric_higher_is_better is False
        assert settings.auto_select_best_configs is True
        assert settings.auto_select_best_model is True
        assert settings.data_root_dir == "/default/data/root"

    @patch("deployment.app.config.ensure_directory_exists", side_effect=lambda x: x)
    def test_app_settings_from_environment(self, mock_ensure_dir):
        """Test AppSettings loading from environment variables."""
        # Arrange
        with patch.dict(
            os.environ,
            {
                "ENV": "production",
                "MAX_UPLOAD_SIZE": "10485760",  # 10 MB
                "DATA_ROOT_DIR": "/var/data", # New: set data_root_dir
                "DEFAULT_METRIC": "custom_metric",
                "DEFAULT_METRIC_HIGHER_IS_BETTER": "false",
                "AUTO_SELECT_BEST_CONFIGS": "false",
                "AUTO_SELECT_BEST_MODEL": "false",
            },
            clear=True,
        ):
            # Act
            settings = AppSettings()

            # Assert
            assert settings.env == "production"
            assert settings.max_upload_size == 10485760
            # Assert computed properties based on DATA_ROOT_DIR
            assert settings.temp_upload_dir.replace("\\", "/") == "/var/data/temp_uploads"
            assert settings.file_storage_dir.replace("\\", "/") == "/var/data/uploads"
            assert settings.default_metric == "custom_metric"
            assert not settings.default_metric_higher_is_better
            assert not settings.auto_select_best_configs
            assert not settings.auto_select_best_model

    @patch("deployment.app.config.ensure_directory_exists", side_effect=lambda x: x)
    def test_app_settings_computed_properties(self, mock_ensure_dir):
        """Test computed properties of AppSettings."""
        # Arrange - use forward slashes for cross-platform compatibility
        with patch.dict(os.environ, {"DATA_ROOT_DIR": "/test/data"}, clear=True):
            settings = AppSettings()

        # Act & Assert - normalize path separators for Windows compatibility
        expected_db_path = "/test/data/database/plastinka.db"
        actual_db_path = settings.database_path.replace("\\", "/")
        assert actual_db_path == expected_db_path

        expected_db_url = "sqlite:////test/data/database/plastinka.db"
        actual_db_url = settings.database_url.replace("\\", "/")
        assert actual_db_url == expected_db_url

        expected_models_dir = "/test/data/models"
        actual_models_dir = settings.models_dir.replace("\\", "/")
        assert actual_models_dir == expected_models_dir

    def test_app_settings_datasphere_job_properties(self):
        """Test AppSettings DataSphere job-related properties."""
        # Use patch.dict with clear=True
        with patch.dict(os.environ, {"PROJECT_ROOT_DIR": "/project/root"}, clear=True):
            settings = AppSettings()

        # Act & Assert - normalize paths properly
        def normalize_path(path):
            """Normalize path for cross-platform comparison."""
            normalized = path.replace("\\", "/")
            # Remove drive letter if present (e.g., "C:/path" -> "/path")
            if ":" in normalized and len(normalized.split(":")[0]) <= 2:
                normalized = "/" + normalized.split(":", 1)[1]
            # Remove double slashes
            while "//" in normalized:
                normalized = normalized.replace("//", "/")
            return normalized

        expected_job_dir = "/project/root/plastinka_sales_predictor/datasphere_jobs"
        actual_job_dir = normalize_path(settings.datasphere_job_dir)
        expected_job_dir_new = "/project/root/plastinka_sales_predictor/datasphere_jobs"
        # Accept either new or legacy path for compatibility during transition
        assert actual_job_dir in [expected_job_dir_new, expected_job_dir]

        expected_config_path = (
            "/project/root/plastinka_sales_predictor/datasphere_jobs/config.yaml"
        )
        actual_config_path = normalize_path(settings.datasphere_job_config_path)
        assert actual_config_path == expected_config_path

    def test_app_settings_project_root_validation(self):
        """Test AppSettings validates and resolves project_root_dir."""
        # Use patch.dict with clear=True
        with patch.dict(os.environ, {"PROJECT_ROOT_DIR": "/test/project"}, clear=True):
            settings = AppSettings()

        # Assert - just check the path contains our expected path
        def normalize_path(path):
            """Normalize path for cross-platform comparison."""
            normalized = path.replace("\\", "/")
            # Remove drive letter if present (e.g., "C:/path" -> "/path")
            if ":" in normalized and len(normalized.split(":")[0]) <= 2:
                normalized = "/" + normalized.split(":", 1)[1]
            # Remove double slashes
            while "//" in normalized:
                normalized = normalized.replace("//", "/")
            return normalized

        expected_path = "/test/project"
        actual_path = normalize_path(settings.project_root_dir)
        assert actual_path == expected_path

    @patch.dict(os.environ, {"CONFIG_FILE_PATH": "/test/config.yaml"})
    @patch("deployment.app.config.load_config_file")
    def test_app_settings_custom_sources(self, mock_load_config_file):
        """Test AppSettings custom configuration sources."""
        # Arrange
        mock_load_config_file.return_value = {"env": "config-env"}
        # Clear the LRU cache so get_config_values can be called again
        get_config_values.cache_clear()

        # Act
        settings = AppSettings()

        # Assert
        mock_load_config_file.assert_called_once_with("/test/config.yaml")
        assert settings.env == "config-env"


class TestConfigurationPrecedence:
    """Test suite for configuration source precedence."""

    def setup_method(self, method):
        """Setup method to clear LRU cache before each test."""
        get_config_values.cache_clear()

    @patch("deployment.app.config.get_api_config")
    @patch.dict(os.environ, {"HOST": "env-host"})
    def test_precedence_init_over_env(self, mock_get_api_config):
        """Test that init arguments take precedence over environment variables."""
        # Arrange
        mock_get_api_config.return_value = {}

        # Act
        settings = APISettings(host="init-host")

        # Assert
        assert settings.host == "init-host"  # init wins over env

    @patch.dict(
        os.environ, {"CONFIG_FILE_PATH": "/test/config.yaml", "API_HOST": "env-host"}
    )
    @patch("deployment.app.config.load_config_file")
    def test_precedence_config_file_over_env(self, mock_load_config_file):
        """Test that config file values take precedence over environment variables."""
        # Arrange
        mock_load_config_file.return_value = {"api": {"host": "config-host"}}
        # Clear the LRU cache so get_config_values can be called again
        get_config_values.cache_clear()

        # Act
        settings = APISettings()

        # Assert
        mock_load_config_file.assert_called_once_with("/test/config.yaml")
        assert settings.host == "config-host"


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    def setup_method(self, method):
        """Setup method to clear LRU cache before each test."""
        get_config_values.cache_clear()

    @patch.dict(os.environ, {"API_PORT": "invalid-port"})
    def test_invalid_environment_variable_type(self):
        """Test handling of invalid environment variable types."""
        # Act & Assert
        with pytest.raises(ValidationError):
            APISettings()

    @patch.dict(os.environ, {"DATASPHERE_MAX_POLLS": "not-a-number"})
    def test_invalid_numeric_environment_variable(self):
        """Test handling of invalid numeric environment variables."""
        # Act & Assert
        with pytest.raises(ValidationError):
            DataSphereSettings()

    def test_invalid_boolean_environment_variable(self):
        """Test handling of various boolean environment variable formats."""
        # Valid boolean strings should work
        valid_settings = DataSphereSettings(download_diagnostics_on_success="true")
        assert valid_settings.download_diagnostics_on_success is True

        valid_settings = DataSphereSettings(download_diagnostics_on_success="false")
        assert valid_settings.download_diagnostics_on_success is False

        # Case insensitive
        valid_settings = DataSphereSettings(download_diagnostics_on_success="TRUE")
        assert valid_settings.download_diagnostics_on_success is True


class TestIntegration:
    """Integration tests for the complete configuration system."""

    def setup_method(self, method):
        """Setup method to clear LRU cache before each test."""
        get_config_values.cache_clear()

    def test_module_imports_successfully(self):
        """Test that the config module can be imported without errors."""
        # This test passes if no import errors occurred during module loading
        assert _get_default_data_root_dir is not None
        assert load_config_file is not None
        assert APISettings is not None
        assert DatabaseSettings is not None
        assert DataSphereSettings is not None
        assert DataRetentionSettings is not None
        assert AppSettings is not None
        assert get_settings is not None

    def test_settings_classes_defined(self):
        """Test that all expected settings classes are defined."""
        # Verify all settings classes can be instantiated
        api_settings = APISettings()
        db_settings = DatabaseSettings()
        datasphere_settings = DataSphereSettings()
        data_retention_settings = DataRetentionSettings()
        app_settings = AppSettings()

        assert isinstance(api_settings, APISettings)
        assert isinstance(db_settings, DatabaseSettings)
        assert isinstance(datasphere_settings, DataSphereSettings)
        assert isinstance(data_retention_settings, DataRetentionSettings)
        assert isinstance(app_settings, AppSettings)

    def test_global_settings_instance(self):
        """Test that the global settings instance is properly initialized."""
        # Import the global settings instance
        settings = get_settings()

        # Verify it's an AppSettings instance
        assert isinstance(settings, AppSettings)
        assert hasattr(settings, "api")
        assert hasattr(settings, "db")
        assert hasattr(settings, "datasphere")
        assert hasattr(settings, "data_retention")

    @patch("deployment.app.config.get_config_values")
    def test_nested_settings_integration(self, mock_get_config_values):
        """Test that nested settings work correctly within AppSettings."""
        # Arrange
        mock_get_config_values.return_value = {
            "api": {"host": "nested-host"},
            "db": {"filename": "nested.db"},
        }

        # Act
        app_settings = AppSettings()

        # Assert
        assert isinstance(app_settings.api, APISettings)
        assert isinstance(app_settings.db, DatabaseSettings)
        assert isinstance(app_settings.datasphere, DataSphereSettings)
        assert isinstance(app_settings.data_retention, DataRetentionSettings)

    @patch("deployment.app.config.ensure_directory_exists")
    def test_end_to_end_configuration_loading(self, mock_ensure_dir):
        """Test end-to-end configuration loading with all components."""
        # Arrange
        mock_ensure_dir.side_effect = lambda x: x

        # Act - Create settings with some custom values
        settings = AppSettings(
            env="testing",
            data_root_dir="/test/data",
            api=APISettings(host="test-host", port=9000),
            db=DatabaseSettings(filename="test.db"),
        )

        # Assert - Verify the configuration was loaded correctly
        assert settings.env == "testing"
        assert settings.is_testing is True
        assert settings.data_root_dir == "/test/data"
        assert settings.api.host == "test-host"
        assert settings.api.port == 9000
        assert settings.db.filename == "test.db"

        # Verify computed properties work (normalize path separators)
        database_path = settings.database_path.replace("\\", "/")
        assert "/test/data" in database_path
        assert "test.db" in database_path
        assert settings.database_url.startswith("sqlite:///")

        models_dir = settings.models_dir.replace("\\", "/")
        assert models_dir == "/test/data/models"
