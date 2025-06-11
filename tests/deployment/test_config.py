import os
import unittest
from unittest import mock
import tempfile
import json
import sys
from unittest.mock import patch, MagicMock

# Create a mock module for retry_monitor
mock_retry_monitor_module = MagicMock()
mock_retry_monitor_module.get_retry_statistics = MagicMock(return_value={
    "total_retries": 0, 
    "successful_retries": 0, 
    "exhausted_retries": 0, 
    "successful_after_retry": 0,
    "high_failure_operations": [], 
    "operation_stats": {},
    "exception_stats": {}, 
    "timestamp": "2021-01-01T00:00:00"
})
mock_retry_monitor_module.reset_retry_statistics = MagicMock()
mock_retry_monitor_module.record_retry = MagicMock()
mock_retry_monitor_module.retry_monitor = MagicMock()
mock_retry_monitor_module.RetryMonitor = MagicMock()

# Store the original module if it exists
original_retry_monitor = sys.modules.get('deployment.app.utils.retry_monitor')

# Replace the module in sys.modules
sys.modules['deployment.app.utils.retry_monitor'] = mock_retry_monitor_module

from deployment.app.config import (
    load_config_file, 
    get_config_values, 
    get_api_config, 
    get_db_config,
    get_datasphere_config,
    get_data_retention_config, # Added
    get_app_config,
    APISettings,
    DatabaseSettings,
    DataSphereSettings,
    DataRetentionSettings, # Added
    AppSettings
)

class TestConfigModule(unittest.TestCase):
    """Test cases for the config module."""

    def setUp(self):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.TemporaryDirectory()
        os.chdir(self.temp_dir.name)
        
        # Clear the lru_cache for get_config_values
        get_config_values.cache_clear()
        
        # Store original environment variables
        self.original_env = os.environ.copy()
        # Clear relevant env vars for tests
        for key in list(os.environ.keys()):
            if key.startswith("API_") or key.startswith("DB_") or key.startswith("CONFIG_FILE_PATH"):
                del os.environ[key]

    def tearDown(self):
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_env)
        
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()
        
        # Clear the lru_cache for get_config_values again
        get_config_values.cache_clear()

    def _create_temp_env_file(self, content: str):
        """Helper to create a .env file in the temporary directory."""
        env_file_path = os.path.join(self.temp_dir.name, ".env")
        with open(env_file_path, "w") as f:
            f.write(content)
        return env_file_path

    def test_datasphere_settings(self):
        """Test DataSphere settings initialization and properties."""
        # Set environment variables
        os.environ['DATASPHERE_PROJECT_ID'] = 'test-project-id'
        os.environ['DATASPHERE_FOLDER_ID'] = 'test-folder-id'
        os.environ['DATASPHERE_OAUTH_TOKEN'] = 'test-token'
        
        # Initialize settings from environment variables
        settings = DataSphereSettings()
        
        # Test values from environment variables
        self.assertEqual(settings.project_id, 'test-project-id')
        self.assertEqual(settings.folder_id, 'test-folder-id')
        self.assertEqual(settings.oauth_token, 'test-token')
        # Check nested settings from env vars (only timeout setting remains)
        self.assertEqual(settings.train_job.wheel_build_timeout_seconds, 300)
        
        # Test client property
        client_config = settings.client
        self.assertEqual(client_config['project_id'], 'test-project-id')
        self.assertEqual(client_config['folder_id'], 'test-folder-id')
        self.assertEqual(client_config['oauth_token'], 'test-token')

    def test_app_settings_includes_datasphere(self):
        """Test that AppSettings includes DataSphere settings."""
        # Set environment variables
        os.environ['DATASPHERE_PROJECT_ID'] = 'test-project-id'
        os.environ['DATASPHERE_FOLDER_ID'] = 'test-folder-id'
        
        # Initialize app settings
        settings = AppSettings()
        
        # Test that datasphere settings are included
        self.assertEqual(settings.datasphere.project_id, 'test-project-id')
        self.assertEqual(settings.datasphere.folder_id, 'test-folder-id')
        
        # Test computed properties for DataSphere paths
        self.assertTrue(settings.datasphere_input_dir.endswith('datasphere_input'))
        self.assertTrue(settings.datasphere_output_dir.endswith('datasphere_output'))
        self.assertTrue(settings.datasphere_job_config_path.endswith('config.yaml'))

    def test_api_settings_precedence(self):
        """Test APISettings loading precedence: init > file > env > .env > default."""
        default_host = "0.0.0.0" # Default for api.host

        # 1. Test Default
        get_config_values.cache_clear()
        settings = APISettings()
        self.assertEqual(settings.host, default_host)

        # 2. Test .env file only
        get_config_values.cache_clear()
        self._create_temp_env_file("API_HOST=dotenv_host")
        settings = APISettings()
        self.assertEqual(settings.host, "dotenv_host")
        os.remove(os.path.join(self.temp_dir.name, ".env")) # Clean up

        # 3. Test Environment variable only
        get_config_values.cache_clear()
        os.environ['API_HOST'] = 'env_host'
        settings = APISettings()
        self.assertEqual(settings.host, "env_host")
        del os.environ['API_HOST']

        # 4. Test Config file only
        get_config_values.cache_clear()
        # Patch the _config_loader_func attribute on the class directly
        with mock.patch.object(APISettings, '_config_loader_func', return_value={"host": "file_host"}):
            settings = APISettings()
            self.assertEqual(settings.host, "file_host")

        # 5. Test .env vs Env Var (Env Var should win)
        get_config_values.cache_clear()
        self._create_temp_env_file("API_HOST=dotenv_host_conflict")
        os.environ['API_HOST'] = 'env_host_wins'
        settings = APISettings()
        self.assertEqual(settings.host, "env_host_wins")
        del os.environ['API_HOST']
        os.remove(os.path.join(self.temp_dir.name, ".env"))

        # 6. Test Env Var vs Config File (Config File should win)
        get_config_values.cache_clear()
        os.environ['API_HOST'] = 'env_host_conflict'
        with mock.patch.object(APISettings, '_config_loader_func', return_value={"host": "file_host_wins"}):
            settings = APISettings()
            self.assertEqual(settings.host, "file_host_wins")
        del os.environ['API_HOST']

        # 7. Test .env vs Config File (Config File should win)
        get_config_values.cache_clear()
        self._create_temp_env_file("API_HOST=dotenv_host_conflict_2")
        with mock.patch.object(APISettings, '_config_loader_func', return_value={"host": "file_host_wins_2"}):
            settings = APISettings()
            self.assertEqual(settings.host, "file_host_wins_2")
        os.remove(os.path.join(self.temp_dir.name, ".env"))

        # 8. Test All three (Config File should win)
        get_config_values.cache_clear()
        self._create_temp_env_file("API_HOST=dotenv_host_all")
        os.environ['API_HOST'] = 'env_host_all'
        with mock.patch.object(APISettings, '_config_loader_func', return_value={"host": "file_host_all_wins"}):
            settings = APISettings()
            self.assertEqual(settings.host, "file_host_all_wins")
        del os.environ['API_HOST']
        os.remove(os.path.join(self.temp_dir.name, ".env"))

        # 9. Test Init Kwarg (Init Kwarg should win over all)
        get_config_values.cache_clear()
        self._create_temp_env_file("API_HOST=dotenv_host_init")
        os.environ['API_HOST'] = 'env_host_init'
        # For init_kwarg test, the _config_loader_func mock is still needed to ensure init has higher precedence
        with mock.patch.object(APISettings, '_config_loader_func', return_value={"host": "file_host_init"}):
            settings = APISettings(host="init_host_wins")
            self.assertEqual(settings.host, "init_host_wins")
        del os.environ['API_HOST']
        os.remove(os.path.join(self.temp_dir.name, ".env"))

    def test_database_settings_precedence(self):
        """Test DatabaseSettings loading precedence: init > file > env > .env > default."""
        # Database path is now computed from AppSettings, test filename only
        default_filename = "plastinka.db" # Default for db.filename

        # 1. Test Default
        get_config_values.cache_clear()
        settings = DatabaseSettings()
        self.assertEqual(settings.filename, default_filename)

        # 2. Test .env file only
        get_config_values.cache_clear()
        self._create_temp_env_file("DB_FILENAME=dotenv_db_filename")
        settings = DatabaseSettings()
        self.assertEqual(settings.filename, "dotenv_db_filename")
        os.remove(os.path.join(self.temp_dir.name, ".env"))

        # 3. Test Environment variable only
        get_config_values.cache_clear()
        os.environ['DB_FILENAME'] = 'env_db_filename'
        settings = DatabaseSettings()
        self.assertEqual(settings.filename, "env_db_filename")
        del os.environ['DB_FILENAME']

        # 4. Test Config file only
        get_config_values.cache_clear()
        with mock.patch.object(DatabaseSettings, '_config_loader_func', return_value={"filename": "file_db_filename"}):
            settings = DatabaseSettings()
            self.assertEqual(settings.filename, "file_db_filename")

        # 5. Test Env Var vs Config File (Config File should win)
        get_config_values.cache_clear()
        os.environ['DB_FILENAME'] = 'env_db_filename_conflict'
        with mock.patch.object(DatabaseSettings, '_config_loader_func', return_value={"filename": "file_db_filename_wins"}):
            settings = DatabaseSettings()
            self.assertEqual(settings.filename, "file_db_filename_wins")
        del os.environ['DB_FILENAME']

        # 6. Test Init Kwarg (Init Kwarg should win over all)
        get_config_values.cache_clear()
        self._create_temp_env_file("DB_FILENAME=dotenv_db_init")
        os.environ['DB_FILENAME'] = 'env_db_init'
        with mock.patch.object(DatabaseSettings, '_config_loader_func', return_value={"filename": "file_db_init"}):
            settings = DatabaseSettings(filename="init_db_filename_wins")
            self.assertEqual(settings.filename, "init_db_filename_wins")
        del os.environ['DB_FILENAME']
        os.remove(os.path.join(self.temp_dir.name, ".env"))


class TestDataSphereSettings(unittest.TestCase):
    def setUp(self):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.TemporaryDirectory()
        os.chdir(self.temp_dir.name)
        get_config_values.cache_clear()
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()): # Clear relevant keys
            if key.startswith("DATASPHERE_"):
                del os.environ[key]

    def tearDown(self): # Added tearDown
        os.environ.clear()
        os.environ.update(self.original_env)
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()
        get_config_values.cache_clear()

    def test_datasphere_settings_defaults(self):
        settings = DataSphereSettings()
        self.assertEqual(settings.project_id, "")
        self.assertEqual(settings.folder_id, "")
        self.assertIsNone(settings.oauth_token)
        self.assertIsNone(settings.yc_profile)
        # Check nested train_job settings (only timeout remains)
        self.assertEqual(settings.train_job.wheel_build_timeout_seconds, 300)
        self.assertEqual(settings.max_polls, 120)
        self.assertEqual(settings.poll_interval, 5.0)

# Stop patches at the end of the file
def tearDownModule():
    # Restore the original retry_monitor module if it existed
    if original_retry_monitor:
        sys.modules['deployment.app.utils.retry_monitor'] = original_retry_monitor
    else:
        # If there was no original module, remove our mock
        if 'deployment.app.utils.retry_monitor' in sys.modules:
            del sys.modules['deployment.app.utils.retry_monitor']

if __name__ == '__main__':
    unittest.main() 