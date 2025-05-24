import os
import unittest
from unittest import mock
import tempfile
import json
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
        # Corrected environment variable name for job_config_path
        os.environ['DATASPHERE_TRAIN_JOB_JOB_CONFIG_PATH'] = 'test/job_config.yaml'
        os.environ['DATASPHERE_TRAIN_JOB_OUTPUT_DIR'] = 'test/output'
        os.environ['DATASPHERE_TRAIN_JOB_INPUT_DIR'] = 'test/input'
        
        # Initialize settings from environment variables
        settings = DataSphereSettings()
        
        # Test values from environment variables
        self.assertEqual(settings.project_id, 'test-project-id')
        self.assertEqual(settings.folder_id, 'test-folder-id')
        self.assertEqual(settings.oauth_token, 'test-token')
        # Check nested settings from env vars
        self.assertEqual(settings.train_job.job_config_path, 'test/job_config.yaml')
        self.assertEqual(settings.train_job.input_dir, 'test/input')
        self.assertEqual(settings.train_job.output_dir, 'test/output')
        
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
        
        # Test default nested values
        self.assertIsNone(settings.datasphere.train_job.job_config_path) # Default is None
        self.assertEqual(settings.datasphere.train_job.input_dir, 'deployment/data/datasphere_input')
        self.assertEqual(settings.datasphere.train_job.output_dir, 'deployment/data/datasphere_output')

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
        default_path = "deployment/data/plastinka.db" # Default for db.path

        # 1. Test Default
        get_config_values.cache_clear()
        settings = DatabaseSettings()
        self.assertEqual(settings.path, default_path)

        # 2. Test .env file only
        get_config_values.cache_clear()
        self._create_temp_env_file("DB_PATH=dotenv_db_path")
        settings = DatabaseSettings()
        self.assertEqual(settings.path, "dotenv_db_path")
        os.remove(os.path.join(self.temp_dir.name, ".env"))

        # 3. Test Environment variable only
        get_config_values.cache_clear()
        os.environ['DB_PATH'] = 'env_db_path'
        settings = DatabaseSettings()
        self.assertEqual(settings.path, "env_db_path")
        del os.environ['DB_PATH']

        # 4. Test Config file only
        get_config_values.cache_clear()
        with mock.patch.object(DatabaseSettings, '_config_loader_func', return_value={"path": "file_db_path"}):
            settings = DatabaseSettings()
            self.assertEqual(settings.path, "file_db_path")

        # 5. Test Env Var vs Config File (Config File should win)
        get_config_values.cache_clear()
        os.environ['DB_PATH'] = 'env_db_path_conflict'
        with mock.patch.object(DatabaseSettings, '_config_loader_func', return_value={"path": "file_db_path_wins"}):
            settings = DatabaseSettings()
            self.assertEqual(settings.path, "file_db_path_wins")
        del os.environ['DB_PATH']

        # 6. Test Init Kwarg (Init Kwarg should win over all)
        get_config_values.cache_clear()
        self._create_temp_env_file("DB_PATH=dotenv_db_init")
        os.environ['DB_PATH'] = 'env_db_init'
        with mock.patch.object(DatabaseSettings, '_config_loader_func', return_value={"path": "file_db_init"}):
            settings = DatabaseSettings(path="init_db_path_wins")
            self.assertEqual(settings.path, "init_db_path_wins")
        del os.environ['DB_PATH']
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
        # Check nested train_job settings
        self.assertIsNone(settings.train_job.job_config_path) # Check renamed field
        self.assertEqual(settings.train_job.input_dir, 'deployment/data/datasphere_input')
        # Check new nested output dir default
        self.assertEqual(settings.train_job.output_dir, 'deployment/data/datasphere_output')
        self.assertEqual(settings.max_polls, 120)
        self.assertEqual(settings.poll_interval, 5.0)

if __name__ == '__main__':
    unittest.main() 