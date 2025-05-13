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
    get_app_config,
    APISettings,
    DatabaseSettings,
    DataSphereSettings,
    AppSettings
)

class TestConfigModule(unittest.TestCase):
    """Test cases for the config module."""
    
    def setUp(self):
        # Clear the lru_cache for get_config_values
        get_config_values.cache_clear()
        
        # Store original environment variables
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Clear the lru_cache for get_config_values again
        get_config_values.cache_clear()
    
    def test_datasphere_settings(self):
        """Test DataSphere settings initialization and properties."""
        # Set environment variables
        os.environ['DATASPHERE_PROJECT_ID'] = 'test-project-id'
        os.environ['DATASPHERE_FOLDER_ID'] = 'test-folder-id'
        os.environ['DATASPHERE_OAUTH_TOKEN'] = 'test-token'
        os.environ['DATASPHERE_JOB_CONFIG'] = 'test/job_config.yaml'
        os.environ['DATASPHERE_OUTPUT_DIR'] = 'test/output'
        
        # Initialize settings from environment variables
        settings = DataSphereSettings()
        
        # Test values from environment variables
        self.assertEqual(settings.project_id, 'test-project-id')
        self.assertEqual(settings.folder_id, 'test-folder-id')
        self.assertEqual(settings.oauth_token, 'test-token')
        # Check nested settings from env vars
        self.assertEqual(settings.train_job.job_config_path, 'test/job_config.yaml') # Check renamed field
        self.assertEqual(settings.train_job.input_dir, 'test/input') # Assuming an env var DATASPHERE_TRAIN_JOB_INPUT_DIR=test/input
        self.assertEqual(settings.train_job.output_dir, 'test/output') # Assuming an env var DATASPHERE_TRAIN_JOB_OUTPUT_DIR=test/output
        
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

class TestDataSphereSettings(unittest.TestCase):
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