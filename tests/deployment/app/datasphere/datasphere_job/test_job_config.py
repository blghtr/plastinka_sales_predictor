import os
import pytest
import yaml # For loading YAML file
from pathlib import Path

# Define the expected path relative to the project root
JOB_REQ_PATH = "plastinka_sales_predictor/datasphere_job/requirements.txt"

# Create an absolute path to the config file
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
CONFIG_YAML_PATH = os.path.join(project_root, "tests", "configs", "datasphere", "job_config.yaml")


def test_config_yaml_file_exists():
    """Check if the job config YAML file exists."""
    assert os.path.exists(CONFIG_YAML_PATH), f"Config YAML not found at {CONFIG_YAML_PATH}"
    # This test will fail until the file is created

def test_config_yaml_structure_and_keys():
    """Check the basic structure and essential keys in config.yaml."""
    if not os.path.exists(CONFIG_YAML_PATH):
        pytest.skip(f"Skipping content check as file {CONFIG_YAML_PATH} doesn't exist.")

    try:
        with open(CONFIG_YAML_PATH, 'r') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        pytest.fail(f"Failed to parse {CONFIG_YAML_PATH}: {e}")
    except Exception as e:
         pytest.fail(f"Failed to read {CONFIG_YAML_PATH}: {e}")

    assert isinstance(config_data, dict), "Config root should be a dictionary."
    
    # Check for expected top-level keys (adjust based on actual DS config needs)
    assert 'job_name' in config_data
    assert 'resources' in config_data
    assert 'environment' in config_data
    assert 'parameters' in config_data

    # Check nested keys (examples)
    assert isinstance(config_data['resources'], dict)
    assert 'instance_type' in config_data['resources']
    
    assert isinstance(config_data['environment'], dict)
    # Example: Check if DB creds can be set via env vars in config
    assert 'DB_HOST' in config_data['environment' ]
    assert 'DB_USER' in config_data['environment']
    
    assert isinstance(config_data['parameters'], dict)
    # Check if script args are defined as parameters
    assert 'db_host' in config_data['parameters'] # Corresponds to --db-host arg
    assert 'db_name' in config_data['parameters']
    # This test will fail until the file is created with the correct structure 