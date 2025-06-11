"""
Direct test script for reset_retry_stats endpoint without using pytest.
"""

import sys
import os
import time
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add the project root to sys.path to fix imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
print(f"Added project root to sys.path: {project_root}")

print("Starting direct test script for reset_retry_stats endpoint")

# Create a mock for the missing misc module
mock_misc_module = MagicMock()
mock_misc_module.camel_to_snake = lambda s: s.lower()
sys.modules['deployment.app.utils.misc'] = mock_misc_module
print("Created mock for missing misc module")

# Create a mock module for retry_monitor
mock_retry_monitor = MagicMock()
mock_retry_monitor.reset_statistics = MagicMock()

# Create a mock module
mock_module = MagicMock()
mock_module.retry_monitor = mock_retry_monitor
mock_module.reset_retry_statistics = lambda: mock_retry_monitor.reset_statistics()

# Store the original module if it exists
original_retry_monitor = sys.modules.get('deployment.app.utils.retry_monitor')

# Replace the module in sys.modules
sys.modules['deployment.app.utils.retry_monitor'] = mock_module
print("Replaced retry_monitor module with mock")

try:
    # Import the settings module
    print("Importing settings...")
    from deployment.app.config import settings
    
    # Create a mock for settings.api.x_api_key
    print("Creating mock for settings.api.x_api_key...")
    original_x_api_key = settings.api.x_api_key
    settings.api.x_api_key = None
    
    # Import the FastAPI test client
    print("Importing FastAPI test client...")
    from fastapi.testclient import TestClient
    from deployment.app.main import app
    
    # Create a client
    print("Creating test client...")
    client = TestClient(app)
    
    # Make the request
    print("Making request to /health/retry-stats/reset...")
    start_time = time.time()
    response = client.post("/health/retry-stats/reset", headers={"X-API-Key": "test_x_api_key"})
    elapsed = time.time() - start_time
    
    # Verify the response
    print(f"Response received in {elapsed:.2f} seconds")
    print(f"Status code: {response.status_code}")
    print(f"Response body: {response.json()}")
    
    # Restore settings
    print("Restoring settings.api.x_api_key...")
    settings.api.x_api_key = original_x_api_key
    
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    # Restore the original module
    if original_retry_monitor:
        sys.modules['deployment.app.utils.retry_monitor'] = original_retry_monitor
        print("Restored original retry_monitor module")
    else:
        del sys.modules['deployment.app.utils.retry_monitor']
        print("Removed mock retry_monitor module")
        
    # Remove the mock misc module
    if 'deployment.app.utils.misc' in sys.modules:
        del sys.modules['deployment.app.utils.misc']
        print("Removed mock misc module") 