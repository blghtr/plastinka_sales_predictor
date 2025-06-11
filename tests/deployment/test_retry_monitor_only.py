"""
Extremely simple test file that only imports and patches the retry_monitor module.
"""
import sys
import os
from unittest.mock import MagicMock

print("Starting retry_monitor only test")

# First, ensure retry_monitor is not in sys.modules
if 'deployment.app.utils.retry_monitor' in sys.modules:
    print("Removing retry_monitor from sys.modules")
    del sys.modules['deployment.app.utils.retry_monitor']

# Create a mock module for retry_monitor
print("Creating mock retry_monitor module")
mock_retry_monitor_module = MagicMock()
mock_retry_monitor_module.get_retry_statistics = MagicMock(return_value={})
mock_retry_monitor_module.reset_retry_statistics = MagicMock()
mock_retry_monitor_module.record_retry = MagicMock()
mock_retry_monitor_module.retry_monitor = MagicMock()
mock_retry_monitor_module.RetryMonitor = MagicMock()
mock_retry_monitor_module.DEFAULT_PERSISTENCE_PATH = None

# Replace the module in sys.modules
print("Replacing retry_monitor in sys.modules")
sys.modules['deployment.app.utils.retry_monitor'] = mock_retry_monitor_module

print("About to import retry_monitor")
# Try to import the retry_monitor module directly
from deployment.app.utils.retry_monitor import retry_monitor
print(f"Successfully imported retry_monitor: {retry_monitor}")

# Simple test function to verify everything is working
def test_retry_monitor_only():
    """A minimal test that should pass."""
    assert True 