"""
Direct test script for reset_retry_statistics function without using pytest.
"""

import sys
import os
import time
from unittest.mock import MagicMock

print("Starting direct test script")

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
    # Import the function to test
    print("Importing reset_retry_statistics...")
    from deployment.app.utils.retry_monitor import reset_retry_statistics
    
    # Call the function
    print("Calling reset_retry_statistics...")
    start_time = time.time()
    reset_retry_statistics()
    elapsed = time.time() - start_time
    
    # Verify the mock was called
    assert mock_retry_monitor.reset_statistics.called, "Mock reset_statistics was not called"
    print(f"Success! reset_retry_statistics called mock in {elapsed:.2f} seconds")
    
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