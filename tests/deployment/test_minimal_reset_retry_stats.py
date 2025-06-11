"""
Minimal test file that isolates the hanging test from test_api_health.py.
"""

import pytest
from unittest.mock import patch, PropertyMock, MagicMock
import sys

# Create a mock module for retry_monitor
mock_retry_monitor = MagicMock()
mock_retry_monitor.reset_statistics = MagicMock()
mock_retry_monitor.get_statistics = MagicMock(return_value={
    "total_retries": 0,
    "successful_retries": 0,
    "exhausted_retries": 0,
    "successful_after_retry": 0,
    "high_failure_operations": [],
    "operation_stats": {},
    "exception_stats": {},
    "timestamp": "2021-01-01T00:00:00"
})

# Create a mock module
mock_module = MagicMock()
mock_module.retry_monitor = mock_retry_monitor
mock_module.reset_retry_statistics = lambda: mock_retry_monitor.reset_statistics()

# Store the original module if it exists
original_retry_monitor = sys.modules.get('deployment.app.utils.retry_monitor')

# Replace the module in sys.modules
sys.modules['deployment.app.utils.retry_monitor'] = mock_module
print("Replaced retry_monitor module with mock")

# Import the settings module
from deployment.app.config import settings

@pytest.fixture
def client():
    """A minimal client fixture."""
    from fastapi.testclient import TestClient
    from deployment.app.main import app
    return TestClient(app)

def test_reset_retry_stats_server_key_not_configured():
    """Test /health/retry-stats/reset endpoint returns 401 if server's x_api_key is None or empty."""
    # Create a mock for settings.api.x_api_key
    with patch.object(settings.api, "x_api_key", new_callable=PropertyMock) as mock_x_api_key_prop:
        mock_x_api_key_prop.return_value = None
        
        # Create a client
        from fastapi.testclient import TestClient
        from deployment.app.main import app
        client = TestClient(app)
        
        # Make the request
        response = client.post("/health/retry-stats/reset", headers={"X-API-Key": "test_x_api_key"})
        
        # Verify the response
        assert response.status_code == 401
        assert "error" in response.json()
        assert "message" in response.json()["error"]
        assert "Invalid X-API-Key" in response.json()["error"]["message"]

# Restore the original module
if original_retry_monitor:
    sys.modules['deployment.app.utils.retry_monitor'] = original_retry_monitor
    print("Restored original retry_monitor module")
else:
    del sys.modules['deployment.app.utils.retry_monitor']
    print("Removed mock retry_monitor module") 