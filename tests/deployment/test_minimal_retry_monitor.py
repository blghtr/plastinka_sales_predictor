"""
Minimal test file that tests the reset_retry_statistics function to check if that causes hanging.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import importlib

@pytest.fixture(scope="function")
def mock_retry_monitor_module(monkeypatch):
    """
    Create a comprehensive mock of the retry_monitor module with all required attributes.
    """
    # Create the mock module
    mock_module = MagicMock()
    
    # Add module-level functions
    mock_module.DEFAULT_PERSISTENCE_PATH = None
    mock_module.get_retry_statistics = MagicMock(return_value={
        "total_retries": 0,
        "successful_retries": 0,
        "exhausted_retries": 0,
        "successful_after_retry": 0,
        "high_failure_operations": [],
        "alerted_operations": [],
        "alert_thresholds": {},
        "operation_stats": {},
        "exception_stats": {},
        "timestamp": "2021-01-01T00:00:00"
    })
    mock_module.reset_retry_statistics = MagicMock(return_value={})
    
    # Replace the actual module with our mock
    monkeypatch.setitem(sys.modules, 'deployment.app.utils.retry_monitor', mock_module)
    
    # Also patch functions in health module to use our mocks
    # This is important to ensure they don't try to import the real retry_monitor
    if 'deployment.app.api.health' in sys.modules:
        del sys.modules['deployment.app.api.health']
    
    # Return the mock module for use in tests
    return mock_module

@pytest.mark.asyncio
async def test_reset_retry_stats_directly(mock_retry_monitor_module):
    """A minimal test that directly patches the reset_retry_statistics endpoint."""
    # Import the health module with our patched retry_monitor
    from deployment.app.api.health import reset_retry_stats
    
    # Create a mock for the api_key dependency
    mock_api_key = True
    
    # Call the async endpoint function directly
    result = await reset_retry_stats(api_key=mock_api_key)
    
    # Verify the mock was called
    mock_retry_monitor_module.reset_retry_statistics.assert_called_once()
    
    # Verify the response
    assert "status" in result
    assert result["status"] == "ok"
    assert result["message"] == "Retry statistics reset successfully" 