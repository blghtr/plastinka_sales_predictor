"""
Test file that specifically tests the isolation of retry_monitor.

The goal is to ensure that the retry_monitor implementation is properly isolated
and can be mocked out for tests without importing the actual module.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib

@pytest.fixture(scope="function")
def mock_retry_monitor_module(monkeypatch):
    """
    Create a comprehensive mock of the retry_monitor module with all required attributes.
    """
    # First clear any existing imports
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('deployment.app.utils.retry') or module_name == 'deployment.app.utils.retry_monitor':
            del sys.modules[module_name]
    
    # Create the mock module
    mock_module = MagicMock()
    
    # Mock RetryMonitor class
    class MockRetryMonitor:
        def __init__(self, capacity=1000, log_interval_seconds=300, persistence_file=None, save_interval_seconds=600):
            self._lock = MagicMock()
            self._persistence_file = persistence_file
            self._high_failure_operations = set()
            self._alerted_operations = set()
            self._alert_handlers = []
            self._alert_thresholds = {
                'total_failures_count': 10,
                'consecutive_failures': 3,
                'failure_rate_percent': 50,
                'response_time_ms': 3000,
                'exhausted_retries_count': 5
            }
            self._reset_stats()
        
        def _reset_stats(self):
            self._total_retries = 0
            self._successful_retries = 0
            self._exhausted_retries = 0
            self._successful_after_retry = 0
            self._operation_failures = {}
        
        def record_retry(self, operation, exception_type, exception_message, attempt, max_attempts, successful, component, duration_ms=None):
            pass
        
        def reset_statistics(self):
            self._reset_stats()
            return {}
        
        def get_statistics(self):
            return {
                'total_retries': self._total_retries,
                'successful_retries': self._successful_retries,
                'exhausted_retries': self._exhausted_retries,
                'successful_after_retry': self._successful_after_retry,
                'high_failure_operations': list(self._high_failure_operations),
                'alerted_operations': list(self._alerted_operations),
                'alert_thresholds': self._alert_thresholds.copy(),
                'operation_stats': {},
                'exception_stats': {},
                'timestamp': '2021-01-01T00:00:00'
            }
        
        def get_high_failure_operations(self):
            return self._high_failure_operations
        
        def get_alerted_operations(self):
            return self._alerted_operations
        
        def set_alert_threshold(self, threshold_name, value):
            if threshold_name in self._alert_thresholds:
                self._alert_thresholds[threshold_name] = value
        
        def register_alert_handler(self, handler):
            self._alert_handlers.append(handler)
    
    # Add RetryMonitor class to the mock module
    mock_module.RetryMonitor = MockRetryMonitor
    
    # Create mock RetryTracker class
    class MockRetryTracker:
        def __init__(self, operation_name, **kwargs):
            self.operation_name = operation_name
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    # Add RetryTracker class to the mock module
    mock_module.RetryTracker = MockRetryTracker
    
    # Create a mock instance of RetryMonitor
    mock_retry_monitor_instance = MockRetryMonitor(persistence_file=None)
    mock_module.retry_monitor = mock_retry_monitor_instance
    
    # Add module-level functions
    mock_module.DEFAULT_PERSISTENCE_PATH = None
    mock_module.record_retry = MagicMock()
    mock_module.get_retry_statistics = MagicMock(return_value=mock_retry_monitor_instance.get_statistics())
    mock_module.reset_retry_statistics = MagicMock(return_value={})
    mock_module.get_high_failure_operations = MagicMock(return_value=set())
    mock_module.set_alert_threshold = MagicMock()
    mock_module.register_alert_handler = MagicMock()
    mock_module.get_alerted_operations = MagicMock(return_value=set())
    mock_module.log_alert_handler = MagicMock()
    
    # Replace the actual module with our mock
    monkeypatch.setitem(sys.modules, 'deployment.app.utils.retry_monitor', mock_module)
    
    # Also patch the retry module to prevent circular import issues
    mock_retry = MagicMock()
    monkeypatch.setitem(sys.modules, 'deployment.app.utils.retry', mock_retry)
    
    # Ensure that other modules that might have imported retry_monitor also get the mocked version
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('deployment.app.') and module_name != 'deployment.app.utils.retry_monitor':
            module = sys.modules[module_name]
            if hasattr(module, 'retry_monitor'):
                setattr(module, 'retry_monitor', mock_module)
    
    # Return the mock module for use in tests
    return mock_module

def test_retry_monitor_isolation(mock_retry_monitor_module):
    """Test that retry_monitor is properly isolated between tests."""
    # Import the retry module which imports retry_monitor
    from deployment.app.utils import retry
    
    # Try to use the retry_monitor directly to verify it's properly mocked
    from deployment.app.utils import retry_monitor
    
    # Verify the DEFAULT_PERSISTENCE_PATH is None
    assert retry_monitor.DEFAULT_PERSISTENCE_PATH is None, "retry_monitor.DEFAULT_PERSISTENCE_PATH should be None but was: " + str(retry_monitor.DEFAULT_PERSISTENCE_PATH)
    
    # Create a tracker instance to verify the RetryTracker class works
    tracker = retry_monitor.RetryTracker("test_operation")
    assert tracker.operation_name == "test_operation"
    
    # The test passes if no errors occur during import and basic usage
    assert True

@pytest.mark.asyncio
async def test_health_endpoint_retry_stats_patching(mock_retry_monitor_module):
    """Test that health endpoint retry stats functions can be patched."""
    # Make sure any previously imported health module is cleared
    if 'deployment.app.api.health' in sys.modules:
        del sys.modules['deployment.app.api.health']
    
    # Set up the mock to return a value that matches what the health endpoint expects
    # Must include ALL fields from RetryStatsResponse model in health.py
    mock_retry_monitor_module.get_retry_statistics.return_value = {
        "total_retries": 0,
        "successful_retries": 0,
        "exhausted_retries": 0,
        "successful_after_retry": 0,
        "high_failure_operations": [],
        "alerted_operations": [],
        "alert_thresholds": {
            "total_failures_count": 10,
            "consecutive_failures": 3,
            "failure_rate_percent": 50,
            "response_time_ms": 3000,
            "exhausted_retries_count": 5
        },
        "operation_stats": {},
        "exception_stats": {},
        "timestamp": "2021-01-01T00:00:00"
    }
    
    # Import the health module which will use our mocked retry_monitor
    from deployment.app.api.health import retry_statistics, reset_retry_stats
    
    # Call the health endpoints
    stats_result = await retry_statistics(api_key=True)
    reset_result = await reset_retry_stats(api_key=True)
    
    # Verify the mock functions were called
    mock_retry_monitor_module.get_retry_statistics.assert_called_once()
    mock_retry_monitor_module.reset_retry_statistics.assert_called_once()
    
    # Verify responses have matching structure
    assert stats_result["total_retries"] == mock_retry_monitor_module.get_retry_statistics.return_value["total_retries"]
    assert stats_result["successful_retries"] == mock_retry_monitor_module.get_retry_statistics.return_value["successful_retries"]
    assert stats_result["exhausted_retries"] == mock_retry_monitor_module.get_retry_statistics.return_value["exhausted_retries"]
    assert stats_result["successful_after_retry"] == mock_retry_monitor_module.get_retry_statistics.return_value["successful_after_retry"]
    assert stats_result["high_failure_operations"] == mock_retry_monitor_module.get_retry_statistics.return_value["high_failure_operations"]
    assert stats_result["operation_stats"] == mock_retry_monitor_module.get_retry_statistics.return_value["operation_stats"]
    assert stats_result["exception_stats"] == mock_retry_monitor_module.get_retry_statistics.return_value["exception_stats"]
    assert stats_result["timestamp"] == mock_retry_monitor_module.get_retry_statistics.return_value["timestamp"]
    assert reset_result["status"] == "ok"
    assert reset_result["message"] == "Retry statistics reset successfully" 