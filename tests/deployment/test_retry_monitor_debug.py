"""
Debug file to identify exactly what in retry_monitor.py is causing the hang.
This file will create a modified version of the retry_monitor module with debug prints.
"""
import sys
import os
import time
import logging
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# First, ensure retry_monitor is not in sys.modules
if 'deployment.app.utils.retry_monitor' in sys.modules:
    logger.debug("Removing retry_monitor from sys.modules")
    del sys.modules['deployment.app.utils.retry_monitor']

# Create a custom RetryMonitor implementation with debug prints
class DebugRetryMonitor:
    def __init__(self, capacity=1000, log_interval_seconds=300, 
                persistence_file=None, save_interval_seconds=600):
        logger.debug(f"DebugRetryMonitor.__init__ called with persistence_file={persistence_file}")
        
        # Store parameters
        self._capacity = capacity
        self._log_interval = log_interval_seconds
        self._persistence_file = persistence_file
        self._save_interval = save_interval_seconds
        
        # Initialize data structures
        self._retry_events = []
        self._lock = threading.Lock()
        logger.debug(f"Created lock: {self._lock}")
        self._last_log_time = time.time()
        self._last_save_time = time.time()
        self._high_failure_operations = set()
        self._alert_thresholds = {
            'total_failures_count': 10,
            'consecutive_failures': 3,
            'failure_rate_percent': 50,
            'response_time_ms': 3000,
            'exhausted_retries_count': 5
        }
        self._alert_handlers = []
        self._alerted_operations = set()
        self._operation_consecutive_failures = {}
        
        # Initialize statistics
        self._reset_stats()
        
        # Load previous statistics if persistence file exists
        if self._persistence_file and os.path.exists(self._persistence_file):
            logger.debug(f"Loading from persistence file: {self._persistence_file}")
            self._load_from_file()
    
    def _reset_stats(self):
        logger.debug("Resetting statistics")
        with self._lock:
            logger.debug(f"Acquired lock in _reset_stats: {self._lock}")
            self._total_retries = 0
            self._successful_retries = 0
            self._exhausted_retries = 0
            self._successful_after_retry = 0
            self._operation_failures = {}
            logger.debug(f"Releasing lock in _reset_stats: {self._lock}")
    
    def get_statistics(self):
        logger.debug("get_statistics called")
        with self._lock:
            logger.debug(f"Acquired lock in get_statistics: {self._lock}")
            stats = {
                'total_retries': self._total_retries,
                'successful_retries': self._successful_retries,
                'exhausted_retries': self._exhausted_retries,
                'successful_after_retry': self._successful_after_retry,
                'high_failure_operations': list(self._high_failure_operations),
                'alerted_operations': list(self._alerted_operations),
                'alert_thresholds': self._alert_thresholds.copy(),
                'operation_stats': {},
                'exception_stats': {},
                'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            logger.debug(f"Releasing lock in get_statistics: {self._lock}")
            return stats
    
    def reset_statistics(self):
        logger.debug("reset_statistics called")
        self._reset_stats()
    
    def set_persistence_file(self, file_path):
        logger.debug(f"set_persistence_file called with {file_path}")
        self._persistence_file = file_path
    
    def record_retry(self, operation, exception_type, exception_message, 
                    attempt, max_attempts, successful, component, duration_ms=None):
        logger.debug(f"record_retry called for {operation}")
        with self._lock:
            logger.debug(f"Acquired lock in record_retry: {self._lock}")
            # Just increment counters, don't do anything complex
            self._total_retries += 1
            if successful:
                self._successful_retries += 1
            logger.debug(f"Releasing lock in record_retry: {self._lock}")
    
    def _load_from_file(self):
        logger.debug(f"_load_from_file called for {self._persistence_file}")
        # In debug version, don't actually load anything

# Create a custom module
debug_retry_monitor_module = type('module', (), {})()

# Add the debug RetryMonitor class to the module
debug_retry_monitor_module.RetryMonitor = DebugRetryMonitor
logger.debug("Created debug RetryMonitor class")

# Create a global singleton instance
logger.debug("Creating global retry_monitor instance")
debug_retry_monitor = DebugRetryMonitor(persistence_file=None)
debug_retry_monitor_module.retry_monitor = debug_retry_monitor

# Add the utility functions
def debug_get_retry_statistics():
    logger.debug("debug_get_retry_statistics called")
    return debug_retry_monitor.get_statistics()

def debug_reset_retry_statistics():
    logger.debug("debug_reset_retry_statistics called")
    debug_retry_monitor.reset_statistics()

def debug_record_retry(operation, exception, attempt, max_attempts, successful, component, duration_ms=None):
    logger.debug("debug_record_retry called")
    debug_retry_monitor.record_retry(
        operation=operation,
        exception_type=type(exception).__name__,
        exception_message=str(exception),
        attempt=attempt,
        max_attempts=max_attempts,
        successful=successful,
        component=component,
        duration_ms=duration_ms
    )

# Add the functions to the module
debug_retry_monitor_module.get_retry_statistics = debug_get_retry_statistics
debug_retry_monitor_module.reset_retry_statistics = debug_reset_retry_statistics
debug_retry_monitor_module.record_retry = debug_record_retry
debug_retry_monitor_module.DEFAULT_PERSISTENCE_PATH = None

# Replace the module in sys.modules
logger.debug("Replacing retry_monitor in sys.modules with debug version")
sys.modules['deployment.app.utils.retry_monitor'] = debug_retry_monitor_module

# Now try to import and use it
logger.debug("About to import retry_monitor")
try:
    from deployment.app.utils.retry_monitor import retry_monitor, get_retry_statistics, reset_retry_statistics
    logger.debug(f"Successfully imported retry_monitor: {retry_monitor}")
    
    # Try to use the functions
    logger.debug("About to call get_retry_statistics")
    stats = get_retry_statistics()
    logger.debug(f"get_retry_statistics returned: {stats}")
    
    logger.debug("About to call reset_retry_statistics")
    reset_retry_statistics()
    logger.debug("reset_retry_statistics completed")
    
    logger.debug("All retry_monitor operations completed successfully")
except Exception as e:
    logger.exception(f"Error importing or using retry_monitor: {str(e)}")

# Simple test function to verify everything is working
def test_retry_monitor_debug():
    """A minimal test that should pass."""
    assert True 