"""
Utility for monitoring and tracking retry statistics.
Provides functionality to collect, analyze, and log retry patterns.
"""
import json
import logging
import time
from datetime import datetime
from threading import Lock
from typing import Dict, Any, List, Optional, Set, Union, Tuple
import statistics

logger = logging.getLogger(__name__)

class RetryMonitor:
    """
    Tracks and analyzes retry statistics for different operations.
    
    This class is thread-safe and provides insights into retry patterns
    across different components of the application.
    """
    
    def __init__(self, capacity: int = 1000, log_interval_seconds: int = 300):
        """
        Initialize retry monitor.
        
        Args:
            capacity: Maximum number of retry events to store
            log_interval_seconds: Interval for periodic logging of statistics
        """
        self._retry_events = []
        self._capacity = capacity
        self._log_interval = log_interval_seconds
        self._lock = Lock()
        self._last_log_time = time.time()
        
        # Track operations with high failure rates
        self._high_failure_operations = set()
        
        # Initialize statistics
        self._reset_stats()
    
    def record_retry(
        self, 
        operation: str, 
        exception_type: str,
        exception_message: str, 
        attempt: int, 
        max_attempts: int,
        successful: bool,
        component: str,
        duration_ms: Optional[int] = None
    ) -> None:
        """
        Record a retry event.
        
        Args:
            operation: Name of the operation being retried
            exception_type: Type of exception that triggered the retry
            exception_message: Exception message
            attempt: Current attempt number
            max_attempts: Maximum number of allowed attempts
            successful: Whether the attempt was ultimately successful
            component: Component name (e.g., 'storage_client', 'function_client')
            duration_ms: Duration of the operation in milliseconds
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'exception_type': exception_type,
            'exception_message': exception_message,
            'attempt': attempt,
            'max_attempts': max_attempts,
            'successful': successful,
            'component': component,
            'duration_ms': duration_ms
        }
        
        with self._lock:
            # Add event to the list, maintaining capacity
            self._retry_events.append(event)
            if len(self._retry_events) > self._capacity:
                self._retry_events.pop(0)
            
            # Update statistics
            self._total_retries += 1
            
            if attempt == max_attempts and not successful:
                self._exhausted_retries += 1
                
                # Track operations with exhausted retries
                operation_key = f"{component}.{operation}"
                self._operation_failures[operation_key] = self._operation_failures.get(operation_key, 0) + 1
                
                # Check if this operation has high failure rates
                if self._operation_failures[operation_key] >= 5:
                    self._high_failure_operations.add(operation_key)
            
            if successful:
                self._successful_retries += 1
                if attempt > 1:
                    self._successful_after_retry += 1
            
            # Check if it's time to log statistics
            current_time = time.time()
            if current_time - self._last_log_time > self._log_interval:
                self._log_statistics()
                self._last_log_time = current_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current retry statistics.
        
        Returns:
            Dictionary with retry statistics
        """
        with self._lock:
            stats = {
                'total_retries': self._total_retries,
                'successful_retries': self._successful_retries,
                'exhausted_retries': self._exhausted_retries,
                'successful_after_retry': self._successful_after_retry,
                'high_failure_operations': list(self._high_failure_operations),
                'operation_stats': self._get_operation_stats(),
                'exception_stats': self._get_exception_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
            return stats
    
    def get_high_failure_operations(self) -> Set[str]:
        """
        Get operations with high failure rates.
        
        Returns:
            Set of operation names
        """
        with self._lock:
            return self._high_failure_operations.copy()
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._reset_stats()
    
    def _reset_stats(self) -> None:
        """Reset all statistical counters."""
        self._total_retries = 0
        self._successful_retries = 0
        self._exhausted_retries = 0
        self._successful_after_retry = 0
        self._operation_failures = {}
        self._high_failure_operations = set()
    
    def _log_statistics(self) -> None:
        """Log current statistics."""
        stats = self.get_statistics()
        
        logger.info(f"Retry statistics: "
                   f"total={stats['total_retries']}, "
                   f"successful={stats['successful_retries']}, "
                   f"exhausted={stats['exhausted_retries']}")
        
        # Log high failure operations
        if stats['high_failure_operations']:
            logger.warning(f"Operations with high failure rates: "
                         f"{', '.join(stats['high_failure_operations'])}")
    
    def _get_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for each operation.
        
        Returns:
            Dictionary with operation statistics
        """
        # Group retry events by operation
        operation_events = {}
        for event in self._retry_events:
            operation = f"{event['component']}.{event['operation']}"
            if operation not in operation_events:
                operation_events[operation] = []
            operation_events[operation].append(event)
        
        # Calculate statistics for each operation
        operation_stats = {}
        for operation, events in operation_events.items():
            attempts = [event['attempt'] for event in events]
            successful = [event for event in events if event['successful']]
            duration_values = [event['duration_ms'] for event in events 
                             if event['duration_ms'] is not None]
            
            operation_stats[operation] = {
                'count': len(events),
                'success_rate': len(successful) / len(events) if events else 0,
                'avg_attempts': statistics.mean(attempts) if attempts else 0,
                'avg_duration_ms': statistics.mean(duration_values) if duration_values else None,
                'last_exception': events[-1]['exception_type'] if events else None,
                'last_exception_message': events[-1]['exception_message'] if events else None
            }
        
        return operation_stats
    
    def _get_exception_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for each exception type.
        
        Returns:
            Dictionary with exception statistics
        """
        # Group retry events by exception type
        exception_events = {}
        for event in self._retry_events:
            exception_type = event['exception_type']
            if exception_type not in exception_events:
                exception_events[exception_type] = []
            exception_events[exception_type].append(event)
        
        # Calculate statistics for each exception type
        exception_stats = {}
        for exception_type, events in exception_events.items():
            by_operation = {}
            for event in events:
                operation = f"{event['component']}.{event['operation']}"
                if operation not in by_operation:
                    by_operation[operation] = 0
                by_operation[operation] += 1
            
            exception_stats[exception_type] = {
                'count': len(events),
                'success_rate': len([e for e in events if e['successful']]) / len(events) if events else 0,
                'by_operation': by_operation
            }
        
        return exception_stats


# Global singleton instance of RetryMonitor
retry_monitor = RetryMonitor()

def record_retry(
    operation: str, 
    exception: Exception, 
    attempt: int, 
    max_attempts: int,
    successful: bool,
    component: str,
    duration_ms: Optional[int] = None
) -> None:
    """
    Record a retry event using the global monitor.
    
    Args:
        operation: Name of the operation being retried
        exception: Exception that triggered the retry
        attempt: Current attempt number
        max_attempts: Maximum number of allowed attempts
        successful: Whether the attempt was ultimately successful
        component: Component name
        duration_ms: Duration of the operation in milliseconds
    """
    retry_monitor.record_retry(
        operation=operation,
        exception_type=type(exception).__name__,
        exception_message=str(exception),
        attempt=attempt,
        max_attempts=max_attempts,
        successful=successful,
        component=component,
        duration_ms=duration_ms
    )


def get_retry_statistics() -> Dict[str, Any]:
    """
    Get current retry statistics from the global monitor.
    
    Returns:
        Dictionary with retry statistics
    """
    return retry_monitor.get_statistics()


def get_high_failure_operations() -> Set[str]:
    """
    Get operations with high failure rates from the global monitor.
    
    Returns:
        Set of operation names
    """
    return retry_monitor.get_high_failure_operations()


def reset_retry_statistics() -> None:
    """Reset all statistics in the global monitor."""
    retry_monitor.reset_statistics() 