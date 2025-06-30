"""
Utility for monitoring and tracking retry statistics.
Provides functionality to collect, analyze, and log retry patterns.
"""

import json
import logging
import os
import statistics
import time
from collections.abc import Callable
from datetime import datetime
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


class RetryMonitor:
    """
    Tracks and analyzes retry statistics for different operations.

    This class is thread-safe and provides insights into retry patterns
    across different components of the application.
    """

    def __init__(
        self,
        capacity: int = 1000,
        log_interval_seconds: int = 300,
        persistence_file: str | None = None,
        save_interval_seconds: int = 600,
    ):
        """
        Initialize retry monitor.

        Args:
            capacity: Maximum number of retry events to store
            log_interval_seconds: Interval for periodic logging of statistics
            persistence_file: Path to file for persisting statistics (None to disable)
            save_interval_seconds: Interval for saving statistics to file
        """
        self._retry_events = []
        self._capacity = capacity
        self._log_interval = log_interval_seconds
        self._lock = RLock()
        self._last_log_time = time.time()

        # Persistence settings
        self._persistence_file = persistence_file
        self._save_interval = save_interval_seconds
        self._last_save_time = time.time()

        # Track operations with high failure rates
        self._high_failure_operations = set()

        # Alert thresholds and handlers
        self._alert_thresholds = {
            "total_failures_count": 10,  # Alert after X total failures
            "consecutive_failures": 3,  # Alert after X consecutive failures
            "failure_rate_percent": 50,  # Alert when failure rate exceeds X%
            "response_time_ms": 3000,  # Alert when response time exceeds X ms
            "exhausted_retries_count": 5,  # Alert after X exhausted retries
        }
        self._alert_handlers = []
        self._alerted_operations = set()  # Track operations that have triggered alerts
        self._operation_consecutive_failures = {}  # Track consecutive failures by operation

        # Initialize statistics
        self._reset_stats()

        # Load previous statistics if persistence file exists
        if self._persistence_file and os.path.exists(self._persistence_file):
            self._load_from_file()

    def record_retry(
        self,
        operation: str,
        exception_type: str,
        exception_message: str,
        attempt: int,
        max_attempts: int,
        successful: bool,
        component: str,
        duration_ms: int | None = None,
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
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "exception_type": exception_type,
            "exception_message": exception_message,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "successful": successful,
            "component": component,
            "duration_ms": duration_ms,
        }

        with self._lock:
            # Add event to the list, maintaining capacity
            self._retry_events.append(event)
            if len(self._retry_events) > self._capacity:
                self._retry_events.pop(0)

            # Update statistics
            self._total_retries += 1

            # Create a key for this operation
            operation_key = f"{component}.{operation}"

            # Update consecutive failures tracking
            if not successful:
                self._operation_consecutive_failures[operation_key] = (
                    self._operation_consecutive_failures.get(operation_key, 0) + 1
                )

                # Check consecutive failures alert threshold
                if (
                    self._operation_consecutive_failures[operation_key]
                    >= self._alert_thresholds["consecutive_failures"]
                ):
                    self._check_alert(
                        operation_key=operation_key,
                        alert_type="consecutive_failures",
                        alert_value=self._operation_consecutive_failures[operation_key],
                        event=event,
                    )
            else:
                # Reset consecutive failures counter on success
                self._operation_consecutive_failures[operation_key] = 0

            if attempt == max_attempts and not successful:
                self._exhausted_retries += 1

                # Track operations with exhausted retries
                self._operation_failures[operation_key] = (
                    self._operation_failures.get(operation_key, 0) + 1
                )

                # Check if this operation has high failure rates
                if self._operation_failures[operation_key] >= 5:
                    self._high_failure_operations.add(operation_key)

                # Check total failures alert threshold
                if (
                    self._operation_failures[operation_key]
                    >= self._alert_thresholds["total_failures_count"]
                ):
                    self._check_alert(
                        operation_key=operation_key,
                        alert_type="total_failures_count",
                        alert_value=self._operation_failures[operation_key],
                        event=event,
                    )

                # Check exhausted retries alert threshold
                if (
                    self._exhausted_retries
                    >= self._alert_thresholds["exhausted_retries_count"]
                ):
                    self._check_alert(
                        operation_key=operation_key,
                        alert_type="exhausted_retries_count",
                        alert_value=self._exhausted_retries,
                        event=event,
                    )

            if successful:
                self._successful_retries += 1
                if attempt > 1:
                    self._successful_after_retry += 1

            # Check response time alert threshold if duration provided
            if duration_ms and duration_ms > self._alert_thresholds["response_time_ms"]:
                self._check_alert(
                    operation_key=operation_key,
                    alert_type="response_time_ms",
                    alert_value=duration_ms,
                    event=event,
                )

            # Calculate and check failure rate
            total_op_attempts = sum(
                1
                for e in self._retry_events
                if f"{e['component']}.{e['operation']}" == operation_key
            )
            if total_op_attempts >= 5:  # Only check if we have enough data
                failed_op_attempts = sum(
                    1
                    for e in self._retry_events
                    if f"{e['component']}.{e['operation']}" == operation_key
                    and not e["successful"]
                )
                failure_rate = (failed_op_attempts / total_op_attempts) * 100
                if failure_rate >= self._alert_thresholds["failure_rate_percent"]:
                    self._check_alert(
                        operation_key=operation_key,
                        alert_type="failure_rate_percent",
                        alert_value=failure_rate,
                        event=event,
                    )

            # Check if it's time to log statistics
            current_time = time.time()
            if current_time - self._last_log_time > self._log_interval:
                self._log_statistics()
                self._last_log_time = current_time

            # Check if it's time to save statistics
            if (
                self._persistence_file
                and current_time - self._last_save_time > self._save_interval
            ):
                self._save_to_file()
                self._last_save_time = current_time

    def set_alert_threshold(self, threshold_name: str, value: int | float) -> None:
        """
        Set an alert threshold value.

        Args:
            threshold_name: Name of the threshold to set
            value: New threshold value
        """
        with self._lock:
            if threshold_name in self._alert_thresholds:
                self._alert_thresholds[threshold_name] = value
            else:
                logger.warning(f"Unknown alert threshold: {threshold_name}")

    def register_alert_handler(
        self, handler: Callable[[str, str, Any, dict[str, Any]], None]
    ) -> None:
        """
        Register a function to be called when alert thresholds are exceeded.

        Args:
            handler: Function to call with parameters (operation_key, alert_type, alert_value, event)
        """
        with self._lock:
            self._alert_handlers.append(handler)

    def get_statistics(self) -> dict[str, Any]:
        """
        Get current retry statistics.

        Returns:
            Dictionary with retry statistics
        """
        with self._lock:
            stats = {
                "total_retries": self._total_retries,
                "successful_retries": self._successful_retries,
                "exhausted_retries": self._exhausted_retries,
                "successful_after_retry": self._successful_after_retry,
                "high_failure_operations": list(self._high_failure_operations),
                "alerted_operations": list(self._alerted_operations),
                "alert_thresholds": self._alert_thresholds.copy(),
                "operation_stats": self._get_operation_stats(),
                "exception_stats": self._get_exception_stats(),
                "timestamp": datetime.now().isoformat(),
            }

            return stats

    def get_high_failure_operations(self) -> set[str]:
        """
        Get operations with high failure rates.

        Returns:
            Set of operation names
        """
        with self._lock:
            return self._high_failure_operations.copy()

    def get_alerted_operations(self) -> set[str]:
        """
        Get operations that have triggered alerts.

        Returns:
            Set of operation names
        """
        with self._lock:
            return self._alerted_operations.copy()

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._reset_stats()

            # Save empty statistics if persistence is enabled
            if self._persistence_file:
                self._save_to_file()

    def set_persistence_file(self, file_path: str | None) -> None:
        """
        Set the persistence file path.

        Args:
            file_path: Path to file for persisting statistics, or None to disable
        """
        with self._lock:
            self._persistence_file = file_path
            if file_path and os.path.exists(file_path):
                self._load_from_file()
            self._last_save_time = time.time()  # Reset save timer

    def _reset_stats(self) -> None:
        """Reset all statistical counters."""
        self._total_retries = 0
        self._successful_retries = 0
        self._exhausted_retries = 0
        self._successful_after_retry = 0
        self._operation_failures = {}
        self._high_failure_operations = set()
        self._alerted_operations = set()
        self._operation_consecutive_failures = {}

    def _check_alert(
        self,
        operation_key: str,
        alert_type: str,
        alert_value: Any,
        event: dict[str, Any],
    ) -> None:
        """
        Check if an alert should be triggered and notify handlers.

        Args:
            operation_key: Operation identifier
            alert_type: Type of alert that was triggered
            alert_value: Value that triggered the alert
            event: The retry event that triggered the alert
        """
        # Mark this operation as alerted
        self._alerted_operations.add(operation_key)

        # Log the alert
        threshold_value = self._alert_thresholds[alert_type]
        logger.warning(
            f"Alert threshold exceeded for {operation_key}: "
            f"{alert_type}={alert_value} (threshold={threshold_value})"
        )

        # Call registered alert handlers
        for handler in self._alert_handlers:
            try:
                handler(operation_key, alert_type, alert_value, event)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")

    def _log_statistics(self) -> None:
        """Log current statistics."""
        stats = self.get_statistics()

        logger.info(
            f"Retry statistics: "
            f"total={stats['total_retries']}, "
            f"successful={stats['successful_retries']}, "
            f"exhausted={stats['exhausted_retries']}"
        )

        # Log high failure operations
        if stats["high_failure_operations"]:
            logger.warning(
                f"Operations with high failure rates: "
                f"{', '.join(stats['high_failure_operations'])}"
            )

        # Log alerted operations
        if stats["alerted_operations"]:
            logger.warning(
                f"Operations that triggered alerts: "
                f"{', '.join(stats['alerted_operations'])}"
            )

    def _save_to_file(self) -> None:
        """Save statistics to the persistence file."""
        if not self._persistence_file:
            return

        try:
            # Get current statistics
            stats = self.get_statistics()

            # Create directory if it doesn't exist
            os.makedirs(
                os.path.dirname(os.path.abspath(self._persistence_file)), exist_ok=True
            )

            # Save to file
            with open(self._persistence_file, "w") as f:
                json.dump(stats, f, indent=2)

        except Exception as e:
            logger.error(
                "[RetryMonitor._save_to_file] Failed to save retry statistics: %s",
                str(e),
            )

    def _load_from_file(self) -> None:
        """Load statistics from the persistence file."""
        if not self._persistence_file or not os.path.exists(self._persistence_file):
            return

        try:
            with open(self._persistence_file) as f:
                stats = json.load(f)

            # Restore statistics
            self._total_retries = stats.get("total_retries", 0)
            self._successful_retries = stats.get("successful_retries", 0)
            self._exhausted_retries = stats.get("exhausted_retries", 0)
            self._successful_after_retry = stats.get("successful_after_retry", 0)
            self._high_failure_operations = set(
                stats.get("high_failure_operations", [])
            )

            # Restore operation failures
            operation_stats = stats.get("operation_stats", {})
            for op, op_stats in operation_stats.items():
                count = op_stats.get("count", 0)
                success_rate = op_stats.get("success_rate", 0)

                # Estimate failures based on count and success rate
                if count > 0 and success_rate < 1.0:
                    failures = int(count * (1 - success_rate))
                    if failures > 0:
                        self._operation_failures[op] = failures

            logger.info(f"Loaded retry statistics from {self._persistence_file}")
        except Exception as e:
            logger.error("Failed to load retry statistics: %s", str(e))
            # Initialize with empty statistics
            self._reset_stats()

    def _get_operation_stats(self) -> dict[str, dict[str, Any]]:
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
            attempts = [event["attempt"] for event in events]
            successful = [event for event in events if event["successful"]]
            duration_values = [
                event["duration_ms"]
                for event in events
                if event["duration_ms"] is not None
            ]

            operation_stats[operation] = {
                "count": len(events),
                "success_rate": len(successful) / len(events) if events else 0,
                "avg_attempts": statistics.mean(attempts) if attempts else 0,
                "avg_duration_ms": statistics.mean(duration_values)
                if duration_values
                else None,
                "last_exception": events[-1]["exception_type"] if events else None,
                "last_exception_message": events[-1]["exception_message"]
                if events
                else None,
            }

        return operation_stats

    def _get_exception_stats(self) -> dict[str, dict[str, Any]]:
        """
        Calculate statistics for each exception type.

        Returns:
            Dictionary with exception statistics
        """
        # Group retry events by exception type
        exception_events = {}
        for event in self._retry_events:
            exception_type = event["exception_type"]
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
                "count": len(events),
                "success_rate": len([e for e in events if e["successful"]])
                / len(events)
                if events
                else 0,
                "by_operation": by_operation,
            }

        return exception_stats


# Global singleton instance of RetryMonitor
# Default persistence file in logs directory
DEFAULT_PERSISTENCE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
    "retry_statistics.json",
)

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(DEFAULT_PERSISTENCE_PATH), exist_ok=True)

retry_monitor = RetryMonitor(persistence_file=DEFAULT_PERSISTENCE_PATH)


def record_retry(
    operation: str,
    exception: Exception,
    attempt: int,
    max_attempts: int,
    successful: bool,
    component: str,
    duration_ms: int | None = None,
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
        duration_ms=duration_ms,
    )


def get_retry_statistics() -> dict[str, Any]:
    """
    Get current retry statistics from the global monitor.

    Returns:
        Dictionary with retry statistics
    """
    return retry_monitor.get_statistics()


def get_high_failure_operations() -> set[str]:
    """
    Get operations with high failure rates from the global monitor.

    Returns:
        Set of operation names
    """
    return retry_monitor.get_high_failure_operations()


def reset_retry_statistics() -> None:
    """Reset all statistics in the global monitor."""
    retry_monitor.reset_statistics()


# Default alert handler that logs alerts
def log_alert_handler(
    operation_key: str, alert_type: str, alert_value: Any, event: dict[str, Any]
) -> None:
    """
    Default alert handler that logs detailed alert information.

    Args:
        operation_key: Operation identifier
        alert_type: Type of alert that was triggered
        alert_value: Value that triggered the alert
        event: The retry event that triggered the alert
    """
    logger.error(
        f"ALERT: {alert_type} threshold exceeded for {operation_key}\n"
        f"Current value: {alert_value}\n"
        f"Exception: {event['exception_type']}: {event['exception_message']}\n"
        f"Component: {event['component']}, Operation: {event['operation']}\n"
        f"Attempt: {event['attempt']}/{event['max_attempts']}, Successful: {event['successful']}"
    )


# Register the default alert handler with the global retry monitor
retry_monitor.register_alert_handler(log_alert_handler)


def set_alert_threshold(threshold_name: str, value: int | float) -> None:
    """
    Set an alert threshold value in the global monitor.

    Args:
        threshold_name: Name of the threshold to set
        value: New threshold value
    """
    retry_monitor.set_alert_threshold(threshold_name, value)


def register_alert_handler(
    handler: Callable[[str, str, Any, dict[str, Any]], None],
) -> None:
    """
    Register an alert handler with the global monitor.

    Args:
        handler: Function to call when thresholds are exceeded
    """
    retry_monitor.register_alert_handler(handler)


def get_alerted_operations() -> set[str]:
    """
    Get operations that have triggered alerts from the global monitor.

    Returns:
        Set of operation names
    """
    return retry_monitor.get_alerted_operations()
