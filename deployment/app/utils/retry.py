"""
Utility functions for implementing retry logic.
Provides decorators and functions for retrying operations with proper backoff and jitter.
"""
import asyncio
import functools
import logging
import random
import time
from typing import Callable, TypeVar, Optional, Type, Dict, Any, Union, List, Tuple, Coroutine
from types import TracebackType

import backoff
import requests
from requests.exceptions import RequestException
from botocore.exceptions import ClientError

from app.utils.error_handling import RetryableError
from app.utils.retry_monitor import record_retry

logger = logging.getLogger(__name__)

# Type variables for function signatures
T = TypeVar('T')
R = TypeVar('R')


def is_retryable_http_error(exception: Exception) -> bool:
    """
    Determine if an HTTP error should be retried.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    if isinstance(exception, requests.HTTPError):
        # Only retry on server errors (5xx) or specific retryable client errors
        status_code = exception.response.status_code
        return (
            status_code >= 500 or  # Server errors
            status_code == 429 or  # Too Many Requests
            status_code == 408      # Request Timeout
        )
    elif isinstance(exception, requests.RequestException):
        # Retry network errors like connection errors, timeouts
        return not isinstance(exception, (
            requests.exceptions.InvalidURL,
            requests.exceptions.InvalidSchema,
            requests.exceptions.MissingSchema,
            requests.exceptions.InvalidHeader,
            requests.exceptions.InvalidProxyURL
        ))
    
    return False


def is_retryable_cloud_error(exception: Exception) -> bool:
    """
    Determine if a cloud service error should be retried.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    if isinstance(exception, ClientError):
        error_code = exception.response.get('Error', {}).get('Code', '')
        status_code = exception.response.get('ResponseMetadata', {}).get('HTTPStatusCode', 0)
        
        # Common retryable cloud service errors
        retryable_codes = {
            # AWS/Generic cloud errors
            'ThrottlingException',
            'Throttling',
            'TooManyRequestsException',
            'RequestLimitExceeded',
            'RequestThrottled',
            'ProvisionedThroughputExceededException',
            'RequestTimeout',
            'SlowDown',
            'InternalServerError',
            'ServiceUnavailable',
            'ServerUnavailable',
            'ServiceFailure',
            'InternalFailure',
            
            # Yandex.Cloud specific errors
            'RESOURCE_EXHAUSTED',
            'UNAVAILABLE',
            'DEADLINE_EXCEEDED',
            'INTERNAL',
            'UNAUTHENTICATED',
            'SERVICE_UNAVAILABLE',
            'TIMEOUT',
            'CONNECTION_FAILURE',
            'TOO_MANY_REQUESTS',
            'INSTANCE_UNAVAILABLE',
            'THROTTLING'
        }
        
        return (
            error_code in retryable_codes or
            status_code >= 500 or
            status_code == 429
        )
    
    # Check for Yandex.Cloud SDK exceptions (assuming specific exception types)
    # These would be different from AWS ClientError
    if hasattr(exception, 'code') and hasattr(exception, 'message'):
        # Common pattern in many cloud SDKs
        error_code = getattr(exception, 'code', '')
        
        yandex_retryable_codes = {
            'RESOURCE_EXHAUSTED',
            'UNAVAILABLE',
            'DEADLINE_EXCEEDED',
            'INTERNAL',
            'UNAUTHENTICATED',
            'SERVICE_UNAVAILABLE',
            'TIMEOUT',
            'CONNECTION_FAILURE',
            'TOO_MANY_REQUESTS'
        }
        
        return error_code in yandex_retryable_codes
    
    # Add any other cloud service exception types here
    
    return False


def calculate_backoff(retry_attempt: int, base_delay: float = 1.0, 
                     max_delay: float = 60.0, jitter: bool = True) -> float:
    """
    Calculate exponential backoff with jitter.
    
    Args:
        retry_attempt: Current retry attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add jitter
        
    Returns:
        Delay in seconds before next retry
    """
    # Calculate exponential backoff
    delay = min(max_delay, base_delay * (2 ** retry_attempt))
    
    # Add jitter if enabled (up to 25% in either direction)
    if jitter:
        delay = delay * (0.75 + random.random() * 0.5)
        
    return delay


def retry_with_backoff(
    max_tries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    giveup_func: Optional[Callable[[Exception], bool]] = None,
    component: str = "generic"
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_tries: Maximum number of attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        retryable_exceptions: Tuple of exceptions that trigger retry
        giveup_func: Function that determines when to give up retrying
        component: Component name for monitoring purposes
        
    Returns:
        Decorated function with retry logic
    """
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            operation = func.__name__
            attempt = 0
            last_exception = None
            start_time = time.time()
            
            while True:
                try:
                    attempt += 1
                    result = func(*args, **kwargs)
                    
                    # Record successful retry if this was a retry attempt
                    if attempt > 1 and last_exception is not None:
                        duration_ms = int((time.time() - start_time) * 1000)
                        record_retry(
                            operation=operation,
                            exception=last_exception,
                            attempt=attempt,
                            max_attempts=max_tries,
                            successful=True,
                            component=component,
                            duration_ms=duration_ms
                        )
                    
                    return result
                except retryable_exceptions as e:
                    last_exception = e
                    duration_ms = int((time.time() - start_time) * 1000)
                    
                    # Check if we should give up
                    if (attempt >= max_tries or 
                        (giveup_func is not None and giveup_func(e))):
                        logger.warning(
                            f"Giving up on {func.__name__} after {attempt} attempts. "
                            f"Last error: {str(e)}"
                        )
                        
                        # Record failed retry
                        record_retry(
                            operation=operation,
                            exception=e,
                            attempt=attempt,
                            max_attempts=max_tries,
                            successful=False,
                            component=component,
                            duration_ms=duration_ms
                        )
                        
                        raise
                    
                    # Calculate backoff time
                    delay = calculate_backoff(
                        retry_attempt=attempt - 1,
                        base_delay=base_delay,
                        max_delay=max_delay
                    )
                    
                    logger.info(
                        f"Retrying {func.__name__} after error: {str(e)}. "
                        f"Attempt {attempt}/{max_tries}, waiting {delay:.2f}s"
                    )
                    
                    # Record retry attempt
                    record_retry(
                        operation=operation,
                        exception=e,
                        attempt=attempt,
                        max_attempts=max_tries,
                        successful=False,  # Not yet successful
                        component=component,
                        duration_ms=duration_ms
                    )
                    
                    time.sleep(delay)
        
        return wrapper
    
    return decorator


async def retry_async_with_backoff(
    max_tries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    giveup_func: Optional[Callable[[Exception], bool]] = None,
    component: str = "async"
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_tries: Maximum number of attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        retryable_exceptions: Tuple of exceptions that trigger retry
        giveup_func: Function that determines when to give up retrying
        component: Component name for monitoring purposes
        
    Returns:
        Decorated async function with retry logic
    """
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)
    
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            operation = func.__name__
            attempt = 0
            last_exception = None
            start_time = time.time()
            
            while True:
                try:
                    attempt += 1
                    result = await func(*args, **kwargs)
                    
                    # Record successful retry if this was a retry attempt
                    if attempt > 1 and last_exception is not None:
                        duration_ms = int((time.time() - start_time) * 1000)
                        record_retry(
                            operation=operation,
                            exception=last_exception,
                            attempt=attempt,
                            max_attempts=max_tries,
                            successful=True,
                            component=component,
                            duration_ms=duration_ms
                        )
                    
                    return result
                except retryable_exceptions as e:
                    last_exception = e
                    duration_ms = int((time.time() - start_time) * 1000)
                    
                    # Check if we should give up
                    if (attempt >= max_tries or 
                        (giveup_func is not None and giveup_func(e))):
                        logger.warning(
                            f"Giving up on {func.__name__} after {attempt} attempts. "
                            f"Last error: {str(e)}"
                        )
                        
                        # Record failed retry
                        record_retry(
                            operation=operation,
                            exception=e,
                            attempt=attempt,
                            max_attempts=max_tries,
                            successful=False,
                            component=component,
                            duration_ms=duration_ms
                        )
                        
                        raise
                    
                    # Calculate backoff time
                    delay = calculate_backoff(
                        retry_attempt=attempt - 1,
                        base_delay=base_delay,
                        max_delay=max_delay
                    )
                    
                    logger.info(
                        f"Retrying {func.__name__} after error: {str(e)}. "
                        f"Attempt {attempt}/{max_tries}, waiting {delay:.2f}s"
                    )
                    
                    # Record retry attempt
                    record_retry(
                        operation=operation,
                        exception=e,
                        attempt=attempt,
                        max_attempts=max_tries,
                        successful=False,  # Not yet successful
                        component=component,
                        duration_ms=duration_ms
                    )
                    
                    await asyncio.sleep(delay)
        
        return wrapper
    
    return decorator


# Pre-configured retry decorators for common use cases

def retry_http_request(max_tries: int = 3, base_delay: float = 2.0, max_delay: float = 30.0, component: str = "http"):
    """
    Decorator specifically for retrying HTTP requests.
    
    Args:
        max_tries: Maximum number of attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        component: Component name for monitoring purposes
        
    Returns:
        Decorated function with HTTP request retry logic
    """
    return retry_with_backoff(
        max_tries=max_tries,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=(RequestException,),
        giveup_func=lambda e: not is_retryable_http_error(e),
        component=component
    )


def retry_cloud_operation(max_tries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0, component: str = "cloud"):
    """
    Decorator specifically for retrying cloud operations.
    
    Args:
        max_tries: Maximum number of attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        component: Component name for monitoring purposes
        
    Returns:
        Decorated function with cloud operation retry logic
    """
    return retry_with_backoff(
        max_tries=max_tries,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=(ClientError, RequestException),
        giveup_func=lambda e: not (is_retryable_cloud_error(e) or is_retryable_http_error(e)),
        component=component
    )


def retry_async_http_request(max_tries: int = 3, base_delay: float = 2.0, max_delay: float = 30.0, component: str = "async_http"):
    """
    Decorator specifically for retrying async HTTP requests.
    
    Args:
        max_tries: Maximum number of attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        component: Component name for monitoring purposes
        
    Returns:
        Decorated async function with HTTP request retry logic
    """
    return retry_async_with_backoff(
        max_tries=max_tries,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=(RequestException,),
        giveup_func=lambda e: not is_retryable_http_error(e),
        component=component
    )


def retry_async_cloud_operation(max_tries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0, component: str = "async_cloud"):
    """
    Decorator specifically for retrying async cloud operations.
    
    Args:
        max_tries: Maximum number of attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        component: Component name for monitoring purposes
        
    Returns:
        Decorated async function with cloud operation retry logic
    """
    return retry_async_with_backoff(
        max_tries=max_tries,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=(ClientError, RequestException),
        giveup_func=lambda e: not (is_retryable_cloud_error(e) or is_retryable_http_error(e)),
        component=component
    )


class RetryContext:
    """
    Context manager for implementing retry logic in a block of code.
    
    Example:
        ```python
        with RetryContext(max_tries=3, exceptions=(ConnectionError,)) as retry:
            while retry.attempts():
                try:
                    # Operation that might fail
                    result = api.call()
                    retry.success()
                    return result
                except retry.exceptions as e:
                    retry.failed(e)
        ```
    """
    
    def __init__(
        self,
        max_tries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        giveup_func: Optional[Callable[[Exception], bool]] = None,
        on_give_up: Optional[Callable[[Exception, int], None]] = None,
        component: str = "context",
        operation: Optional[str] = None
    ):
        """
        Initialize retry context.
        
        Args:
            max_tries: Maximum number of attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds 
            exceptions: Tuple of exceptions that trigger retry
            giveup_func: Function that determines when to give up retrying
            on_give_up: Callback when giving up retries
            component: Component name for monitoring purposes
            operation: Operation name (defaults to caller's function name)
        """
        self.max_tries = max_tries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions or (Exception,)
        self.giveup_func = giveup_func
        self.on_give_up = on_give_up
        self.component = component
        self.operation = operation or self._get_caller_name()
        
        self._attempt = 0
        self._last_error = None
        self._exhausted = False
        self._succeeded = False
        self._start_time = time.time()
    
    def __enter__(self) -> 'RetryContext':
        return self
    
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> bool:
        if exc_type and issubclass(exc_type, self.exceptions):
            self._last_error = exc_val
            if self._attempt < self.max_tries:
                self._delay()
                return True  # Suppress exception and retry
        return False
    
    def attempts(self) -> bool:
        """
        Check if attempts are available and increment attempt counter.
        
        Returns:
            True if attempts remain, False otherwise
        """
        if self._exhausted or self._succeeded:
            return False
            
        self._attempt += 1
        return True
    
    def failed(self, exception: Exception) -> None:
        """
        Handle failure in retry loop.
        
        Args:
            exception: The exception that caused the failure
        
        Raises:
            Exception: If max retries reached or should give up
        """
        self._last_error = exception
        duration_ms = int((time.time() - self._start_time) * 1000)
        
        # Check if we should give up
        if (self._attempt >= self.max_tries or 
            (self.giveup_func and self.giveup_func(exception))):
            
            self._exhausted = True
            
            # Record failed retry in monitoring
            record_retry(
                operation=self.operation,
                exception=exception,
                attempt=self._attempt,
                max_attempts=self.max_tries,
                successful=False,
                component=self.component,
                duration_ms=duration_ms
            )
            
            if self.on_give_up:
                self.on_give_up(exception, self._attempt)
                
            raise exception
        
        # Record retry attempt in monitoring
        record_retry(
            operation=self.operation,
            exception=exception,
            attempt=self._attempt,
            max_attempts=self.max_tries,
            successful=False,  # Not yet successful
            component=self.component,
            duration_ms=duration_ms
        )
        
        # Calculate and sleep for backoff delay
        self._delay()
    
    def success(self) -> None:
        """Mark the operation as successful."""
        self._succeeded = True
        
        # Record successful retry if this was a retry attempt
        if self._attempt > 1 and self._last_error is not None:
            duration_ms = int((time.time() - self._start_time) * 1000)
            record_retry(
                operation=self.operation,
                exception=self._last_error,
                attempt=self._attempt,
                max_attempts=self.max_tries,
                successful=True,
                component=self.component,
                duration_ms=duration_ms
            )
    
    def _delay(self) -> None:
        """Sleep for calculated backoff period."""
        delay = calculate_backoff(
            retry_attempt=self._attempt - 1,
            base_delay=self.base_delay,
            max_delay=self.max_delay
        )
        
        logger.info(
            f"Retry attempt {self._attempt}/{self.max_tries}, "
            f"waiting {delay:.2f}s after error: {str(self._last_error)}"
        )
        
        time.sleep(delay)
    
    def _get_caller_name(self) -> str:
        """Get the name of the calling function or module."""
        import inspect
        
        try:
            # Create a stack of frames
            frames = inspect.stack()
            
            # We need to find the caller of RetryContext
            # frames[0] is this function (_get_caller_name)
            # frames[1] is the __init__ method of RetryContext
            # We need to look further back to get proper caller
            
            # Position in the stack where the actual caller is likely to be
            LIKELY_CALLER_FRAME = 3
            
            if len(frames) <= LIKELY_CALLER_FRAME:
                return "unknown_operation"
            
            # First, try at the most likely position where actual business logic calls RetryContext
            frame = frames[LIKELY_CALLER_FRAME]
            
            # Capture information from frame
            module_name = frame.frame.f_globals.get('__name__', 'unknown_module')
            function_name = frame.function
            
            # Check if frame has 'self' in locals (instance method)
            if 'self' in frame.frame.f_locals:
                instance = frame.frame.f_locals['self']
                try:
                    class_name = instance.__class__.__name__
                    return f"{module_name}.{class_name}.{function_name}"
                except:
                    pass
            
            # Return module.function_name
            return f"{module_name}.{function_name}"
        
        except Exception as e:
            logger.debug(f"Error getting caller name: {str(e)}")
            return "unknown_operation"
        finally:
            # Prevent memory leaks by explicitly cleaning up
            del frames 