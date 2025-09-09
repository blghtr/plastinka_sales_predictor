import atexit
import logging
import logging.config
import os
import queue
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler
from pathlib import Path

DEFAULT_LOGGER_NAME = "plastinka_sales_predictor"
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# Environment variable defaults
DEFAULT_LOG_MODE = os.getenv("PLASTINKA_LOG_MODE", "simple").lower()
DEFAULT_LOG_LEVEL = os.getenv("PLASTINKA_LOG_LEVEL", "INFO")
DEFAULT_LOG_DIR = os.getenv("PLASTINKA_LOG_DIR")

# Singleton objects to ensure idempotency
_log_queue: queue.Queue | None = None
_queue_listener: QueueListener | None = None

def setup_logging(
    mode: str = 'simple',
    log_level: str = 'INFO',
    log_dir: str | Path | None = None,
    logger_name: str = DEFAULT_LOGGER_NAME,
) -> None:
    """
    Configure root logger with handlers based on the specified mode.

    Args:
        mode: Logging mode. 'simple' for direct synchronous logging (console + file),
              'async' for non-blocking queue-based logging.
        log_level: The logging level (e.g., 'INFO', 'DEBUG').
        log_dir: Directory to save log files.
        logger_name: The root logger name for the application.
    """
    if mode not in ['simple', 'async']:
        raise ValueError("Invalid logging mode. Choose 'simple' or 'async'.")

    root_logger = logging.getLogger(logger_name)
    root_logger.setLevel(log_level.upper())

    # Remove all existing handlers to prevent duplicates
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    
    # Also clear handlers for the root of all loggers if we are configuring our main logger
    if logger_name == DEFAULT_LOGGER_NAME:
        for h in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(h)


    if mode == 'simple':
        handlers = _build_simple_handlers(log_level, log_dir)
        for handler in handlers:
            root_logger.addHandler(handler)
    
    elif mode == 'async':
        global _log_queue, _queue_listener
        if _log_queue is None and _queue_listener is None:
            handlers = _build_async_handlers(log_level, log_dir)
            _log_queue = queue.Queue(-1)
            queue_handler = QueueHandler(_log_queue)
            root_logger.addHandler(queue_handler)

            _queue_listener = QueueListener(
                _log_queue, *handlers, respect_handler_level=True
            )
            _queue_listener.start()

            def _shutdown_logging():
                if _queue_listener:
                    _queue_listener.stop()
            
            atexit.register(_shutdown_logging)

def _build_simple_handlers(
    log_level: str, log_dir: str | Path | None
) -> list[logging.Handler]:
    """Build handlers for simple, synchronous logging."""
    formatter = logging.Formatter(_LOG_FORMAT)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level.upper())
    
    handlers = [console_handler]
    
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / "train.log"
        
        file_handler = TimedRotatingFileHandler(
            filename=file_path,
            when="D",
            interval=1,
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level.upper())
        handlers.append(file_handler)
        
    return handlers

def _build_async_handlers(
    log_level: str, log_dir: str | Path | None
) -> list[logging.Handler]:
    """Build handlers for async, queue-based logging."""
    if not log_dir:
        raise ValueError("log_dir must be provided for async logging mode.")
        
    formatter = logging.Formatter(_LOG_FORMAT)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level.upper())

    # Timed rotating file handler for async
    file_path = log_dir / "api.log"
    file_handler = TimedRotatingFileHandler(
        filename=file_path,
        when="D",
        interval=30,
        backupCount=6,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level.upper())

    return [console_handler, file_handler]

# ----------------------------------------------------------------------------
# Backward-compatible API: configure_logger
# ----------------------------------------------------------------------------

def configure_logger(
    logger_name: str | None = None,
    child_logger_name: str | None = None,
    *,
    mode: str | None = None,
    log_level: str | None = None,
    log_dir: str | Path | None = None,
) -> logging.Logger:
    """
    Backward-compatible wrapper that returns a configured logger.

    This preserves the old API used throughout the project. If the base logger
    has no handlers, it will configure logging automatically. Otherwise, it just
    returns the appropriate logger instance.

    Args:
        logger_name: Base logger name. Defaults to DEFAULT_LOGGER_NAME.
        child_logger_name: Optional child logger to return (hierarchical).
        mode: Optional override for logging mode ('simple' or 'async').
        log_level: Optional override for level.
        log_dir: Optional directory for log files (required for 'async').

    Returns:
        A configured Logger instance (base or child).
    """
    base_name = logger_name or DEFAULT_LOGGER_NAME
    base_logger = logging.getLogger(base_name)
    
    # Only configure if no handlers exist (prevents duplication)
    if not base_logger.handlers:
        resolved_mode = (mode or DEFAULT_LOG_MODE)
        resolved_level = (log_level or DEFAULT_LOG_LEVEL)
        resolved_dir: str | Path | None = log_dir or DEFAULT_LOG_DIR

        # Provide a sensible default directory for async mode if not supplied
        if resolved_mode == "async" and resolved_dir is None:
            resolved_dir = "logs"

        setup_logging(
            mode=resolved_mode,
            log_level=resolved_level,
            log_dir=resolved_dir,
            logger_name=base_name,
        )

    logger = base_logger
    if child_logger_name:
        logger = logger.getChild(child_logger_name)
    return logger
