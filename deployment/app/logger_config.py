import atexit
import logging
import logging.config
import os
import queue
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler

from deployment.app.config import get_settings

# -----------------------------------------------------------------------------
# Centralised asynchronous logging configuration
# -----------------------------------------------------------------------------
#   • Non-blocking:   Root logger publishes records via QueueHandler → log_queue
#   • Background IO:  QueueListener drains the queue and delegates to concrete
#                     handlers (console + rotating file)
#   • Rotation:       Monthly using TimedRotatingFileHandler (approx. every 30 d)
#   • Retention:      Keep 6 archived log files
# -----------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# Singleton objects so configure_logging() can be called more than once safely
_log_queue: queue.Queue | None = None
_queue_listener: QueueListener | None = None


def _build_handlers() -> list[logging.Handler]:
    """Create concrete handlers that will be managed by QueueListener."""
    formatter = logging.Formatter(_LOG_FORMAT)
    settings = get_settings()

    # Console handler (stderr – integrates with Uvicorn/Starlette)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(settings.api.log_level.upper())

    # Timed rotating file handler – monthly rotation (~30 days)
    log_dir = settings.logs_dir
    file_path = os.path.join(log_dir, "api.log")
    file_handler = TimedRotatingFileHandler(
        filename=file_path,
        when="D",        # Rotate by interval of days
        interval=30,      # Approximate a month
        backupCount=6,    # Keep 6 old logs
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(settings.api.log_level.upper())

    return [console_handler, file_handler]


def configure_logging() -> None:
    """Configure root logger with asynchronous queue-based handlers.

    This is idempotent – calling it multiple times will not duplicate handlers.
    """
    global _log_queue, _queue_listener

    if _log_queue is not None and _queue_listener is not None:
        # Already configured – nothing to do.
        return

    # Concrete handlers managed by QueueListener
    handlers = _build_handlers()

    # Non-blocking queue handler attached to root logger
    _log_queue = queue.Queue(-1)  # Infinite size
    queue_handler = QueueHandler(_log_queue)
    settings = get_settings()

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.api.log_level.upper())

    # Remove pre-existing handlers (e.g., default Uvicorn handler) to prevent duplicate logs
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    root_logger.addHandler(queue_handler)

    # Start background listener
    _queue_listener = QueueListener(_log_queue, *handlers, respect_handler_level=True)
    _queue_listener.start()

    # Ensure graceful shutdown
    def _shutdown_logging():
        if _queue_listener:
            _queue_listener.stop()

    atexit.register(_shutdown_logging)
