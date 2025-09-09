import atexit
import logging
import logging.config
import os
import queue
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler

from deployment.app.config import get_settings
from plastinka_sales_predictor.logger_setup import setup_logging


def configure_logging() -> None:
    """Configure logging for the deployment environment."""
    settings = get_settings()
    setup_logging(
        mode='async',
        log_level=settings.api.log_level,
        log_dir=settings.logs_dir,
    )
