import logging
import os
from logging.handlers import TimedRotatingFileHandler

DEFAULT_LOG_FILE = 'logs/train.log'
DEFAULT_LOGGER_NAME = 'plastinka_sales_predictor'


def configure_logger(
        logger_name: str | None = None,
        child_logger_name: str | None = None,
        log_file: str | None = None
) -> logging.Logger:
    """Configure logger with unique file for each run.

    Args:
        logger_name: Name of the main logger
        child_logger_name: Name of child logger if needed
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    if not logger_name:
        logger_name = DEFAULT_LOGGER_NAME
    if not log_file:
        log_file = DEFAULT_LOG_FILE

    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            if os.path.dirname(log_file):
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

            file_handler = TimedRotatingFileHandler(
                log_file,
                backupCount=5,
                encoding='utf-8',
                when='d',
                interval=1
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    if child_logger_name:
        logger = logger.getChild(child_logger_name)

    return logger
