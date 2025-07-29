import io
import logging
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger("plastinka.validation")


class ValidationError(Exception):
    """Exception raised for validation errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


def validate_excel_file(
    file_content: bytes, expected_columns: list[str] = None
) -> tuple[bool, str]:
    """
    Validate that the provided file is a valid Excel file with the expected structure.

    Args:
        file_content: The content of the uploaded file
        expected_columns: Optional list of column names expected in the file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to read as Excel file, specify engine
        df = pd.read_excel(io.BytesIO(file_content), engine="openpyxl")

        # Check if dataframe is empty
        if df.empty:
            return False, "Excel file is empty"

        # Check for expected columns if provided
        if expected_columns:
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"

        return True, ""
    except Exception as e:
        logger.error(f"Excel validation error: {str(e)}", exc_info=True)
        return False, f"Invalid Excel file: {str(e)}"


def validate_csv_file(
    file_content: bytes, expected_columns: list[str] = None
) -> tuple[bool, str]:
    """
    Validate that the provided file is a valid CSV file with the expected structure.

    Args:
        file_content: The content of the uploaded file
        expected_columns: Optional list of column names expected in the file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try different encodings
        for encoding in ['utf-8', 'windows-1251', 'cp1252']:
            try:
                content_str = file_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return False, "Unable to decode CSV file. Please ensure it's in UTF-8 or Windows-1251 encoding."

        # Try to read as CSV file with automatic separator detection
        df = pd.read_csv(io.StringIO(content_str), sep=None, engine='python')

        # Check if dataframe is empty
        if df.empty:
            return False, "CSV file is empty"

        # Check for expected columns if provided
        if expected_columns:
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"

        return True, ""
    except Exception as e:
        logger.error(f"CSV validation error: {str(e)}", exc_info=True)
        return False, f"Invalid CSV file: {str(e)}"


def validate_data_file_content(
    file_content: bytes, expected_columns: list[str] = None, filename: str = ""
) -> tuple[bool, str]:
    """
    Universal validation for data file content (Excel or CSV).
    Automatically detects file type and applies appropriate validation.

    Args:
        file_content: The content of the uploaded file
        expected_columns: Optional list of column names expected in the file
        filename: Filename to help determine file type

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Determine file type by extension
    file_ext = os.path.splitext(filename.lower())[1] if filename else ""

    if file_ext in [".xls", ".xlsx"]:
        return validate_excel_file(file_content, expected_columns)
    elif file_ext == ".csv":
        return validate_csv_file(file_content, expected_columns)
    else:
        # Try Excel first, then CSV as fallback
        is_valid, error = validate_excel_file(file_content, expected_columns)
        if is_valid:
            return True, ""

        is_valid_csv, error_csv = validate_csv_file(file_content, expected_columns)
        if is_valid_csv:
            return True, ""

        return False, f"File is neither valid Excel nor CSV. Excel error: {error}. CSV error: {error_csv}"


def validate_date_format(
    date_str: str, format_str: str = "%d.%m.%Y"
) -> tuple[bool, datetime | None]:
    """
    Validate that the provided string is a valid date in the expected format.

    Args:
        date_str: Date string to validate
        format_str: Expected date format

    Returns:
        Tuple of (is_valid, parsed_date or None)
    """
    try:
        parsed_date = datetime.strptime(date_str, format_str)
        return True, parsed_date
    except ValueError:
        return False, None


def validate_date_range(
    start_date: str | datetime,
    end_date: str | datetime,
    format_str: str = "%d.%m.%Y",
    min_date: datetime | None = None,
    max_date: datetime | None = None,
    max_range_days: int | None = None,
) -> tuple[bool, str, datetime | None, datetime | None]:
    """
    Validate a date range with comprehensive checks:
    - Validate date formats
    - Ensure start_date <= end_date
    - Check if dates are within min/max allowed range
    - Validate range size doesn't exceed maximum

    Args:
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        format_str: Date format if string dates provided
        min_date: Optional minimum allowed date
        max_date: Optional maximum allowed date
        max_range_days: Optional maximum allowed range in days

    Returns:
        Tuple of (is_valid, error_message, parsed_start_date, parsed_end_date)
    """
    # Convert string dates to datetime objects
    start_dt = start_date
    end_dt = end_date

    if isinstance(start_date, str):
        is_valid, start_dt = validate_date_format(start_date, format_str)
        if not is_valid:
            return (
                False,
                f"Invalid start date format. Expected format: {format_str}",
                None,
                None,
            )

    if isinstance(end_date, str):
        is_valid, end_dt = validate_date_format(end_date, format_str)
        if not is_valid:
            return (
                False,
                f"Invalid end date format. Expected format: {format_str}",
                None,
                None,
            )

    # Ensure start_date <= end_date
    if start_dt > end_dt:
        return False, "Start date must be before or equal to end date", None, None

    # Check if dates are within allowed range
    if min_date and start_dt < min_date:
        return (
            False,
            f"Start date must be on or after {min_date.strftime(format_str)}",
            None,
            None,
        )

    if max_date and end_dt > max_date:
        return (
            False,
            f"End date must be on or before {max_date.strftime(format_str)}",
            None,
            None,
        )

    # Check range size
    if max_range_days:
        range_days = (end_dt - start_dt).days
        if range_days > max_range_days:
            return (
                False,
                f"Date range exceeds maximum allowed ({max_range_days} days)",
                None,
                None,
            )

    return True, "", start_dt, end_dt


def validate_forecast_date_range(
    start_date: str | datetime, end_date: str | datetime, format_str: str = "%d.%m.%Y"
) -> tuple[bool, str, datetime | None, datetime | None]:
    """
    Validate forecast date range with specific constraints for forecasting.

    Args:
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        format_str: Date format if string dates provided

    Returns:
        Tuple of (is_valid, error_message, parsed_start_date, parsed_end_date)
    """
    # Set forecast-specific constraints
    min_date = datetime.now()
    max_date = datetime.now() + timedelta(days=365 * 2)  # Max 2 years in the future
    max_range_days = 365  # Max 1 year forecast range

    return validate_date_range(
        start_date, end_date, format_str, min_date, max_date, max_range_days
    )


def validate_historical_date_range(
    start_date: str | datetime, end_date: str | datetime, format_str: str = "%d.%m.%Y"
) -> tuple[bool, str, datetime | None, datetime | None]:
    """
    Validate historical date range with specific constraints for historical data.

    Args:
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        format_str: Date format if string dates provided

    Returns:
        Tuple of (is_valid, error_message, parsed_start_date, parsed_end_date)
    """
    # Set historical data specific constraints
    min_date = datetime(1950, 1, 1)  # Reasonable minimum date
    max_date = datetime.now()  # Can't be in the future
    max_range_days = 365 * 5  # Max 5 years of historical data at once

    return validate_date_range(
        start_date, end_date, format_str, min_date, max_date, max_range_days
    )


def validate_stock_file(file_content: bytes, filename: str = "") -> tuple[bool, str]:
    """
    Validate that the provided file is a valid stock data file (Excel or CSV).

    Args:
        file_content: The content of the uploaded file
        filename: Filename to help determine file type

    Returns:
        Tuple of (is_valid, error_message)
    """
    expected_columns = [
        "Штрихкод",
        "Исполнитель",
        "Альбом",
        "Дата создания",
        "Экземпляры",
    ]
    return validate_data_file_content(file_content, expected_columns, filename)


def validate_sales_file(file_content: bytes, filename: str = "") -> tuple[bool, str]:
    """
    Validate that the provided file is a valid sales data file (Excel or CSV).

    Args:
        file_content: The content of the uploaded file
        filename: Filename to help determine file type

    Returns:
        Tuple of (is_valid, error_message)
    """
    expected_columns = [
        "Barcode",
        "Исполнитель",
        "Альбом",
        "Дата добавления",
        "Дата продажи",
    ]
    return validate_data_file_content(file_content, expected_columns, filename)


def validate_pagination_params(offset: int, limit: int) -> tuple[int, int]:
    """
    Validate and normalize pagination parameters.

    Args:
        offset: Starting position
        limit: Number of items to return

    Returns:
        Tuple of (normalized_offset, normalized_limit)
    """
    normalized_offset = max(0, offset)
    normalized_limit = max(1, min(1000, limit))

    return normalized_offset, normalized_limit


def validate_date_range_or_none(start_date, end_date, format_str="%Y-%m-%d"):
    """
    Validates that start_date and end_date (if both provided) form a valid range.
    Raises ValueError if invalid. Does nothing if either is None.
    """
    if start_date and end_date:
        is_valid, error_msg, _, _ = validate_historical_date_range(
            start_date.strftime(format_str),
            end_date.strftime(format_str),
            format_str=format_str
        )
        if not is_valid:
            raise ValueError(error_msg)
