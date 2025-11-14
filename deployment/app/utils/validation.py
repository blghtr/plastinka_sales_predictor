import io
import logging
import os
from datetime import datetime, timedelta
from typing import Any
from plastinka_sales_predictor.data_preparation import read_data_file

logger = logging.getLogger("plastinka.validation")

# Default size limits
DEFAULT_MAX_FILE_SIZE = int(
    os.environ.get("MAX_FILE_SIZE", "10485760")
)  # 10MB by default
MAX_EXCEL_FILE_SIZE = int(
    os.environ.get("MAX_EXCEL_FILE_SIZE", "5242880")
)  # 5MB for Excel files
MAX_CSV_FILE_SIZE = int(
    os.environ.get("MAX_CSV_FILE_SIZE", "5242880")
)  # 5MB for CSV files

# Valid content types
VALID_CONTENT_TYPES = [
    ".xlsx",
    ".xls",
    ".csv"
]


class ValidationError(Exception):
    """Exception raised for validation errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


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
        start_date, 
        end_date, 
        format_str, 
        min_date, 
        max_date, 
        max_range_days
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


def validate_file_size(
    file: io.BytesIO, 
    max_size: int = DEFAULT_MAX_FILE_SIZE
) -> tuple[bool, str | None]:
    """
    Validate that the file size is within allowed limits.

    Args:
        file: The file to validate
        max_size: Maximum allowed file size in bytes

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Get file size
    file_size = file.getbuffer().nbytes
    if file_size > max_size:
        error_msg = f"File too large: {file_size} bytes (max {max_size} bytes)"
        logger.warning(f"File size validation failed: {error_msg}")
        return False, error_msg
    return True, None


def validate_content_type(
    path: str
) -> tuple[bool, str | None]:
    """
    Validate that the file's content type is allowed.

    Args:
        path: Path to the file

    Returns:
        Tuple of (is_valid, error_message)
    """
    ext = (os.path.splitext(path)[1] or "").lower()
    if ext not in VALID_CONTENT_TYPES:
        error_msg = f"Invalid content type: {ext}"
        logger.warning(f"Content type validation failed: {error_msg}")
        return False, error_msg

    return True, None


def validate_data_file_content(
    file: io.BytesIO,
    expected_columns: list[str] = None,
    path: str = None
) -> tuple[bool, str]:
    """
    Validate that the provided file is a valid CSV\Excel file with the expected structure.

    Args:
        file: The content of the uploaded file
        expected_columns: Optional list of column names expected in the file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        df = read_data_file(
            file=file,
            path=path
        )

        # Check if dataframe is empty
        if df.empty:
            return False, "File is empty"

        # Check for expected columns if provided
        if expected_columns:
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"

        return True, ""
    except Exception as e:
        logger.error(f"File validation error: {str(e)}", exc_info=True)
        return False, f"Invalid file: {str(e)}"


def validate_data_file_upload(file: io.BytesIO, path: str) -> tuple[bool, str]:
    """
    Validate that the provided file is a valid data file (Excel or CSV) of a valid size.

    Args:
        file: The uploaded file
        path: Path to the file

    Returns:
        Tuple of (is_valid, error_message)
    """
    size_valid, size_error = validate_file_size(file)
    if not size_valid:
        return False, size_error
    content_valid, content_error = validate_content_type(path)
    if not content_valid:
        return False, content_error
    return True, None


def validate_stock_file(file: io.BytesIO, path: str = None) -> tuple[bool, str]:
    """
    Validate that the provided file is a valid stock data file (Excel or CSV).

    Args:
        file: The uploaded file
        path: Path to the file

    Returns:
        Tuple of (is_valid, error_message)
    """
    expected_columns = [
        "barcode",
        "artist",
        "album",
        "cover_type",
        "price",
        "release_type",
        "recording_year",
        "release_year",
        "style",
        "created_date",
        "count",
    ]
    upload_valid, upload_error = validate_data_file_upload(file, path)
    if not upload_valid:
        return False, upload_error
    file.seek(0)
    return validate_data_file_content(file, expected_columns, path)


def validate_sales_file(file: io.BytesIO, path: str = None) -> tuple[bool, str]:
    """
    Validate that the provided file is a valid sales data file (Excel or CSV).

    Args:
        file: The content of the uploaded file
        path: Path to the file

    Returns:
        Tuple of (is_valid, error_message)
    """
    expected_columns = [
        "barcode",
        "artist",
        "album",
        "cover_type",
        "price",
        "release_type",
        "recording_year",
        "release_year",
        "style",
        "created_date",
        "sold_date",
    ]
    upload_valid, upload_error = validate_data_file_upload(file, path)
    if not upload_valid:
        return False, upload_error
    file.seek(0)
    return validate_data_file_content(file, expected_columns, path)


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


def validate_date_attributes(*date_attrs, param_names=None):
    """
    Validates that date attributes are valid date objects.
    
    Args:
        *date_attrs: Variable number of date values to validate
        param_names: Optional list of parameter names for error messages (same order as date_attrs)
    
    Raises:
        ValueError: If any date is not None and not a date object
    
    Example:
        validate_date_attributes(start_date, end_date, param_names=["start_date", "end_date"])
    """
    from datetime import date
    
    if param_names is None:
        param_names = [f"date_{i}" for i in range(len(date_attrs))]
    
    for date_attr, param_name in zip(date_attrs, param_names):
        if date_attr is not None and not isinstance(date_attr, date):
            raise ValueError(
                f"Invalid {param_name} type. Expected date object, got {type(date_attr).__name__}"
            )


def validate_date_range_or_none(start_date, end_date, format_str="%Y-%m-%d"):
    """
    Validates that start_date and end_date (if both provided) form a valid range.
    Also validates that dates are valid date objects.
    Raises ValueError if invalid. Does nothing if either is None.
    """
    from datetime import date
    
    # Validate that dates are date objects if provided
    validate_date_attributes(
        start_date, end_date,
        param_names=["start_date", "end_date"]
    )
    
    # Validate date range if both provided
    if start_date and end_date:
        is_valid, error_msg, _, _ = validate_historical_date_range(
            start_date.strftime(format_str),
            end_date.strftime(format_str),
            format_str=format_str
        )
        if not is_valid:
            raise ValueError(error_msg)
