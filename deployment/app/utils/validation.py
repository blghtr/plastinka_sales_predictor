import io
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger("plastinka.validation")

class ValidationError(Exception):
    """Exception raised for validation errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


def validate_excel_file(file_content: bytes, expected_columns: List[str] = None) -> Tuple[bool, str]:
    """
    Validate that the provided file is a valid Excel file with the expected structure.
    
    Args:
        file_content: The content of the uploaded file
        expected_columns: Optional list of column names expected in the file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to read as Excel file
        df = pd.read_excel(io.BytesIO(file_content))
        
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


def validate_date_format(date_str: str, format_str: str = "%d.%m.%Y") -> Tuple[bool, Optional[datetime]]:
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


def validate_date_range(start_date: Union[str, datetime], 
                        end_date: Union[str, datetime],
                        format_str: str = "%d.%m.%Y",
                        min_date: Optional[datetime] = None,
                        max_date: Optional[datetime] = None,
                        max_range_days: Optional[int] = None) -> Tuple[bool, str, Optional[datetime], Optional[datetime]]:
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
            return False, f"Invalid start date format. Expected format: {format_str}", None, None
    
    if isinstance(end_date, str):
        is_valid, end_dt = validate_date_format(end_date, format_str)
        if not is_valid:
            return False, f"Invalid end date format. Expected format: {format_str}", None, None
    
    # Ensure start_date <= end_date
    if start_dt > end_dt:
        return False, "Start date must be before or equal to end date", None, None
    
    # Check if dates are within allowed range
    if min_date and start_dt < min_date:
        return False, f"Start date must be on or after {min_date.strftime(format_str)}", None, None
    
    if max_date and end_dt > max_date:
        return False, f"End date must be on or before {max_date.strftime(format_str)}", None, None
    
    # Check range size
    if max_range_days:
        range_days = (end_dt - start_dt).days
        if range_days > max_range_days:
            return False, f"Date range exceeds maximum allowed ({max_range_days} days)", None, None
    
    return True, "", start_dt, end_dt


def validate_forecast_date_range(start_date: Union[str, datetime], 
                                end_date: Union[str, datetime],
                                format_str: str = "%d.%m.%Y") -> Tuple[bool, str, Optional[datetime], Optional[datetime]]:
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
    max_date = datetime.now() + timedelta(days=365*2)  # Max 2 years in the future
    max_range_days = 365  # Max 1 year forecast range
    
    return validate_date_range(
        start_date, 
        end_date, 
        format_str,
        min_date, 
        max_date, 
        max_range_days
    )


def validate_historical_date_range(start_date: Union[str, datetime], 
                                  end_date: Union[str, datetime],
                                  format_str: str = "%d.%m.%Y") -> Tuple[bool, str, Optional[datetime], Optional[datetime]]:
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
    min_date = datetime(2000, 1, 1)  # Reasonable minimum date
    max_date = datetime.now()  # Can't be in the future
    max_range_days = 365 * 5  # Max 5 years of historical data at once
    
    return validate_date_range(
        start_date, 
        end_date, 
        format_str,
        min_date, 
        max_date, 
        max_range_days
    )


def validate_stock_file(file_content: bytes) -> Tuple[bool, str]:
    """
    Validate that the provided file is a valid stock data file.
    
    Args:
        file_content: The content of the uploaded file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    expected_columns = ["Штрихкод", "Исполнитель", "Альбом", "Остаток", "Дата"]
    return validate_excel_file(file_content, expected_columns)


def validate_sales_file(file_content: bytes) -> Tuple[bool, str]:
    """
    Validate that the provided file is a valid sales data file.
    
    Args:
        file_content: The content of the uploaded file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    expected_columns = ["Штрихкод", "Исполнитель", "Альбом", "Продажи", "Дата"]
    return validate_excel_file(file_content, expected_columns)


def validate_pagination_params(offset: int, limit: int) -> Tuple[int, int]:
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