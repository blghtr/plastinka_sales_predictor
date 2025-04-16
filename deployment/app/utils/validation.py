import io
import pandas as pd
from datetime import datetime
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