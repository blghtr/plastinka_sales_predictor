import os
import logging
from typing import Tuple, Optional, List
from fastapi import UploadFile

from deployment.app.utils.validation import ValidationError

logger = logging.getLogger("plastinka.file_validation")

# Default size limits
DEFAULT_MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", "10485760"))  # 10MB by default
MAX_EXCEL_FILE_SIZE = int(os.environ.get("MAX_EXCEL_FILE_SIZE", "5242880"))  # 5MB for Excel files

# Valid content types
VALID_EXCEL_CONTENT_TYPES = [
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/octet-stream",  # Sometimes Excel files get this generic type
]


async def validate_file_size(
    file: UploadFile, 
    max_size: int = DEFAULT_MAX_FILE_SIZE
) -> Tuple[bool, Optional[str]]:
    """
    Validate that the file size is within allowed limits.
    
    Args:
        file: The file to validate
        max_size: Maximum allowed file size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Get file size
    file_pos = file.file.tell()
    try:
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(file_pos)  # Reset to original position
        
        if file_size > max_size:
            error_msg = f"File too large: {file_size} bytes (max {max_size} bytes)"
            logger.warning(f"File size validation failed: {error_msg}")
            return False, error_msg
            
        return True, None
    except Exception as e:
        error_msg = f"Error checking file size: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


async def validate_content_type(
    file: UploadFile,
    valid_types: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Validate that the file's content type is allowed.
    
    Args:
        file: The file to validate
        valid_types: List of allowed content types
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    content_type = file.content_type or ""
    
    if content_type not in valid_types:
        # Check file extension as a fallback
        filename = file.filename or ""
        file_ext = os.path.splitext(filename.lower())[1]
        
        if valid_types == VALID_EXCEL_CONTENT_TYPES and file_ext in [".xls", ".xlsx"]:
            # Allow Excel files by extension if content_type is unreliable
            return True, None
            
        error_msg = f"Invalid content type: {content_type}"
        logger.warning(f"Content type validation failed: {error_msg}")
        return False, error_msg
        
    return True, None


async def validate_excel_file_upload(file: UploadFile) -> None:
    """
    Comprehensive validation for Excel file uploads.
    Raises ValidationError if any validation fails.
    
    Args:
        file: The Excel file to validate
    """
    # Check content type
    is_valid_type, type_error = await validate_content_type(file, VALID_EXCEL_CONTENT_TYPES)
    if not is_valid_type:
        raise ValidationError(
            message=f"Invalid Excel file: {type_error}",
            details={
                "filename": file.filename,
                "content_type": file.content_type
            }
        )
    
    # Check file size
    is_valid_size, size_error = await validate_file_size(file, MAX_EXCEL_FILE_SIZE)
    if not is_valid_size:
        raise ValidationError(
            message=f"Excel file too large: {size_error}",
            details={
                "filename": file.filename,
                "max_size_mb": MAX_EXCEL_FILE_SIZE / (1024 * 1024)
            }
        ) 