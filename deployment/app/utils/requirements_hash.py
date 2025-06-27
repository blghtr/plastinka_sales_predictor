"""
Utilities for working with requirements.txt hashes for DataSphere job cloning optimization.

This module provides functions to:
- Calculate SHA256 hashes of requirements.txt files
- Compare requirements between jobs
- Manage requirements file paths
"""

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_requirements_hash(requirements_path: str) -> str:
    """
    Calculate SHA256 hash of requirements.txt file.
    
    Args:
        requirements_path: Path to the requirements.txt file
        
    Returns:
        SHA256 hash in hex format
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        OSError: If there are file reading errors
    """
    if not os.path.exists(requirements_path):
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")
    
    if not os.path.isfile(requirements_path):
        raise OSError(f"Path is not a file: {requirements_path}")
    
    try:
        hash_sha256 = hashlib.sha256()
        with open(requirements_path, 'rb') as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        result_hash = hash_sha256.hexdigest()
        return result_hash
        
    except IOError as e:
        logger.error(f"Error reading requirements file {requirements_path}: {e}")
        raise OSError(f"Failed to read requirements file: {e}") from e


def get_requirements_file_path() -> str:
    """
    Return path to requirements.txt file from settings.
    
    Returns:
        Absolute path to requirements.txt
        
    Raises:
        FileNotFoundError: If the requirements file doesn't exist
        OSError: If path resolution fails
    """
    from deployment.app.config import settings
    
    try:
        # Get the requirements file path from settings
        requirements_path = settings.datasphere.requirements_file_path
        
        # Convert to absolute path
        if not os.path.isabs(requirements_path):
            # If relative path, resolve relative to project root
            project_root = settings.project_root_dir
            requirements_path = os.path.join(project_root, requirements_path)
        
        requirements_path = os.path.abspath(requirements_path)
        
        # Verify the file exists
        if not os.path.exists(requirements_path):
            raise FileNotFoundError(f"Requirements file not found at configured path: {requirements_path}")
        
        return requirements_path
        
    except AttributeError as e:
        logger.error(f"Settings configuration error: {e}")
        raise OSError(f"Failed to get requirements file path from settings: {e}") from e
    except Exception as e:
        logger.error(f"Error resolving requirements file path: {e}")
        raise OSError(f"Failed to resolve requirements file path: {e}") from e


def requirements_changed_since_job(job_id: str, current_hash: str) -> bool:
    """
    Check if requirements have changed since the specified job was executed.
    
    Args:
        job_id: ID of the job to compare against
        current_hash: Current hash of requirements
        
    Returns:
        True if requirements have changed, False if they haven't
        
    Raises:
        ValueError: If job_id is invalid or job not found
    """
    from deployment.app.db.database import get_job_requirements_hash
    
    if not job_id:
        raise ValueError("job_id cannot be empty")
    
    if not current_hash:
        raise ValueError("current_hash cannot be empty")
    
    try:
        # Get the requirements hash for the specified job
        job_hash = get_job_requirements_hash(job_id)
        
        if job_hash is None:
            logger.warning(f"No requirements hash found for job {job_id}")
            return True  # Assume changed if we can't find the hash
        
        changed = job_hash != current_hash
        return changed
        
    except Exception as e:
        logger.error(f"Error comparing requirements for job {job_id}: {e}")
        return True  # Assume changed on error for safety


def get_requirements_hash_for_current_state() -> Optional[str]:
    """
    Get the SHA256 hash for the current requirements.txt file.
    
    Returns:
        SHA256 hash of the current requirements file, or None if calculation fails
    """
    try:
        requirements_path = get_requirements_file_path()
        return calculate_requirements_hash(requirements_path)
    except Exception as e:
        logger.error(f"Failed to calculate current requirements hash: {e}")
        return None


def validate_requirements_file(requirements_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a requirements file exists and is readable.
    
    Args:
        requirements_path: Path to the requirements file to validate
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    try:
        if not requirements_path:
            return False, "Requirements path is empty"
        
        if not os.path.exists(requirements_path):
            return False, f"Requirements file does not exist: {requirements_path}"
        
        if not os.path.isfile(requirements_path):
            return False, f"Path is not a file: {requirements_path}"
        
        # Try to read the file to ensure it's accessible
        with open(requirements_path, 'r', encoding='utf-8') as f:
            f.read(1)  # Read just one character to test accessibility
        
        return True, None
        
    except PermissionError:
        return False, f"Permission denied reading requirements file: {requirements_path}"
    except UnicodeDecodeError:
        return False, f"Requirements file contains invalid UTF-8: {requirements_path}"
    except Exception as e:
        return False, f"Error validating requirements file: {e}" 