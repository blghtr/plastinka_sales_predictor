"""Utility functions for database operations."""

import json
import logging
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

import asyncpg

from deployment.app.db.exceptions import DatabaseError

logger = logging.getLogger("plastinka.database")


def json_default_serializer(obj):
    """
    JSON serializer for objects not serializable by default json code
    """
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def generate_id() -> str:
    """Generate a unique ID for jobs or results"""
    return str(uuid.uuid4())


def _convert_query_to_postgres(query: str, params: tuple) -> tuple[str, tuple]:
    """
    Convert query with ? placeholders to PostgreSQL format with $1, $2, etc.
    
    Args:
        query: SQL query with ? placeholders
        params: Parameters tuple
        
    Returns:
        Tuple of (converted_query, params) where params remain unchanged
    """
    if not params:
        return query, params
    
    # Replace ? with $1, $2, etc.
    converted_query = query
    param_count = 1
    for _ in params:
        converted_query = converted_query.replace("?", f"${param_count}", 1)
        param_count += 1
    
    return converted_query, params


def _db_should_give_up(exception: Exception) -> bool:
    """Decide when DB retries should give up (non-retryable errors).

    Give up on integrity/programming errors (e.g., constraint failures, syntax),
    keep retrying on transient operational issues like connection errors.
    
    Priority order:
    1. PostgreSQL error codes (most reliable)
    2. Exception type checks
    3. String matching (fallback for localized messages)
    """
    try:
        if isinstance(exception, DatabaseError):
            orig = getattr(exception, "original_error", None)
            msg = (str(exception) or "").lower()
            
            # Check PostgreSQL error codes FIRST (most reliable)
            # According to PostgreSQL docs: 23xxx = Integrity constraint violations
            if isinstance(orig, asyncpg.exceptions.PostgresError):
                error_code = getattr(orig, "sqlstate", None)
                if error_code:
                    # 23xxx = All integrity constraint violations (permanent errors)
                    if error_code.startswith("23"):
                        return True  # Don't retry
                    # Explicit checks for common constraint violations:
                    # 23505 = unique_violation
                    # 23503 = foreign_key_violation
                    # 23514 = check_violation
                    # 23502 = not_null_violation
                    if error_code in ("23505", "23503", "23514", "23502"):
                        return True
                    # Client errors (40000-40007) - don't retry
                    if error_code.startswith("40"):
                        return True
                    # Server errors (50xxx, 53xxx, 57xxx) - retry
                    if error_code.startswith("50") or error_code.startswith("53") or error_code.startswith("57"):
                        return False
            
            # Type-based checks (asyncpg exception hierarchy)
            # According to asyncpg docs: UniqueViolationError is a subclass of IntegrityConstraintViolationError
            if isinstance(orig, asyncpg.exceptions.IntegrityConstraintViolationError):
                return True
            # Explicit check for UniqueViolationError (defensive)
            if isinstance(orig, asyncpg.exceptions.UniqueViolationError):
                return True
            
            # Syntax errors - don't retry
            if isinstance(orig, asyncpg.exceptions.PostgresSyntaxError):
                return True
            if "syntax error" in msg or "no such table" in msg:
                return True
            
            # String matching fallback (handle Russian/localized error messages)
            constraint_keywords = [
                "constraint", "violation", "unique", "duplicate", "key",
                "нарушает", "ограничение", "уникальности", "повторяющееся"
            ]
            if any(keyword in msg for keyword in constraint_keywords):
                return True
            
            # Connection errors - retry
            if isinstance(orig, asyncpg.exceptions.ConnectionDoesNotExistError):
                return False
            if "connection" in msg and ("refused" in msg or "timeout" in msg or "closed" in msg):
                return False
            
        return False  # Default: retry (conservative)
    except Exception:
        # Be conservative on error - allow retry
        return False


def get_batch_size() -> int:
    """Get the configured batch size for PostgreSQL queries.
    
    PostgreSQL supports large parameter lists, but we still
    use batching for performance reasons with very large lists.
    """
    try:
        from deployment.app.config import get_settings
        # Use a reasonable default batch size for PostgreSQL
        return getattr(get_settings(), "postgres_batch_size", 10000)
    except ImportError:
        # Fallback to default if settings not available
        return 10000


def split_ids_for_batching(ids: list[Any], batch_size: int | None = None) -> list[list[Any]]:
    """
    Split a list of IDs into batches for processing.

    Args:
        ids: List of IDs to split
        batch_size: Maximum size of each batch

    Returns:
        List of batches (each batch is a list of IDs)
    """
    if not ids:
        return []

    # Use configured batch size if not specified
    if batch_size is None:
        batch_size = get_batch_size()

    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


def check_batch_size_needed(ids: list[Any], batch_size: int | None = None) -> bool:
    """
    Check if batching is needed for the given list of IDs.

    Args:
        ids: List of IDs to check
        batch_size: Maximum allowed size

    Returns:
        True if batching is needed, False otherwise
    """
    # Use configured batch size if not specified
    if batch_size is None:
        batch_size = get_batch_size()

    return len(ids) > batch_size


def _is_path_safe(base_dir: str | Path, path_to_check: str | Path) -> bool:
    """
    Check if path_to_check is inside base_dir to prevent path traversal attacks.

    Args:
        base_dir: The base directory that should contain the path
        path_to_check: The path to validate

    Returns:
        bool: True if the path is safe (inside base_dir), False otherwise
    """
    try:
        # Resolve both paths to their absolute form
        resolved_base = Path(base_dir).resolve()
        resolved_path = Path(path_to_check).resolve()
        # Check if the resolved path is a subpath of the base directory
        return resolved_path.is_relative_to(resolved_base)
    except Exception:
        return False

