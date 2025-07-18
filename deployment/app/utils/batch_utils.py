"""
Utility functions for handling SQL queries with large parameter lists.
Provides batching functionality to avoid SQLite's variable limit.
"""

import logging
from typing import Any, List, Optional, Union
import sqlite3

from deployment.app.db.database import execute_query, execute_many

logger = logging.getLogger(__name__)

# SQLite default limit is 999, use 900 for safety
# This can be overridden by settings.sqlite_max_variables
SQLITE_MAX_VARIABLES = 900


def get_batch_size() -> int:
    """Get the configured batch size for SQLite queries."""
    try:
        from deployment.app.config import get_settings
        return get_settings().sqlite_max_variables
    except ImportError:
        # Fallback to default if settings not available
        return SQLITE_MAX_VARIABLES


def execute_query_with_batching(
    query_template: str,
    ids: List[Any],
    batch_size: Optional[int] = None,
    connection: Optional[sqlite3.Connection] = None,
    fetchall: bool = True,
    placeholder_name: str = "placeholders"
) -> List[Any]:
    """
    Execute query with IN clause using batching to avoid SQLite variable limit.
    
    Args:
        query_template: SQL query with {placeholder_name} placeholder for IN clause
        ids: List of IDs to use in IN clause
        batch_size: Maximum number of IDs per batch (default: 900)
        connection: Database connection
        fetchall: Whether to fetch all results
        placeholder_name: Name of placeholder in query_template (default: "placeholders")
    
    Returns:
        Combined results from all batches
        
    Example:
        query = "SELECT * FROM table WHERE id IN ({placeholders})"
        results = execute_query_with_batching(query, [1, 2, 3, ...], connection=conn)
    """
    if not ids:
        return []
    
    # Use configured batch size if not specified
    if batch_size is None:
        batch_size = get_batch_size()
    
    all_results = []
    
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        placeholders = ", ".join("?" * len(batch_ids))
        
        # Replace the placeholder in the query template
        query = query_template.format(**{placeholder_name: placeholders})
        
        try:
            batch_results = execute_query(
                query, tuple(batch_ids), fetchall=fetchall, connection=connection
            )
            
            if batch_results:
                if fetchall:
                    all_results.extend(batch_results)
                else:
                    all_results.append(batch_results)
                    
        except Exception as e:
            logger.error(f"Error executing batch query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Batch size: {len(batch_ids)}")
            raise
    
    return all_results


def execute_many_with_batching(
    query: str,
    params_list: List[tuple],
    batch_size: Optional[int] = None,
    connection: Optional[sqlite3.Connection] = None
) -> None:
    """
    Execute many queries using batching to avoid SQLite variable limit.
    
    Args:
        query: SQL query to execute multiple times
        params_list: List of parameter tuples
        batch_size: Maximum number of parameters per batch
        connection: Database connection
    """
    if not params_list:
        return
    
    # Use configured batch size if not specified
    if batch_size is None:
        batch_size = get_batch_size()
    
    for i in range(0, len(params_list), batch_size):
        batch_params = params_list[i:i + batch_size]
        
        try:
            execute_many(query, batch_params, connection=connection)
        except Exception as e:
            logger.error(f"Error executing batch insert/update: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Batch size: {len(batch_params)}")
            raise


def split_ids_for_batching(ids: List[Any], batch_size: Optional[int] = None) -> List[List[Any]]:
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


def check_batch_size_needed(ids: List[Any], batch_size: Optional[int] = None) -> bool:
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