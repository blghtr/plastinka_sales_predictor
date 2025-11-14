"""Database queries for feature management."""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any

import asyncpg

from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.core import execute_query, execute_many
from deployment.app.db.types import EXPECTED_REPORT_FEATURES, EXPECTED_REPORT_FEATURES_SET
from deployment.app.db.utils import split_ids_for_batching

logger = logging.getLogger("plastinka.database")


async def get_feature_dataframe(
    table_name: str,
    columns: list[str],
    connection: asyncpg.Connection,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
) -> list[dict]:
    """
    Fetches feature data from a specified table for a given date range.

    Selects 'multiindex_id', 'data_date', and the specified value columns
    from the given table, filtering by date if provided.

    Args:
        table_name: The name of the feature table (e.g., 'fact_sales').
        columns: A list of value columns to select (e.g., ['value'] or
                 ['availability', 'confidence']).
        connection: Required database connection.
        start_date: The start date for the range (YYYY-MM-DD).
        end_date: The end date for the range (YYYY-MM-DD).

    Returns:
        A list of dictionaries, where each dictionary represents a row of data.
        Returns an empty list if no data is found.
    """
    # Basic sanitization for table and column names
    def _is_safe_identifier(name: str) -> bool:
        return all(c.isalnum() or c == '_' for c in name)
    
    if not _is_safe_identifier(table_name) or not all(_is_safe_identifier(col) for col in columns):
        logger.error(f"Invalid characters in table or column names: {table_name}, {columns}")
        raise ValueError("Invalid table or column names provided.")

    select_columns = ["multiindex_id", "data_date"] + columns
    select_clause = ", ".join(f'"{c}"' for c in select_columns)  # Quote to be safe

    query = f'SELECT {select_clause} FROM "{table_name}"'
    
    params = []
    where_clauses = []
    param_num = 1

    if start_date:
        # Convert string to date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        where_clauses.append(f"data_date >= ${param_num}")
        params.append(start_date)
        param_num += 1
    if end_date:
        # Convert string to date if needed
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        where_clauses.append(f"data_date <= ${param_num}")
        params.append(end_date)
        param_num += 1

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    try:
        return await execute_query(query=query, connection=connection, params=tuple(params), fetchall=True) or []
    except DatabaseError as e:
        logger.error(f"Failed to get feature dataframe from {table_name}: {e}")
        raise


async def get_features_by_date_range(
    table: str,
    start_date: str | None,
    end_date: str | None,
    connection: asyncpg.Connection = None
) -> list[dict]:
    """Generic function to get features from a table by date range."""
    date_column = "data_date"

    query = f'SELECT * FROM "{table}"'
    params = []
    where_clauses = []
    param_num = 1

    if start_date:
        where_clauses.append(f"{date_column} >= ${param_num}")
        params.append(start_date)
        param_num += 1
    if end_date:
        where_clauses.append(f"{date_column} <= ${param_num}")
        params.append(end_date)
        param_num += 1

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += f" ORDER BY {date_column}"

    try:
        return await execute_query(query=query, connection=connection, params=tuple(params), fetchall=True) or []
    except DatabaseError as e:
        logger.error(f"Failed to get features from {table}: {e}")
        raise


async def insert_features_batch(
    table: str,
    params_list: list[tuple],
    connection: asyncpg.Connection = None
) -> None:
    """Insert a batch of feature records into the specified table."""
    if not params_list:
        return
    
    # PostgreSQL uses ON CONFLICT instead of INSERT OR REPLACE
    query = f'INSERT INTO "{table}" (multiindex_id, data_date, value) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING'
    await execute_many(query, params_list, connection=connection)


async def delete_features_by_table(
    table: str,
    connection: asyncpg.Connection = None
) -> None:
    """Delete all records from a feature table."""
    query = f'DELETE FROM "{table}"'
    await execute_query(query, connection=connection)


async def get_report_features(
    multiidx_ids: list[int] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    feature_subset: list[str] | None = None,
    connection: asyncpg.Connection = None,
) -> list[dict]:
    """
    Retrieves a specific subset of report features and associated product
    attributes, filtered by a specific month or a date range.

    If `multiidx_ids` is provided, it filters for those specific IDs. Otherwise,
    it fetches features for all available items within the date filter.

    Args:
        multiidx_ids: Optional list of multiindex_ids to retrieve features for.
        start_date: The start of the date range (inclusive).
        end_date: The end of the date range (inclusive).
        feature_subset: Optional list of specific features to retrieve from report_features
                        (e.g., ["avg_sales_items", "lost_sales_rub"]). If None, all are fetched.
        connection: Required database connection to use
        
    Returns:
        A list of dictionaries, where each dictionary contains product attributes
        and its requested features. Returns an empty list if no matching data is found.
    """
    # Build the SELECT clause
    select_parts = ["dmm.*", "rf.multiindex_id as rf_multiindex_id", "rf.data_date"]
    if feature_subset:
        selected_features = list(
            EXPECTED_REPORT_FEATURES_SET.intersection(feature_subset)
        )
    else:
        selected_features = EXPECTED_REPORT_FEATURES
    
    for feature in selected_features:
        select_parts.append(f"rf.{feature}")

    select_clause = "SELECT " + ", ".join(select_parts)
    from_clause = """
    FROM report_features rf
    JOIN dim_multiindex_mapping dmm ON rf.multiindex_id = dmm.multiindex_id
    """

    # Date filtering logic
    where_clauses = []
    params = []
    param_num = 1

    if start_date:
        where_clauses.append(f"rf.data_date >= ${param_num}")
        params.append(start_date.strftime("%Y-%m-%d"))
        param_num += 1
    if end_date:
        where_clauses.append(f"rf.data_date <= ${param_num}")
        params.append(end_date.strftime("%Y-%m-%d"))
        param_num += 1

    if multiidx_ids:
        # Use batching for large multiidx_ids lists
        all_results = []
        for batch in split_ids_for_batching(multiidx_ids, batch_size=1000):
            batch_placeholders = ", ".join(f"${i+param_num}" for i in range(len(batch)))
            batch_params = params + list(batch)
            
            where_clause_with_batch = where_clauses + [f"rf.multiindex_id IN ({batch_placeholders})"]
            query = f"{select_clause} {from_clause} WHERE {' AND '.join(where_clause_with_batch)}"
            
            batch_results = await execute_query(
                query, connection=connection, params=tuple(batch_params), fetchall=True
            ) or []
            all_results.extend(batch_results)
        
        return all_results
    else:
        # No multiidx_ids, single query for all items in the date range
        if where_clauses:
            query = f"{select_clause} {from_clause} WHERE {' AND '.join(where_clauses)}"
        else:
            query = f"{select_clause} {from_clause}"

        try:
            results = await execute_query(query=query, connection=connection, params=tuple(params), fetchall=True) or []
            return results
        except DatabaseError as e:
            logger.error(f"Failed to get report features: {e}")
            raise


async def insert_report_features(
    features_data: list[tuple],
    connection: asyncpg.Connection = None
) -> None:
    """
    Inserts a batch of report features into the report_features table.
    """
    query = """
        INSERT INTO report_features (
            data_date, multiindex_id, availability, confidence, masked_mean_sales_items, 
            masked_mean_sales_rub, lost_sales, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (data_date, multiindex_id) 
        DO UPDATE SET
            availability = EXCLUDED.availability,
            confidence = EXCLUDED.confidence,
            masked_mean_sales_items = EXCLUDED.masked_mean_sales_items,
            masked_mean_sales_rub = EXCLUDED.masked_mean_sales_rub,
            lost_sales = EXCLUDED.lost_sales,
            created_at = EXCLUDED.created_at
    """
    await execute_many(query, features_data, connection=connection)


async def adjust_dataset_boundaries(
    start_date: date | None = None,
    end_date: date | None = None,
    connection: asyncpg.Connection = None,
) -> date | None:
    """
    Adjust training dataset end date to the last available month in the database.

    Behavior (monthly granularity):
    - Database stores monthly dates (e.g., 01.MM.YYYY) rather than daily data.
    - If no dates are provided, returns the last available month (first day of month).
      The caller can then derive the prediction month as the next month.
    - If dates are provided, the last available month is computed within the range
      and the returned end date is clipped to that month if necessary.

    Returns:
        A date object representing the first day of the last available month within
        the optional range, or the provided end_date if no data is found.
    """
    # Base query to find the last available (monthly) date
    query = "SELECT MAX(data_date) as last_date FROM fact_sales"
    params = []
    param_num = 1

    # Add date range conditions if provided
    if start_date and end_date:
        # Normalize to first day of month for querying consistency
        start_norm = start_date.replace(day=1)
        end_norm = end_date.replace(day=1)
        query += f" WHERE data_date BETWEEN ${param_num} AND ${param_num + 1}"
        params.extend([start_norm, end_norm])
    elif start_date:
        start_norm = start_date.replace(day=1)
        query += f" WHERE data_date >= ${param_num}"
        params.append(start_norm)
    elif end_date:
        end_norm = end_date.replace(day=1)
        query += f" WHERE data_date <= ${param_num}"
        params.append(end_norm)

    try:
        result = await execute_query(query, connection=connection, params=tuple(params))

        if not result or not result.get("last_date"):
            # No data found — return the original end_date (may be None)
            logger.warning("No sales data found for date range, returning original end_date.")
            return end_date

        last_date = result["last_date"]
        if isinstance(last_date, str):
            last_available_month_first = date.fromisoformat(last_date).replace(day=1)
        elif isinstance(last_date, datetime):
            last_available_month_first = last_date.date().replace(day=1)
        else:
            last_available_month_first = last_date.replace(day=1)

        # Compute month-end for a given date
        def month_end(d: date) -> date:
            if d.month == 12:
                return date(d.year, 12, 31)
            return date(d.year, d.month + 1, 1) - timedelta(days=1)

        last_available_month_end = month_end(last_available_month_first)

        if end_date is None:
            # No explicit constraint: return the last available month END (last day of month)
            return last_available_month_end

        # Normalize provided end_date to its month end
        end_month_end = month_end(end_date)

        # Clip provided end to not exceed last available month end
        if end_month_end > last_available_month_end:
            return last_available_month_end
        return end_month_end

    except (DatabaseError, TypeError, ValueError) as e:
        logger.error(f"Failed to adjust dataset boundaries: {e}", exc_info=True)
        return end_date

