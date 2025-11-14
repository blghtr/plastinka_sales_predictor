"""Core query execution functions."""

import logging
from typing import Any

import asyncpg

from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.utils import _convert_query_to_postgres, _db_should_give_up
from deployment.app.utils.retry import retry_with_backoff

logger = logging.getLogger("plastinka.database")


@retry_with_backoff(
    max_tries=3,
    base_delay=1.0,
    max_delay=10.0,
    component="database_query",
    giveup_func=_db_should_give_up
)
async def execute_query(
    query: str,
    connection: asyncpg.Connection,
    params: tuple = (),
    fetchall: bool = False,
) -> list[dict] | dict | None:
    """
    Execute a query and optionally return results.
    This function *always* requires an explicit `connection` parameter and
    *never* creates or closes its own connections. It operates purely within
    the transaction context provided by the caller.

    Args:
        query: SQL query with placeholders (? will be converted to $1, $2, etc.)
        params: Parameters for the query
        fetchall: Whether to fetch all results or just one
        connection: The asyncpg database connection to use. This function will NOT commit or rollback.

    Returns:
        Query results as dict or list of dicts, or None for operations

    Raises:
        DatabaseError: If database operation fails
    """
    try:
        # Convert SQLite-style ? placeholders to PostgreSQL $1, $2, etc.
        pg_query, pg_params = _convert_query_to_postgres(query, params)
        
        # Determine if this is a SELECT query or has RETURNING clause
        query_upper = pg_query.strip().upper()
        is_select = (
            query_upper.startswith("SELECT") or
            "SELECT" in query_upper[:50] or
            query_upper.startswith("WITH")
        )
        has_returning = "RETURNING" in query_upper
        
        if is_select:
            if fetchall:
                rows = await connection.fetch(pg_query, *pg_params)
                # Convert asyncpg Record objects to dicts
                result = [dict(row) for row in rows] if rows else []
            else:
                row = await connection.fetchrow(pg_query, *pg_params)
                result = dict(row) if row else None
        elif has_returning:
            # For INSERT/UPDATE with RETURNING clause
            row = await connection.fetchrow(pg_query, *pg_params)
            result = dict(row) if row else None
        else:
            # For INSERT, UPDATE, DELETE, etc. without RETURNING
            await connection.execute(pg_query, *pg_params)
            result = None

        return result

    except asyncpg.PostgresError as e:
        safe_params = "..." if params else "()"
        logger.error(
            f"Database error in query: {query[:100]} with params: {safe_params}: {str(e)}",
            exc_info=True,
        )

        raise DatabaseError(
            message=f"Database operation failed: {str(e)}",
            query=query,
            params=params,
            original_error=e,
        ) from e
    except Exception as e:
        safe_params = "..." if params else "()"
        logger.error(
            f"Unexpected error in query: {query[:100]} with params: {safe_params}: {str(e)}",
            exc_info=True,
        )
        raise DatabaseError(
            message=f"Database operation failed: {str(e)}",
            query=query,
            params=params,
            original_error=e,
        ) from e


@retry_with_backoff(
    max_tries=3,
    base_delay=1.0,
    max_delay=10.0,
    component="database_batch",
    giveup_func=_db_should_give_up
)
async def execute_many(
    query: str,
    params_list: list[tuple],
    connection: asyncpg.Connection,
) -> None:
    """
    Execute a query with multiple parameter sets.
    This function *always* requires an explicit `connection` parameter and
    *never* creates or closes its own connections. It operates purely within
    the transaction context provided by the caller.

    Args:
        query: SQL query with placeholders (? will be converted to $1, $2, etc.)
        params_list: List of parameter tuples
        connection: The asyncpg database connection to use. This function will NOT commit or rollback.

    Raises:
        DatabaseError: If database operation fails
    """
    if not params_list:
        return

    try:
        # Convert query for first set of params to determine placeholder count
        if params_list:
            pg_query, _ = _convert_query_to_postgres(query, params_list[0])
            
            # Execute all parameter sets
            for params in params_list:
                pg_query_iter, pg_params = _convert_query_to_postgres(query, params)
                await connection.execute(pg_query_iter, *pg_params)

    except asyncpg.PostgresError as e:
        logger.error(
            f"Database error in executemany: {query[:100]}, params count: {len(params_list)}: {str(e)}",
            exc_info=True,
        )

        raise DatabaseError(
            message=f"Batch database operation failed: {str(e)}",
            query=query,
            original_error=e,
        ) from e
    except Exception as e:
        logger.error(
            f"Unexpected error in executemany: {query[:100]}, params count: {len(params_list)}: {str(e)}",
            exc_info=True,
        )
        raise DatabaseError(
            message=f"Batch database operation failed: {str(e)}",
            query=query,
            original_error=e,
        ) from e

