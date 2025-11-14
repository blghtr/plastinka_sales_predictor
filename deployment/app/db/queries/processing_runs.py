"""Database queries for processing runs management."""

import logging
from datetime import datetime
from typing import Any

import asyncpg

from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.core import execute_query

logger = logging.getLogger("plastinka.database")


async def create_processing_run(
    start_time: datetime,
    status: str,
    source_files: str,
    end_time: datetime = None,
    connection: asyncpg.Connection = None,
) -> int:
    """
    Create a processing run record

    Args:
        start_time: Start time of processing
        status: Status of the run
        source_files: Comma-separated list of source files
        end_time: Optional end time of processing
        connection: Required database connection to use

    Returns:
        Generated run ID
    """
    query = """
    INSERT INTO processing_runs (start_time, status, source_files, end_time)
    VALUES ($1, $2, $3, $4)
    RETURNING run_id
    """

    params = (
        start_time,
        status,
        source_files,
        end_time if end_time else None,
    )

    try:
        result = await execute_query(query, connection=connection, params=params)
        if result:
            return result["run_id"]
        raise DatabaseError("Failed to get run_id from INSERT")
    except DatabaseError:
        logger.error("Failed to create processing run")
        raise


async def update_processing_run(
    run_id: int,
    status: str,
    end_time: datetime = None,
    connection: asyncpg.Connection = None,
) -> None:
    """
    Update a processing run

    Args:
        run_id: ID of the run to update
        status: New status
        end_time: Optional end time
        connection: Required database connection to use
    """
    set_clauses = ["status = $1"]
    params = [status]
    param_num = 2

    if end_time:
        set_clauses.append(f"end_time = ${param_num}")
        params.append(end_time)
        param_num += 1

    query = f"UPDATE processing_runs SET {', '.join(set_clauses)} WHERE run_id = ${param_num}"
    params.append(run_id)

    try:
        await execute_query(query, connection=connection, params=tuple(params))
    except DatabaseError:
        logger.error(f"Failed to update processing run {run_id}")
        raise

