"""Database queries for retry events management."""

import logging
from typing import Any

import asyncpg

from deployment.app.db.queries.core import execute_query

logger = logging.getLogger("plastinka.database")


async def insert_retry_event(event: dict[str, Any], connection: asyncpg.Connection) -> None:
    """Insert a single retry event into retry_events table.

    Args:
        event: Dict with keys matching retry_events columns.
        connection: Required database connection.
    """
    event = event.copy()
    event["successful"] = event.get("successful", False)  # PostgreSQL boolean

    query = """
        INSERT INTO retry_events 
        (timestamp, component, operation, attempt, max_attempts, successful, duration_ms, exception_type, exception_message)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    """
    
    params = (
        event.get("timestamp"),
        event.get("component"),
        event.get("operation"),
        event.get("attempt"),
        event.get("max_attempts"),
        event.get("successful"),
        event.get("duration_ms"),
        event.get("exception_type"),
        event.get("exception_message"),
    )
    
    try:
        await execute_query(query, connection=connection, params=params)
    except Exception as e:
        logger.error(
            f"[insert_retry_event] Error during insert for event {event.get('operation')}: {e}",
            exc_info=True,
        )
        raise


async def fetch_recent_retry_events(
    limit: int = 1000,
    connection: asyncpg.Connection = None
) -> list[dict[str, Any]]:
    """Fetch most recent retry events ordered oldest->newest up to *limit*."""
    query = "SELECT * FROM retry_events ORDER BY id DESC LIMIT $1"
    rows = await execute_query(query, connection=connection, params=(limit,), fetchall=True) or []
    rows.reverse()
    return rows

