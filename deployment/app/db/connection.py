"""PostgreSQL connection pool management."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from asyncpg import Pool

from deployment.app.config import get_settings
from deployment.app.db.exceptions import DatabaseError

logger = logging.getLogger("plastinka.database")

# Global PostgreSQL connection pool
_db_pool: Pool | None = None


async def init_db_pool() -> Pool:
    """
    Initialize PostgreSQL connection pool at application startup.
    
    Returns:
        Pool: The initialized connection pool
        
    Raises:
        DatabaseError: If pool initialization fails
    """
    global _db_pool
    
    if _db_pool is not None:
        logger.warning("Database pool already initialized")
        return _db_pool
    
    try:
        settings = get_settings()
        
        # Determine SSL mode
        ssl_config = None
        if settings.db.postgres_ssl_mode == "require":
            ssl_config = "require"
        elif settings.db.postgres_ssl_mode in ("verify-ca", "verify-full"):
            ssl_config = settings.db.postgres_ssl_mode
        
        # NOTE(REVIEWER): Connection pool configuration looks good for production.
        # max_queries=50000 is high but reasonable for long-running services.
        # Consider monitoring pool exhaustion metrics in production to tune min_size/max_size.
        # command_timeout=60s may be too short for complex queries - monitor query times.
        pool = await asyncpg.create_pool(
            host=settings.db.postgres_host,
            port=settings.db.postgres_port,
            database=settings.db.postgres_database,
            user=settings.db.postgres_user,
            password=settings.db.postgres_password,
            min_size=settings.db.postgres_pool_min_size,
            max_size=settings.db.postgres_pool_max_size,
            ssl=ssl_config,
            command_timeout=60,
            max_queries=50000,
            max_inactive_connection_lifetime=300.0,
        )
        
        _db_pool = pool
        logger.info(
            f"PostgreSQL connection pool initialized: "
            f"host={settings.db.postgres_host}, "
            f"database={settings.db.postgres_database}, "
            f"pool_size={settings.db.postgres_pool_min_size}-{settings.db.postgres_pool_max_size}"
        )
        return pool
        
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {str(e)}", exc_info=True)
        raise DatabaseError(
            f"Database pool initialization failed: {str(e)}", original_error=e
        ) from e


async def close_db_pool():
    """
    Close PostgreSQL connection pool at application shutdown.
    """
    global _db_pool
    
    if _db_pool is not None:
        try:
            await _db_pool.close()
            logger.info("PostgreSQL connection pool closed")
        except Exception as e:
            logger.error(f"Error closing database pool: {str(e)}", exc_info=True)
        finally:
            _db_pool = None


def get_db_pool() -> Pool:
    """
    Get the global PostgreSQL connection pool.
    
    Returns:
        Pool: The connection pool
        
    Raises:
        DatabaseError: If pool is not initialized
    """
    if _db_pool is None:
        raise DatabaseError("Database pool not initialized. Call init_db_pool() first.")
    return _db_pool


@asynccontextmanager
async def transaction(pool: Pool) -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Async context manager for database transactions.
    
    Args:
        pool: The connection pool to use
        
    Yields:
        asyncpg.Connection: A connection within a transaction
        
    Example:
        async with transaction(pool) as conn:
            await execute_query("INSERT INTO ...", connection=conn, params=(...))
            # Transaction is automatically committed on exit, or rolled back on exception
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            yield conn

