"""QueryMethod descriptor for automatic method generation with authorization and transaction management."""

import logging
from typing import Any, Callable

import asyncpg

from deployment.app.db.connection import transaction

logger = logging.getLogger("plastinka.database")


class QueryMethod:
    """
    Descriptor that automatically applies authorization and transaction management
    to query functions.
    
    Usage:
        class DataAccessLayer:
            create_job = QueryMethod(
                queries.jobs.create_job,
                required_roles=[UserRoles.ADMIN, UserRoles.USER, UserRoles.SYSTEM],
                requires_transaction=True
            )
    """
    
    def __init__(
        self,
        query_func: Callable,
        required_roles: list[str],
        requires_transaction: bool = False,
    ):
        """
        Args:
            query_func: The async query function to wrap
            required_roles: List of roles that can access this method
            requires_transaction: Whether this method requires a transaction
        """
        self.query_func = query_func
        self.required_roles = required_roles
        self.requires_transaction = requires_transaction
        self.name = None  # Will be set by __set_name__
    
    def __set_name__(self, owner, name):
        """Called when the descriptor is assigned to a class attribute."""
        self.name = name
    
    def __get__(self, instance, owner):
        """Called when the attribute is accessed."""
        if instance is None:
            return self
        
        async def wrapper(*args, **kwargs):
            # Automatically apply authorization
            instance._authorize(self.required_roles)
            
            # Automatically manage transaction if required
            if self.requires_transaction:
                async with transaction(instance._pool) as conn:
                    # Pass connection to query function if it's not already in kwargs
                    if 'connection' not in kwargs:
                        kwargs['connection'] = conn
                    return await self.query_func(*args, **kwargs)
            else:
                # For read-only operations, acquire connection from pool
                async with instance._pool.acquire() as conn:
                    # Pass connection to query function if it's not already in kwargs
                    if 'connection' not in kwargs:
                        kwargs['connection'] = conn
                    return await self.query_func(*args, **kwargs)
        
        return wrapper

