from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends

from deployment.app.db.data_access_layer import DataAccessLayer, UserContext, UserRoles
from deployment.app.services.auth import get_unified_auth, get_admin_token_validated


async def get_dal() -> AsyncGenerator[DataAccessLayer, None]:
    """
    Dependency that provides a DataAccessLayer instance for API endpoints.
    Manages the database connection lifecycle.
    """
    dal = DataAccessLayer(user_context=UserContext(roles=[UserRoles.SYSTEM])) # Default to system for general DAL
    try:
        yield dal
    finally:
        # Assuming DAL manages its own connection closing or it's handled by a connection pool
        pass


async def get_dal_for_general_user(api_key_valid: Annotated[dict, Depends(get_unified_auth)]) -> AsyncGenerator[DataAccessLayer, None]:
    """
    FastAPI dependency for general users authenticated via X-API-Key.
    Provides DataAccessLayer with a 'user' role.
    """
    # Create user context based on authentication type
    if api_key_valid.get("type") == "admin_token":
        # Admin users get ADMIN role
        user_context = UserContext(roles=[UserRoles.ADMIN])
    else:
        # Regular API key users get USER role
        user_context = UserContext(roles=[UserRoles.USER])

    dal = DataAccessLayer(user_context=user_context)
    try:
        yield dal
    finally:
        pass


async def get_dal_for_admin_user(admin_token: Annotated[dict, Depends(get_admin_token_validated)]) -> AsyncGenerator[DataAccessLayer, None]:
    """
    FastAPI dependency for admin users authenticated via Bearer token.
    Provides DataAccessLayer with ADMIN role.
    """
    user_context = UserContext(roles=[UserRoles.ADMIN])
    dal = DataAccessLayer(user_context=user_context)
    try:
        yield dal
    finally:
        pass


# Optional: A dependency for operations that don't require specific user roles
# but still might benefit from DAL for consistent db interaction.
async def get_dal_system() -> AsyncGenerator[DataAccessLayer, None]:
    """
    FastAPI dependency that provides a DataAccessLayer instance with SYSTEM roles.
    To be used for internal or background tasks that operate with full system privileges.
    """
    user_context = UserContext(roles=[UserRoles.SYSTEM])
    dal = DataAccessLayer(user_context=user_context)
    try:
        yield dal
    finally:
        pass


def get_dal_system_sync(connection=None) -> DataAccessLayer:
    """
    Synchronous version of get_dal_system for use in non-FastAPI contexts.
    Provides a DataAccessLayer instance with SYSTEM roles.
    
    Args:
        connection: Optional database connection to use. If None, DAL creates its own.
    """
    user_context = UserContext(roles=[UserRoles.SYSTEM])
    return DataAccessLayer(user_context=user_context, connection=connection)
