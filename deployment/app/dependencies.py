from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends

from deployment.app.db.connection import get_db_pool
from deployment.app.db.data_access_layer import DataAccessLayer, UserContext, UserRoles
from deployment.app.services.auth import get_unified_auth, get_admin_token_validated


async def get_dal() -> AsyncGenerator[DataAccessLayer, None]:
    """
    Dependency that provides a DataAccessLayer instance for API endpoints.
    Uses the global PostgreSQL connection pool.
    """
    pool = get_db_pool()
    dal = DataAccessLayer(user_context=UserContext(roles=[UserRoles.SYSTEM]), pool=pool)
    try:
        yield dal
    finally:
        # Pool is managed globally, no cleanup needed here
        pass


async def get_dal_for_general_user(api_key_valid: Annotated[dict, Depends(get_unified_auth)]) -> AsyncGenerator[DataAccessLayer, None]:
    """
    FastAPI dependency for general users authenticated via X-API-Key.
    Provides DataAccessLayer with a 'user' role.
    """
    pool = get_db_pool()
    # Create user context based on authentication type
    if api_key_valid.get("type") == "admin_token":
        # Admin users get ADMIN role
        user_context = UserContext(roles=[UserRoles.ADMIN])
    else:
        # Regular API key users get USER role
        user_context = UserContext(roles=[UserRoles.USER])

    dal = DataAccessLayer(user_context=user_context, pool=pool)
    try:
        yield dal
    finally:
        pass


async def get_dal_for_admin_user(admin_token: Annotated[dict, Depends(get_admin_token_validated)]) -> AsyncGenerator[DataAccessLayer, None]:
    """
    FastAPI dependency for admin users authenticated via Bearer token.
    Provides DataAccessLayer with ADMIN role.
    """
    pool = get_db_pool()
    user_context = UserContext(roles=[UserRoles.ADMIN])
    dal = DataAccessLayer(user_context=user_context, pool=pool)
    try:
        yield dal
    finally:
        pass


async def get_dal_system() -> AsyncGenerator[DataAccessLayer, None]:
    """
    FastAPI dependency that provides a DataAccessLayer instance with SYSTEM roles.
    To be used for internal or background tasks that operate with full system privileges.
    """
    pool = get_db_pool()
    user_context = UserContext(roles=[UserRoles.SYSTEM])
    dal = DataAccessLayer(user_context=user_context, pool=pool)
    try:
        yield dal
    finally:
        pass
