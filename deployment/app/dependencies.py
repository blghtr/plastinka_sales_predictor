from typing import AsyncGenerator, Annotated

from fastapi import Depends

from deployment.app.db.database import get_db_connection
from deployment.app.db.data_access_layer import DataAccessLayer, UserContext, UserRoles
from deployment.app.services.auth import get_admin_user, get_current_api_key_validated


async def get_dal(admin_user: Annotated[dict, Depends(get_admin_user)]) -> AsyncGenerator[DataAccessLayer, None]:
    """
    FastAPI dependency that provides a DataAccessLayer instance configured with the authenticated user's context.

    This dependency assumes that `get_admin_user` (or a similar dependency) has already validated
    the user and provided their roles/permissions.

    For simplicity, this example uses `get_admin_user` for admin role. A more complex application
    might have a generic `get_current_user` that returns a user object with roles.
    """
    # Determine roles based on the authentication dependency used.
    # For `get_admin_user`, we know it provides an 'admin' role.
    roles = admin_user.get("roles", [])
    user_context = UserContext(roles=roles)
    
    # For this application, we can use a single DB connection for the request lifetime
    # managed by db_transaction if it's called internally by DAL, or passed if needed.
    # For simplicity, DAL methods will get their own connections for now, consistent with existing `execute_query`.
    # If a shared connection per request is desired, it would be passed to DAL init.

    dal = DataAccessLayer(user_context=user_context)
    try:
        yield dal
    finally:
        # No explicit close needed here as DAL methods manage their own connections.
        pass


async def get_dal_for_general_user(api_key_valid: Annotated[bool, Depends(get_current_api_key_validated)]) -> AsyncGenerator[DataAccessLayer, None]:
    """
    FastAPI dependency for general users authenticated via X-API-Key.
    Provides DataAccessLayer with a 'user' role.
    """
    if api_key_valid:
        # Assuming general users authenticated via X-API-Key get 'viewer' permissions
        # In a real app, this would be more granular based on user type/scope.
        user_context = UserContext(roles=[UserRoles.USER])
    else:
        # This path should ideally not be hit if auto_error=True for APIKeyHeader
        # or if get_current_api_key_validated raises HTTPException.
        # Providing a minimal context for safety, though it implies unauthenticated access.
        user_context = UserContext(roles=[]) # No roles for unauthenticated

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