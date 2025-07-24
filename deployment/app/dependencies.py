from typing import Generator, AsyncGenerator, Annotated

from fastapi import Depends

from deployment.app.db.database import get_db_connection
from deployment.app.db.data_access_layer import DataAccessLayer, UserContext, UserRoles
from deployment.app.services.auth import  get_unified_auth, get_admin_token_validated


def get_dal() -> Generator[DataAccessLayer, None, None]:
    """
    Dependency that provides a DataAccessLayer instance for API endpoints.
    Manages the database connection lifecycle.
    """
    db = None
    try:
        db = get_db_connection()
        yield DataAccessLayer(db)
    finally:
        if db:
            db.close()


async def get_dal_for_general_user(api_key_valid: Annotated[bool, Depends(get_unified_auth)]) -> AsyncGenerator[DataAccessLayer, None]:
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