"""
Authentication service for API security.
"""

import logging
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from ..config import AppSettings, get_settings

logger = logging.getLogger(__name__)

# Define security scheme for bearer token
bearer_scheme = HTTPBearer()

# Define security scheme for X-API-Key header
api_key_header_scheme = APIKeyHeader(
    name="X-API-Key", auto_error=False
)  # auto_error=False to handle empty config case


async def get_admin_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
    settings: AppSettings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Validate admin token and return user information.

    Args:
        credentials: Bearer token credentials from the HTTP Authorization header

    Returns:
        Dict containing admin user information

    Raises:
        HTTPException: If the token is missing or invalid
    """

    if not settings.api.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin API key not configured on server",
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != settings.api.admin_api_key:
        logger.warning(
            f"[get_admin_user] Invalid Bearer token provided. Expected: {settings.api.admin_api_key}, Got: {credentials.credentials}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Return some basic admin user information
    return {
        "user_type": "admin",
        "roles": ["admin"],
        "permissions": ["manage_data_retention", "manage_models", "manage_jobs"],
    }


async def get_current_api_key_validated(
    api_key: str = Security(api_key_header_scheme),
    settings: AppSettings = Depends(get_settings),
) -> bool:
    """
    Validate X-API-Key header.

    Args:
        api_key: The API key from the X-API-Key header.

    Returns:
        True if the API key is valid.

    Raises:
        HTTPException: If the API key is missing, not configured, or invalid.
    """

    if not settings.api.x_api_key:
        # Log this situation as it's a server misconfiguration if endpoints are protected
        # but no key is set. For now, we'll raise an error.
        # Consider logging a warning here in a real scenario.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="X-API-Key authentication is not configured on the server.",
        )

    if (
        not api_key
    ):  # This case is handled by auto_error=True in APIKeyHeader if we set it
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: X-API-Key header is missing.",
        )

    if api_key != settings.api.x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-API-Key."
        )
    return True


async def get_docs_auth(
    bearer_token: HTTPAuthorizationCredentials | None = Security(bearer_scheme, auto_error=False),
    api_key: str | None = Security(api_key_header_scheme, auto_error=False),
    settings: AppSettings = Depends(get_settings),
) -> bool:
    """
    Authenticate access to API documentation.
    Supports both Bearer token (for admins) and X-API-Key (for general use).
    """
    if not settings.api.docs_security_enabled:
        return True  # Security is disabled

    # Check for Bearer token first
    if bearer_token:
        try:
            # Reuse admin validation logic but handle exceptions locally
            if bearer_token.scheme.lower() == "bearer" and bearer_token.credentials == settings.api.admin_api_key:
                return True
        except HTTPException:
            # This will be caught and re-raised below if no other key is valid
            pass

    # Check for X-API-Key
    if api_key:
        try:
            # Reuse API key validation logic
            if await get_current_api_key_validated(api_key, settings):
                return True
        except HTTPException:
            # This will be caught and re-raised below
            pass

    # If neither key is valid, raise an error
    logger.warning("Unauthorized attempt to access API documentation.")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="You are not authorized to view the API documentation.",
        headers={"WWW-Authenticate": "Bearer"},
    )
