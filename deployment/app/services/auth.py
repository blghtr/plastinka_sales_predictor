"""
Authentication service for API security.

This module provides secure authentication using bcrypt hashing for both API keys
and admin tokens. It supports Authorization header authentication with proper
Swagger UI integration.
"""

import logging
import secrets
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)
from passlib.context import CryptContext

from ..config import AppSettings, get_settings
from ..utils.error_handling import ErrorDetail

logger = logging.getLogger(__name__)

# Initialize CryptContext for secure password/key hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes for Swagger UI integration
api_key_header = APIKeyHeader(
    name="X-API-Key",
    description="Enter your API key (without 'Bearer ' prefix)",
    auto_error=False  # Don't auto-generate 401 errors
)

bearer_scheme = HTTPBearer(
    description="Enter your admin Bearer token",
    auto_error=False  # Don't auto-generate 401 errors
)

docs_security = HTTPBasic()


def get_docs_user(credentials: HTTPBasicCredentials = Depends(docs_security), settings: AppSettings = Depends(get_settings)):
    """Dependency to protect documentation endpoints with Basic Auth."""
    # Use secrets.compare_digest to prevent timing attacks
    correct_username = secrets.compare_digest(credentials.username.encode("utf8"), b"admin")
    # IMPORTANT: This compares against the RAW token, not the hash.
    # The admin_api_key_hash setting should hold the raw token for this to work.
    # Ensure the config `admin_api_key_hash` holds the raw token value.
    correct_password = pwd_context.verify(credentials.password, settings.api.admin_api_key_hash)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


async def get_current_api_key_validated(
    api_key: str | None = Depends(api_key_header),
    settings: AppSettings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Validate API key from Authorization header using bcrypt verification.

    Args:
        api_key: The API key from the Authorization header
        settings: Application settings containing the hashed API key

    Returns:
        Dict containing authentication type and value

    Raises:
        HTTPException: If authentication fails or is not configured
    """
    # Check if API key authentication is configured
    if not settings.api.x_api_key_hash:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                message="API key authentication is not configured on the server.",
                code="internal_error"
            ).to_dict(),
        )

    # Check if API key is provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorDetail(
                message="API key is required in Authorization header.",
                code="authentication_error"
            ).to_dict(),
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Securely verify the API key against the hash
    try:
        is_valid = pwd_context.verify(api_key, settings.api.x_api_key_hash)
    except Exception as e:
        logger.warning(f"Error verifying API key: {e}")
        is_valid = False

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorDetail(
                message="Invalid API key.",
                code="authentication_error"
            ).to_dict(),
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return {"type": "api_key", "value": api_key}


async def get_admin_token_validated(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: AppSettings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Validate admin Bearer token using bcrypt verification.

    Args:
        credentials: The Bearer token credentials
        settings: Application settings containing the hashed admin token

    Returns:
        Dict containing authentication type and value

    Raises:
        HTTPException: If authentication fails or is not configured
    """
    # Check if admin token authentication is configured
    if not settings.api.admin_api_key_hash:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                message="Admin token authentication is not configured on the server.",
                code="internal_error"
            ).to_dict(),
        )

    # Check if Bearer token is provided
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorDetail(
                message="Bearer token is required for admin access.",
                code="authentication_error"
            ).to_dict(),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Securely verify the admin token against the hash
    try:
        is_valid = pwd_context.verify(credentials.credentials, settings.api.admin_api_key_hash)
    except Exception as e:
        logger.warning(f"Error verifying admin token: {e}")
        is_valid = False

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorDetail(
                message="Invalid admin token.",
                code="authentication_error"
            ).to_dict(),
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"type": "admin_token", "value": credentials.credentials}


async def get_unified_auth(
    api_key: str | None = Depends(api_key_header),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: AppSettings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Unified authentication that accepts both Bearer tokens (admin) and API keys.
    It prioritizes the Bearer token if both are provided.
    """
    # Try admin token first
    if credentials and settings.api.admin_api_key_hash:
        try:
            if pwd_context.verify(credentials.credentials, settings.api.admin_api_key_hash):
                return {"type": "admin_token", "value": credentials.credentials}
        except Exception:
            # Fall through to check API key or raise error
            pass

    # Then try API key
    if api_key and settings.api.x_api_key_hash:
        try:
            if pwd_context.verify(api_key, settings.api.x_api_key_hash):
                return {"type": "api_key", "value": api_key}
        except Exception:
            # Fall through to raise error
            pass

    # If neither is present or valid, raise an error
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=ErrorDetail(
            message="Invalid or missing credentials. Provide a valid Bearer token or X-API-Key.",
            code="authentication_error"
        ).to_dict(),
        headers={"WWW-Authenticate": "Bearer, ApiKey"},
    )


# Convenience functions for common authentication patterns
async def require_api_key(
    auth_result: dict[str, Any] = Depends(get_unified_auth)
) -> str:
    """
    Dependency that requires API key authentication.

    Returns:
        The API key value

    Raises:
        HTTPException: If API key authentication fails
    """
    return auth_result["value"]


async def require_admin_token(
    auth_result: dict[str, Any] = Depends(get_admin_token_validated)
) -> str:
    """
    Dependency that requires admin token authentication.

    Returns:
        The admin token value

    Raises:
        HTTPException: If admin token authentication fails
    """
    return auth_result["value"]


async def require_any_auth(
    auth_result: dict[str, Any] = Depends(get_unified_auth)
) -> dict[str, Any]:
    """
    Dependency that accepts either API key or admin token authentication.

    Returns:
        Dict containing authentication type and value

    Raises:
        HTTPException: If authentication fails
    """
    return auth_result


