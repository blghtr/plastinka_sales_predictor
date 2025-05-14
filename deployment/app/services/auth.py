"""
Authentication service for API security.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any

from ..config import settings

# Define security scheme for bearer token
security = HTTPBearer()

async def get_admin_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Validate admin token and return user information.
    
    Args:
        credentials: Bearer token credentials from the HTTP Authorization header
    
    Returns:
        Dict containing admin user information
    
    Raises:
        HTTPException: If the token is missing or invalid
    """
    admin_token = settings.api.api_key
    
    if not admin_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin API key not configured on server"
        )
    
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Return some basic admin user information
    return {
        "user_type": "admin",
        "roles": ["admin"],
        "permissions": ["manage_data_retention", "manage_models", "manage_jobs"]
    } 