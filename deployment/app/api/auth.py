"""
Authentication related API endpoints.
"""

import os
import subprocess
from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel
import logging
from typing import Any

from ..services.auth import get_admin_user
from ..models.api_models import YandexCloudToken
from ..config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/yc-token", status_code=status.HTTP_204_NO_CONTENT,
             summary="Set the Yandex Cloud OAuth token.")
async def set_yc_token(
    payload: YandexCloudToken = Body(..., description="A JSON object containing the `token` string."),
    admin_user: dict[str, Any] = Depends(get_admin_user),
):
    """
    Configures the Yandex Cloud CLI profile with the provided OAuth token. This token is
    used for authenticating with the DataSphere API for running jobs.
    Requires admin authentication.
    """
    if not payload.token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token cannot be empty.",
        )
    
    try:
        # Get the profile name from settings
        settings = get_settings()
        profile_name = settings.datasphere.yc_profile or "datasphere-prod"
        
        # Configure yc CLI with the token for the specific profile
        subprocess.run(
            ["yc", "config", "set", "token", payload.token, "--profile", profile_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Also set the environment variable for immediate use by the application
        os.environ["YC_OAUTH_TOKEN"] = payload.token
        
        logger.info(f"Yandex Cloud OAuth token has been updated for profile '{profile_name}' via yc CLI and environment variable.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to configure Yandex Cloud CLI profile '{profile_name}': {e.stderr}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure Yandex Cloud CLI profile '{profile_name}': {e.stderr}",
        )
    except Exception as e:
        logger.error(f"Failed to set Yandex Cloud token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set Yandex Cloud token.",
        )