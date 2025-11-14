"""
Authentication related API endpoints.
"""

import logging
import os

from fastapi import APIRouter, Body, Depends, HTTPException, status

from ..models.api_models import YandexCloudToken
from ..services.auth import get_admin_token_validated

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/yc-token", status_code=status.HTTP_204_NO_CONTENT,
             summary="Set the Yandex Cloud OAuth token.")
async def set_yc_token(
    payload: YandexCloudToken = Body(..., description="A JSON object containing the `token` string."),
    _admin_user = Depends(get_admin_token_validated),  # PROTECTED by admin dependency
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
        # Store token only in environment; do not pass via CLI args or profiles
        os.environ["YC_OAUTH_TOKEN"] = payload.token
        logger.info("Yandex Cloud OAuth token updated via environment variable.")
    except Exception as e:
        logger.error(f"Failed to set Yandex Cloud token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set Yandex Cloud token.",
        ) from e
