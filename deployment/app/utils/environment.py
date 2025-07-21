"""
Environment validation utilities.
Centralized logic for checking required environment variables.
"""

import os
import json
import urllib.request
from typing import Dict, List, Any
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict


class ComponentHealth(BaseModel):
    """Component health status."""
    
    status: str
    details: Dict[str, Any] = {}
    
    model_config = ConfigDict(from_attributes=True)


class EnvironmentConfig:
    """Configuration for environment variable validation."""
    
    # Required environment variables
    REQUIRED_VARS = {
        "DATASPHERE_PROJECT_ID": "DataSphere project ID for ML model training and inference",
        "DATASPHERE_FOLDER_ID": "DataSphere folder ID (may differ from YANDEX_CLOUD_FOLDER_ID)",
        "API_ADMIN_API_KEY": "Admin API key for Bearer authentication (replaces API_API_KEY)",
        "API_X_API_KEY": "Public API key for X-API-Key header",
    }
    
    # DataSphere authentication alternatives (need at least one)
    DATASPHERE_AUTH_VARS = {
        "YC_OAUTH_TOKEN": "OAuth token for DataSphere API access",
    }


def get_missing_required_variables() -> List[str]:
    """Get list of missing required environment variables."""
    load_dotenv()  # Загружаем переменные из .env файла
    missing = []
    for var in EnvironmentConfig.REQUIRED_VARS:
        if var == "API_ADMIN_API_KEY":
            # Accept legacy variable as satisfying the requirement
            if not os.environ.get("API_ADMIN_API_KEY") and not os.environ.get("API_API_KEY"):
                missing.append(var)
        else:
            if not os.environ.get(var):
                missing.append(var)
    return missing


def check_datasphere_authentication() -> tuple[bool, List[str]]:
    """
    Check DataSphere authentication configuration.
    Returns (has_auth, missing_description).
    """
    has_oauth = bool(os.environ.get("YC_OAUTH_TOKEN"))
    has_yc_profile = bool(os.environ.get("DATASPHERE_YC_PROFILE"))
    
    if has_oauth or has_yc_profile:
        return True, []
    
    return False, ["YC_OAUTH_TOKEN or DATASPHERE_YC_PROFILE (DataSphere Authentication)"]


def check_yc_profile_health() -> ComponentHealth:
    """
    Check if the Yandex Cloud CLI profile is valid and accessible.
    """
    profile_name = os.environ.get("DATASPHERE_YC_PROFILE", "datasphere-prod")
    
    try:
        # Check if yc CLI is available
        import subprocess
        result = subprocess.run(
            ["yc", "config", "profile", "get", profile_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return ComponentHealth(status="healthy", details={"profile": profile_name})
        else:
            return ComponentHealth(
                status="unhealthy", 
                details={
                    "error": f"Profile '{profile_name}' not found or invalid",
                    "stderr": result.stderr
                }
            )
    except subprocess.TimeoutExpired:
        return ComponentHealth(
            status="unhealthy",
            details={"error": f"Timeout checking profile '{profile_name}'"}
        )
    except FileNotFoundError:
        return ComponentHealth(
            status="unhealthy",
            details={"error": "yc CLI not found. Please install Yandex Cloud CLI"}
        )
    except Exception as e:
        return ComponentHealth(
            status="unhealthy",
            details={"error": f"Error checking YC profile: {str(e)}"}
        )


def check_yc_token_health() -> ComponentHealth:
    """
    Check if the Yandex Cloud token is valid by making an API call.
    """
    token = os.environ.get("YC_OAUTH_TOKEN")
    if not token:
        return ComponentHealth(status="unhealthy", details={"error": "YC_OAUTH_TOKEN is not set."})

    url = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
    data = {"yandexPassportOauthToken": token}
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                return ComponentHealth(status="healthy")
            else:
                return ComponentHealth(
                    status="unhealthy",
                    details={
                        "error": f"Yandex IAM API returned status {response.status}",
                        "response": response.read().decode("utf-8"),
                    },
                )
    except urllib.error.HTTPError as e:
        return ComponentHealth(
            status="unhealthy",
            details={
                "error": f"Yandex IAM API request failed with HTTP status {e.code}",
                "response": e.read().decode("utf-8"),
            },
        )
    except Exception as e:
        return ComponentHealth(
            status="unhealthy",
            details={"error": f"An unexpected error occurred while checking YC token health: {str(e)}"},
        )


def get_environment_status() -> ComponentHealth:
    """
    Get comprehensive environment status for health checks.
    Returns ComponentHealth object with status and details.
    """
    missing = get_missing_required_variables()
    has_auth, auth_missing = check_datasphere_authentication()
    
    all_missing = missing + auth_missing
    
    if all_missing:
        # Create detailed missing variables list with descriptions
        missing_with_desc = []
        for var in missing:
            desc = EnvironmentConfig.REQUIRED_VARS.get(var, "")
            missing_with_desc.append(f"{var} ({desc})" if desc else var)
        
        missing_with_desc.extend(auth_missing)
        
        details = {
            "missing_variables": missing_with_desc,
            "message": "Some required environment variables are missing",
        }
        return ComponentHealth(status="degraded", details=details)
    
    # Check authentication method and validate accordingly
    has_oauth = bool(os.environ.get("YC_OAUTH_TOKEN"))
    has_yc_profile = bool(os.environ.get("DATASPHERE_YC_PROFILE"))
    
    if has_oauth:
        # Check YC token health
        token_health = check_yc_token_health()
        if token_health.status != "healthy":
            return token_health
    elif has_yc_profile:
        # Check YC profile health
        profile_health = check_yc_profile_health()
        if profile_health.status != "healthy":
            return profile_health
    else:
        # This shouldn't happen due to check_datasphere_authentication above
        return ComponentHealth(
            status="unhealthy",
            details={"error": "No authentication method configured"}
        )

    return ComponentHealth(status="healthy")


def get_missing_variables_simple() -> List[str]:
    """
    Get simple list of missing variables for script usage.
    Combines required vars and auth requirements.
    """
    missing = get_missing_required_variables()
    has_auth, _ = check_datasphere_authentication()
    
    if not has_auth:
        # Add auth option to the missing list for script display
        missing.append("YC_OAUTH_TOKEN or DATASPHERE_YC_PROFILE")
    
    return missing


def get_all_variable_descriptions() -> Dict[str, str]:
    """Get all environment variables with their descriptions for template generation."""
    all_vars = {}
    all_vars.update(EnvironmentConfig.REQUIRED_VARS)
    all_vars.update(EnvironmentConfig.DATASPHERE_AUTH_VARS)
    return all_vars
