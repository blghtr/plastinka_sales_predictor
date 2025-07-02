"""
Environment validation utilities.
Centralized logic for checking required environment variables.
"""

import os
from typing import Dict, List, Any
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
        "DATASPHERE_OAUTH_TOKEN": "OAuth token for DataSphere API access",
        "DATASPHERE_YC_PROFILE": "YC CLI profile name for service account authentication",
    }


def get_missing_required_variables() -> List[str]:
    """Get list of missing required environment variables."""
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
    has_oauth = bool(os.environ.get("DATASPHERE_OAUTH_TOKEN"))
    has_profile = bool(os.environ.get("DATASPHERE_YC_PROFILE"))
    
    if has_oauth or has_profile:
        return True, []
    
    return False, ["DATASPHERE_OAUTH_TOKEN or DATASPHERE_YC_PROFILE (DataSphere Authentication)"]


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
    
    return ComponentHealth(status="healthy")


def get_missing_variables_simple() -> List[str]:
    """
    Get simple list of missing variables for script usage.
    Combines required vars and auth requirements.
    """
    missing = get_missing_required_variables()
    has_auth, _ = check_datasphere_authentication()
    
    if not has_auth:
        # Add both auth options to the missing list for script display
        missing.extend(["DATASPHERE_OAUTH_TOKEN", "DATASPHERE_YC_PROFILE"])
    
    return missing


def get_all_variable_descriptions() -> Dict[str, str]:
    """Get all environment variables with their descriptions for template generation."""
    all_vars = {}
    all_vars.update(EnvironmentConfig.REQUIRED_VARS)
    all_vars.update(EnvironmentConfig.DATASPHERE_AUTH_VARS)
    return all_vars 