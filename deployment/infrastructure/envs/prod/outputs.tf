# =============================================================================
# OUTPUTS FOR PROD ENVIRONMENT
# =============================================================================
# This file contains all outputs for the production environment.
# These outputs are used by DataSphere and other external systems.

# =============================================================================
# DATASPHERE OUTPUTS
# =============================================================================

output "datasphere_project_id" {
  description = "DataSphere project ID"
  value       = local.project_id
}

output "datasphere_cloud_id" {
  description = "DataSphere Cloud ID"
  value       = var.yc_cloud_id
}

output "service_account_id" {
  description = "ID of the DataSphere service account"
  value       = local.service_account_id
}

output "service_account_name" {
  description = "Name of the DataSphere service account"
  value = var.existing_service_account_id != null ? data.yandex_iam_service_account.existing[0].name : yandex_iam_service_account.datasphere[0].name
}

output "static_access_key_id" {
  description = "Static access key ID for the DataSphere service account"
  value       = local.static_access_key_id
  sensitive   = true
}

output "static_secret_key" {
  description = "Static secret key for the DataSphere service account"
  value       = local.static_secret_key
  sensitive   = true
}

output "community_id" {
  description = "DataSphere Community ID"
  value       = local.community_id
}

# =============================================================================
# CONVENIENCE OUTPUTS
# =============================================================================

output "datasphere_summary" {
  description = "Summary of DataSphere configuration"
  value = {
    PROJECT_ID              = local.project_id
    SERVICE_ACCOUNT_ID      = local.service_account_id
    COMMUNITY_ID           = local.community_id
    CLOUD_ID               = var.yc_cloud_id
    FOLDER_ID              = var.yc_folder_id
    ORGANIZATION_ID        = var.yc_organization_id
    
    # Индикаторы режима работы
    USING_EXISTING_SA      = var.existing_service_account_id != null
    USING_EXISTING_COMMUNITY = var.existing_community_id != null
    USING_EXISTING_PROJECT = var.existing_project_id != null
    IS_IMPORT_MODE         = (var.existing_service_account_id != null) || (var.existing_community_id != null) || (var.existing_project_id != null)
  }
}

# =============================================================================
# IMPORT INFORMATION
# =============================================================================

output "import_status" {
  description = "Status of imported vs created resources"
  value = {
    service_account = var.existing_service_account_id != null ? "existing" : "created"
    community      = var.existing_community_id != null ? "existing" : "created"
    project        = var.existing_project_id != null ? "existing" : "created"
    static_key     = var.existing_static_key_id != null ? "existing" : (var.existing_service_account_id != null ? "none" : "created")
  }
}

 