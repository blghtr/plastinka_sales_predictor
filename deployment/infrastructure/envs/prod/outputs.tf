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
  value       = module.project.project_id
}

output "datasphere_cloud_id" {
  description = "DataSphere Cloud ID"
  value       = var.yc_cloud_id
}

output "service_account_id" {
  description = "ID of the DataSphere service account"
  value       = module.datasphere_service_account.service_account_id
}

output "service_account_name" {
  description = "Name of the DataSphere service account"
  value       = module.datasphere_service_account.service_account_name
}

output "static_access_key_id" {
  description = "Static access key ID for the DataSphere service account"
  value       = module.datasphere_service_account.static_access_key_id
  sensitive   = true
}

output "static_secret_key" {
  description = "Static secret key for the DataSphere service account"
  value       = module.datasphere_service_account.static_secret_key
  sensitive   = true
}

# =============================================================================
# CONVENIENCE OUTPUTS
# =============================================================================

output "datasphere_summary" {
  description = "Summary of DataSphere configuration"
  value = {
    PROJECT_ID              = module.project.project_id
    SERVICE_ACCOUNT_ID      = module.datasphere_service_account.service_account_id
    CLOUD_ID               = var.yc_cloud_id
    FOLDER_ID              = var.yc_folder_id
    ORGANIZATION_ID        = var.yc_organization_id
  }
}

 