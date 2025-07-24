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
    COMMUNITY_ID           = local.community_id
    CLOUD_ID               = var.yc_cloud_id
    FOLDER_ID              = var.yc_folder_id
    ORGANIZATION_ID        = var.yc_organization_id
    
    # Индикаторы режима работы
    USING_EXISTING_COMMUNITY = var.existing_community_id != null
    USING_EXISTING_PROJECT = var.existing_project_id != null
  }
}

# =============================================================================
# IMPORT INFORMATION
# =============================================================================

output "import_status" {
  description = "Status of imported vs created resources"
  value = {
    community      = var.existing_community_id != null ? "existing" : "created"
    project        = var.existing_project_id != null ? "existing" : "created"
  }
}

 