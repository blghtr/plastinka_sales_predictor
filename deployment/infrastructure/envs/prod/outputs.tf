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
    postgres_cluster = var.existing_postgres_cluster_id != null ? "existing" : "created"
  }
}

# =============================================================================
# POSTGRESQL OUTPUTS
# =============================================================================

output "postgres_cluster_id" {
  description = "PostgreSQL cluster ID"
  value       = var.existing_postgres_cluster_id != null ? var.existing_postgres_cluster_id : (length(yandex_mdb_postgresql_cluster.ml_postgres) > 0 ? yandex_mdb_postgresql_cluster.ml_postgres[0].id : null)
}

output "postgres_host" {
  description = "PostgreSQL cluster FQDN"
  value       = var.existing_postgres_cluster_id != null ? null : (length(yandex_mdb_postgresql_cluster.ml_postgres) > 0 ? yandex_mdb_postgresql_cluster.ml_postgres[0].host[0].fqdn : null)
}

output "postgres_port" {
  description = "PostgreSQL port"
  value       = 6432  # Default Yandex Managed PostgreSQL port
}

output "postgres_database" {
  description = "PostgreSQL database name"
  value       = var.postgres_database_name
}

output "postgres_user" {
  description = "PostgreSQL user name"
  value       = var.postgres_user_name
}

output "postgres_password" {
  description = "PostgreSQL user password (sensitive)"
  value       = var.existing_postgres_cluster_id != null ? null : (length(random_password.postgres_password) > 0 ? random_password.postgres_password[0].result : null)
  sensitive   = true
}

output "postgres_connection_string" {
  description = "PostgreSQL connection string (without password)"
  value       = var.existing_postgres_cluster_id != null ? null : (length(yandex_mdb_postgresql_cluster.ml_postgres) > 0 ? "postgresql://${var.postgres_user_name}@${yandex_mdb_postgresql_cluster.ml_postgres[0].host[0].fqdn}:6432/${var.postgres_database_name}" : null)
}

 