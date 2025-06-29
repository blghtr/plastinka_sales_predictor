provider "yandex" {
  token      = var.yc_token
  cloud_id   = var.yc_cloud_id
  folder_id  = var.yc_folder_id
}

# Service Account for DataSphere operations
module "datasphere_service_account" {
  source = "../../modules/service_account"

  name               = "datasphere-sa-prod"
  description        = "Service account for DataSphere operations in production"
  folder_id          = var.yc_folder_id
  cloud_id           = var.yc_cloud_id
  create_static_key  = true

  # DataSphere specific roles 
  folder_roles = [
    "datasphere.user",
    "datasphere.communities.developer",
    "storage.admin",    # For accessing Object Storage
    "compute.admin",    # For compute resources
    "vpc.user"          # For network access
  ]

  # Cloud-level roles if needed
  cloud_roles = []
}

module "community" {
  source = "../../modules/datasphere_community"

  name            = "prod-ds-community"
  organization_id = var.yc_organization_id
  labels = {
    env     = "prod"
    project = "plastinka-sales-predictor"
  }
}

module "project" {
  source = "../../modules/datasphere_project"

  name                    = "prod-ds-project"
  description             = "Plastinka Sales Predictor DataSphere Project (prod)"
  community_id            = module.community.community_id
  service_account_id      = module.datasphere_service_account.service_account_id
  max_units_per_hour      = 20
  max_units_per_execution = 200
  labels = {
    env     = "prod"
    project = "plastinka-sales-predictor"
  }
}
