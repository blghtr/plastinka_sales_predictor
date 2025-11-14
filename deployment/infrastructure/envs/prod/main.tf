provider "yandex" {
  token      = var.yc_token
  cloud_id   = var.yc_cloud_id
  folder_id  = var.yc_folder_id
}

# ============================================================================
# DATA SOURCES ДЛЯ СУЩЕСТВУЮЩИХ РЕСУРСОВ
# ============================================================================

# Получаем данные о существующем DataSphere Community, если ID предоставлен
data "yandex_datasphere_community" "existing" {
  count = var.existing_community_id != null ? 1 : 0
  id    = var.existing_community_id
}

# Получаем данные о существующем DataSphere Project, если ID предоставлен
data "yandex_datasphere_project" "existing" {
  count = var.existing_project_id != null ? 1 : 0
  id    = var.existing_project_id
}

# ============================================================================
# ПРЯМЫЕ РЕСУРСЫ (БЕЗ МОДУЛЕЙ) ДЛЯ НОВОГО СОЗДАНИЯ
# ============================================================================

# DataSphere Community - создается только если не предоставлен existing_community_id
resource "yandex_datasphere_community" "datasphere" {
  count           = var.existing_community_id == null ? 1 : 0
  name            = "prod-ds-community"
  organization_id = var.yc_organization_id
  
  labels = {
    env     = "prod"
    project = "plastinka-sales-predictor"
  }
  
  lifecycle {
    prevent_destroy = true
    # Игнорировать изменения в метках
    ignore_changes = [
      labels
    ]
  }
}

# DataSphere Project - создается только если не предоставлен existing_project_id
resource "yandex_datasphere_project" "datasphere" {
  count        = var.existing_project_id == null ? 1 : 0
  name         = "prod-ds-project"
  description  = "Plastinka Sales Predictor DataSphere Project (prod)"
  community_id = local.community_id

  settings = {
    # service_account_id is no longer used
  }

  limits = {
    max_units_per_hour      = 20
    max_units_per_execution = 200
  }

  labels = {
    env     = "prod"
    project = "plastinka-sales-predictor"
  }

  lifecycle {
    prevent_destroy = true
    # Игнорировать изменения в описании и метках
    ignore_changes = [
      description,
      labels
    ]
  }
}

# ============================================================================
# ЛОКАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ УНИФИКАЦИИ
# ============================================================================

locals {
  community_id = var.existing_community_id != null ? var.existing_community_id : (length(yandex_datasphere_community.datasphere) > 0 ? yandex_datasphere_community.datasphere[0].id : null)
  project_id = var.existing_project_id != null ? var.existing_project_id : (length(yandex_datasphere_project.datasphere) > 0 ? yandex_datasphere_project.datasphere[0].id : null)
  project_root_env_path = "${path.root}/../../../../.env"
  deployment_scripts_path = "${path.root}/../../../scripts"
}

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ РЕСУРСЫ
# ============================================================================

resource "null_resource" "create_env_file" {
  triggers = {
    # Запускается только если файл не существует
    file_exists = fileexists("${local.project_root_env_path}") ? "true" : "false"
    project_id = local.project_id
    folder_id = var.yc_folder_id
  }

  provisioner "local-exec" {
    interpreter = ["bash", "-c"]
    command = <<-EOT
      echo "# DataSphere Configuration (REQUIRED)" > "${local.project_root_env_path}"
      echo "DATASPHERE_PROJECT_ID=${local.project_id}" >> "${local.project_root_env_path}"
      echo "DATASPHERE_FOLDER_ID=${var.yc_folder_id}" >> "${local.project_root_env_path}"
      # DATASPHERE_YC_PROFILE removed – using YC_OAUTH_TOKEN only
      echo "" >> "${local.project_root_env_path}"
      echo "# API Security (REQUIRED)" >> "${local.project_root_env_path}"
      echo "# These will be automatically generated and populated by the Python script." >> "${local.project_root_env_path}"
      echo "# Ensure Python 3.x is installed in the environment where 'terraform apply' is run." >> "${local.project_root_env_path}"
      echo "# API_ADMIN_API_KEY_HASH=" >> "${local.project_root_env_path}"
      echo "# API_X_API_KEY_HASH=" >> "${local.project_root_env_path}"
      echo "Env file created successfully."
    EOT
  }
}

// configure_yc_cli_profile removed – no CLI profiles

resource "null_resource" "generate_api_keys" {
  # Запускается всегда, когда запускается create_env_file
  triggers = {
    create_env_file_id = null_resource.create_env_file.id
    project_id = local.project_id
  }

  provisioner "local-exec" {
    command = <<-EOT
      envFilePath="${local.project_root_env_path}"
      
      # Check if API keys already exist in the file
      if grep -q "^API_ADMIN_API_KEY_HASH=" "$envFilePath" && grep -q "^API_X_API_KEY_HASH=" "$envFilePath"; then
          echo "API keys already exist in .env file. Skipping generation."
          exit 0
      fi
      
      echo "Generating and adding API keys to .env using Python script..."
      uv run "${local.deployment_scripts_path}/generate_api_keys.py" "$envFilePath"
      echo "API keys added to .env successfully."
    EOT
    interpreter = ["bash", "-c"]
  }

  depends_on = [
    null_resource.create_env_file,
    # configure_yc_cli_profile removed
  ]
}

# ============================================================================
# POSTGRESQL CLUSTER FOR ML SERVICE
# ============================================================================

# PostgreSQL cluster - создается только если не предоставлен existing_postgres_cluster_id
resource "yandex_mdb_postgresql_cluster" "ml_postgres" {
  count       = var.existing_postgres_cluster_id == null ? 1 : 0
  name        = var.postgres_cluster_name
  description = "PostgreSQL cluster for Plastinka ML Service"
  environment = "PRODUCTION"
  network_id  = null  # Will use default network, can be configured via subnet_id
  folder_id   = var.yc_folder_id

  labels = {
    env     = "prod"
    project = "plastinka-sales-predictor"
    service = "ml-database"
  }

  config {
    version = var.postgres_version

    postgresql_config = {
      max_connections                   = 100
      shared_buffers                   = 256MB
      effective_cache_size             = 1GB
      maintenance_work_mem             = 64MB
      checkpoint_completion_target     = 0.9
      wal_buffers                      = 16MB
      default_statistics_target        = 100
      random_page_cost                 = 4.0
      effective_io_concurrency         = 2
      work_mem                         = 4MB
      min_wal_size                     = 1GB
      max_wal_size                     = 4GB
      max_worker_processes             = 4
      max_parallel_workers_per_gather  = 2
      max_parallel_workers            = 4
      max_parallel_maintenance_workers = 2
    }

    resources {
      resource_preset_id = var.postgres_resource_preset_id
      disk_type_id       = var.postgres_disk_type_id
      disk_size          = var.postgres_disk_size
    }

    backup_window_start {
      hours   = 2
      minutes = 0
    }

    backup_retain_period_days = var.postgres_backup_retention_period
  }

  database {
    name  = var.postgres_database_name
    owner = var.postgres_user_name
  }

  user {
    name     = var.postgres_user_name
    password = random_password.postgres_password[0].result
    grants   = []
    permission {
      database_name = var.postgres_database_name
    }
  }

  host {
    zone      = var.postgres_zone_id
    subnet_id = var.postgres_subnet_id
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes = [
      labels
    ]
  }
}

# Generate random password for PostgreSQL user
resource "random_password" "postgres_password" {
  count   = var.existing_postgres_cluster_id == null ? 1 : 0
  length  = 32
  special = true
}

# Store PostgreSQL password in Yandex Lockbox secret (optional, for production)
# For now, password will be output and should be stored securely
