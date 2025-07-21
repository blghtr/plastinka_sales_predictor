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
    # Запускается, если .env файл не существует
    run_once = fileexists("${local.project_root_env_path}") ? "exists" : "create"
  }

  provisioner "local-exec" {
    when    = create
    interpreter = ["bash", "-c"]
    command = <<-EOT
      echo "# DataSphere Configuration (REQUIRED)" > "${local.project_root_env_path}"
      echo "DATASPHERE_PROJECT_ID=${local.project_id}" >> "${local.project_root_env_path}"
      echo "DATASPHERE_FOLDER_ID=${var.yc_folder_id}" >> "${local.project_root_env_path}"
      echo "DATASPHERE_YC_PROFILE=datasphere-prod" >> "${local.project_root_env_path}"
      echo "" >> "${local.project_root_env_path}"
      echo "# API Security (REQUIRED)" >> "${local.project_root_env_path}"
      echo "# These will be automatically generated and populated by the Python script." >> "${local.project_root_env_path}"
      echo "# Ensure Python 3.x is installed in the environment where 'terraform apply' is run." >> "${local.project_root_env_path}"
      echo "# API_ADMIN_API_KEY=" >> "${local.project_root_env_path}"
      echo "# API_X_API_KEY=" >> "${local.project_root_env_path}"
    EOT
  }
}

resource "null_resource" "configure_yc_cli_profile" {
  # Only run when token changes
  triggers = {
    yc_token_md5 = md5(var.yc_token != null ? var.yc_token : "")
  }

  provisioner "local-exec" {
    command = <<-EOT
      profileName="datasphere-prod"
      cloudId="${var.yc_cloud_id}"
      ycToken="${var.yc_token}" # Using var.yc_token as it's passed via TF_VAR_yc_token
      
      # Check if profile exists
      if yc config profile get "$profileName" >/dev/null 2>&1; then
          echo "yc CLI profile '$profileName' already exists. Updating..."
      else
          echo "yc CLI profile '$profileName' does not exist. Creating..."
          yc config profile create "$profileName"
      fi

      # Set OAuth token
      if [ -n "$ycToken" ]; then
          echo "Setting OAuth token for profile '$profileName'."
          yc config set token "$ycToken" --profile "$profileName"
      else
          echo "TF_VAR_yc_token is not set. Skipping OAuth token setup."
      fi

      echo "yc CLI profile configuration complete for '$profileName'."
    EOT
    interpreter = ["bash", "-c"]
  }
}

resource "null_resource" "generate_api_keys" {
  # Only run when env file changes or doesn't exist
  triggers = {
    create_env_file_id = null_resource.create_env_file.id
  }

  provisioner "local-exec" {
    command = <<-EOT
      envFilePath="${local.project_root_env_path}"
      
      # Check if API keys already exist in the file
      if grep -q "^API_ADMIN_API_KEY=" "$envFilePath" && grep -q "^API_X_API_KEY=" "$envFilePath"; then
          echo "API keys already exist in .env file. Skipping generation."
          exit 0
      fi
      
      echo "Generating and adding API keys to .env using Python script..."
      python "${local.deployment_scripts_path}/generate_api_keys.py" "$envFilePath"
      echo "API keys added to .env successfully."
    EOT
    interpreter = ["bash", "-c"]
  }

  depends_on = [
    null_resource.create_env_file,
    null_resource.configure_yc_cli_profile
  ]
}
