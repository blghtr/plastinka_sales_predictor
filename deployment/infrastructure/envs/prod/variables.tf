variable "yc_token" {
  description = "Yandex Cloud OAuth token"
  type        = string
  sensitive   = true
  default     = null
}

variable "yc_folder_id" {
  description = "Target Yandex Cloud Folder ID"
  type        = string
}

variable "yc_cloud_id" {
  description = "Yandex Cloud ID"
  type        = string
  default     = null
}

variable "yc_organization_id" {
  description = "Yandex Cloud Organization ID"
  type        = string
} 

# Project paths
variable "project_root" {
  description = "Path to project root directory"
  type        = string
  default     = "../../../../"
}

# ============================================================================
# ПЕРЕМЕННЫЕ ДЛЯ УСЛОВНОГО ИМПОРТА СУЩЕСТВУЮЩИХ РЕСУРСОВ
# ============================================================================

variable "existing_service_account_id" {
  description = "ID существующего сервисного аккаунта для импорта (опционально). Если указан, новый сервисный аккаунт создаваться не будет."
  type        = string
  default     = null
  
  validation {
    condition = var.existing_service_account_id == null || can(regex("^[a-z0-9]+$", var.existing_service_account_id))
    error_message = "Service Account ID должен содержать только строчные буквы и цифры."
  }
}

variable "existing_community_id" {
  description = "ID существующего DataSphere Community для импорта (опционально). Если указан, новый community создаваться не будет."
  type        = string
  default     = null
  
  validation {
    condition = var.existing_community_id == null || can(regex("^[a-z0-9]+$", var.existing_community_id))
    error_message = "Community ID должен содержать только строчные буквы и цифры."
  }
}

variable "existing_project_id" {
  description = "ID существующего DataSphere Project для импорта (опционально). Если указан, новый project создаваться не будет."
  type        = string
  default     = null
  
  validation {
    condition = var.existing_project_id == null || can(regex("^[a-z0-9]+$", var.existing_project_id))
    error_message = "Project ID должен содержать только строчные буквы и цифры."
  }
}

variable "existing_static_key_id" {
  description = "ID существующего статического ключа доступа для импорта (опционально). Если указан, новый ключ создаваться не будет."
  type        = string
  default     = null
  
  validation {
    condition = var.existing_static_key_id == null || can(regex("^[a-zA-Z0-9]+$", var.existing_static_key_id))
    error_message = "Static Key ID должен содержать только заглавные и строчные буквы и цифры."
  }
}

# ============================================================================
# POSTGRESQL CLUSTER VARIABLES
# ============================================================================

variable "postgres_cluster_name" {
  description = "Name of the PostgreSQL cluster for ML service"
  type        = string
  default     = "plastinka-ml-postgres"
}

variable "postgres_database_name" {
  description = "Name of the database within PostgreSQL cluster"
  type        = string
  default     = "plastinka_ml"
}

variable "postgres_user_name" {
  description = "Name of the PostgreSQL user"
  type        = string
  default     = "plastinka_ml_user"
}

variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15"
}

variable "postgres_resource_preset_id" {
  description = "Resource preset ID for PostgreSQL hosts (e.g., s2.micro, s2.small)"
  type        = string
  default     = "s2.micro"
}

variable "postgres_disk_type_id" {
  description = "Disk type ID for PostgreSQL (e.g., network-ssd, network-hdd)"
  type        = string
  default     = "network-ssd"
}

variable "postgres_disk_size" {
  description = "Disk size in GB for PostgreSQL"
  type        = number
  default     = 20
}

variable "postgres_zone_id" {
  description = "Availability zone ID for PostgreSQL cluster"
  type        = string
  default     = "ru-central1-a"
}

variable "postgres_subnet_id" {
  description = "Subnet ID for PostgreSQL cluster (optional, will use default if not provided)"
  type        = string
  default     = null
}

variable "postgres_backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

variable "existing_postgres_cluster_id" {
  description = "ID существующего PostgreSQL кластера для импорта (опционально). Если указан, новый кластер создаваться не будет."
  type        = string
  default     = null
} 