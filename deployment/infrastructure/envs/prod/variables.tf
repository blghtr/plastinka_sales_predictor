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