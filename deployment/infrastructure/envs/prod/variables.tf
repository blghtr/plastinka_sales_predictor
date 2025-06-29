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