variable "yc_folder_id" {
  description = "Yandex Cloud Folder ID"
  type        = string
  // No default, should be provided via environment variable (TF_VAR_yc_folder_id) or .tfvars file
}

variable "yc_cloud_id" {
  description = "Yandex Cloud ID"
  type        = string
  default     = null // Optional when using service account key
  // No default, should be provided via environment variable (TF_VAR_yc_cloud_id) or .tfvars file
}

variable "yc_token" {
  description = "Yandex Cloud OAuth Token (optional, used for provider auth if SA key not provided)"
  type        = string
  sensitive   = true
  default     = null // Prefer SA key or environment variables
}

variable "yc_organization_id" {
  description = "Yandex Cloud Organization ID (Required for DataSphere Community)"
  type        = string
  default     = "" # Ensure this is set via environment variable or terraform.tfvars
}

variable "service_account_key_file" {
  description = "Path to the service account key file (optional, used for provider auth)"
  type        = string
  default     = null // Prefer environment variables or token
}

variable "trigger_function_name" {
  description = "Name for the trigger Cloud Function"
  type        = string
  default     = "datasphere-trigger-function"
}

variable "trigger_function_runtime" {
  description = "Runtime environment for the trigger function"
  type        = string
  default     = "python311" // Example, adjust as needed
}

variable "trigger_function_memory" {
  description = "Memory allocation for the trigger function in MB"
  type        = number
  default     = 128
}

variable "yc_zone" {
  description = "Yandex Cloud Availability Zone"
  type        = string
  default     = "ru-central1-a" // Or another suitable default
}

// Add other variables as needed (e.g., for S3 bucket names if not generated, SA names, etc.) 