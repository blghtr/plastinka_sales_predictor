variable "name" {
  description = "Name of the service account"
  type        = string
}

variable "description" {
  description = "Description of the service account"
  type        = string
  default     = ""
}

variable "folder_id" {
  description = "Folder ID where service account will be created"
  type        = string
}

variable "cloud_id" {
  description = "Cloud ID for cloud-level role assignments"
  type        = string
  default     = ""
}

variable "create_static_key" {
  description = "Whether to create static access key for the service account"
  type        = bool
  default     = true
}

variable "folder_roles" {
  description = "List of roles to assign to the service account at folder level"
  type        = list(string)
  default     = []
}

variable "cloud_roles" {
  description = "List of roles to assign to the service account at cloud level"
  type        = list(string)
  default     = []
} 