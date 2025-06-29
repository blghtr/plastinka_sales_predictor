variable "name" {
  description = "Project name"
  type        = string
}

variable "description" {
  description = "Project description"
  type        = string
  default     = ""
}

variable "community_id" {
  description = "Community ID to attach project"
  type        = string
}

variable "max_units_per_hour" {
  description = "Billing limit per hour"
  type        = number
  default     = 20
}

variable "max_units_per_execution" {
  description = "Billing limit per execution"
  type        = number
  default     = 200
}

variable "labels" {
  description = "Labels to attach"
  type        = map(string)
  default     = {}
}

variable "service_account_id" {
  description = "Service account ID to use for DataSphere project operations"
  type        = string
  default     = null
} 