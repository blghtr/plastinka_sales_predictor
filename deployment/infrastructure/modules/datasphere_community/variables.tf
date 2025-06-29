variable "name" {
  description = "Community name"
  type        = string
}

variable "organization_id" {
  description = "Organization ID"
  type        = string
}

variable "labels" {
  description = "Labels to attach to community"
  type        = map(string)
  default     = {}
} 