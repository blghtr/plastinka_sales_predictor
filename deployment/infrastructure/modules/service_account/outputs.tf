output "service_account_id" {
  description = "ID of the created service account"
  value       = yandex_iam_service_account.this.id
}

output "service_account_name" {
  description = "Name of the created service account"
  value       = yandex_iam_service_account.this.name
}

output "static_access_key_id" {
  description = "Access key ID for the service account"
  value       = var.create_static_key ? yandex_iam_service_account_static_access_key.this[0].access_key : null
}

output "static_secret_key" {
  description = "Secret access key for the service account"
  value       = var.create_static_key ? yandex_iam_service_account_static_access_key.this[0].secret_key : null
  sensitive   = true
} 