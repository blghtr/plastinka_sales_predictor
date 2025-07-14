resource "yandex_datasphere_project" "this" {
  name         = var.name
  community_id = var.community_id
  description  = var.description

  settings = {
    service_account_id = var.service_account_id
  }

  limits = {
    max_units_per_hour      = var.max_units_per_hour
    max_units_per_execution = var.max_units_per_execution
  }

  labels = var.labels

  lifecycle {
    prevent_destroy = true
    # Игнорировать изменения в описании и метках
    ignore_changes = [
      description,
      labels
    ]
  }
} 