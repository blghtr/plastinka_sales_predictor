resource "yandex_datasphere_community" "this" {
  name            = var.name
  organization_id = var.organization_id
  labels          = var.labels
  
  lifecycle {
    prevent_destroy = true
    # Игнорировать изменения в метках
    ignore_changes = [
      labels
    ]
  }
} 