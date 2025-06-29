resource "yandex_datasphere_community" "this" {
  name            = var.name
  organization_id = var.organization_id
  labels          = var.labels
} 