resource "yandex_iam_service_account" "this" {
  name        = var.name
  description = var.description
  folder_id   = var.folder_id
}

resource "yandex_iam_service_account_static_access_key" "this" {
  count              = var.create_static_key ? 1 : 0
  service_account_id = yandex_iam_service_account.this.id
  description        = "Static access key for ${var.name}"
}

# Assign roles at folder level for DataSphere operations
resource "yandex_resourcemanager_folder_iam_member" "this" {
  for_each  = toset(var.folder_roles)
  folder_id = var.folder_id
  role      = each.value
  member    = "serviceAccount:${yandex_iam_service_account.this.id}"
}

# Assign roles at cloud level if needed
resource "yandex_resourcemanager_cloud_iam_member" "this" {
  for_each = toset(var.cloud_roles)
  cloud_id = var.cloud_id
  role     = each.value
  member   = "serviceAccount:${yandex_iam_service_account.this.id}"
} 