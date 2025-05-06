terraform {
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.84.0" // Specify a suitable version
    }
  }
  required_version = ">= 1.0.0" // OpenTofu version requirement

  // Using local backend for initial testing
  // Commented out S3 backend configuration for later use
  /*
  backend "s3" {
    endpoint   = "storage.yandexcloud.net"
    bucket     = "plastinka-tofu-state" // Static bucket name for state
    key        = "tofu-state/plastinka.tfstate" // Path within the bucket for the state file
    region     = "ru-central1" // Specify the region of the bucket

    // Authentication: Configure via static credentials or environment variables
    // access_key = "..."
    // secret_key = "..."

    // Optional: Enable state locking if using DynamoDB compatible table in YDB
    // dynamodb_endpoint = "https://docapi.serverless.yandexcloud.net/ru-central1/..."
    // dynamodb_table  = "terraform_locks"

    skip_region_validation      = true
    skip_credentials_validation = true
  }
  */
}

provider "yandex" {
  // Configuration options:
  // Use variables for configuration flexibility
  token                 = var.yc_token
  service_account_key_file = null // Explicitly set to null to prioritize token auth
  folder_id             = var.yc_folder_id
  zone                  = var.yc_zone
  organization_id       = var.yc_organization_id # Use variable for Organization ID
  // zone                  : Default availability zone - configure if needed
}

// Placeholder for DataSphere Project and other resources
resource "yandex_datasphere_project" "project" {
  name         = "plastinka-sales-predictor-ds" // Consider parameterizing or using a variable
  description  = "DataSphere project for Plastinka Sales Predictor"
  community_id = yandex_datasphere_community.main.id // Assumes community is managed here

  // Default settings, adjust as needed
  settings = {
    max_units_per_hour      = 10
    max_units_per_execution = 100
    // Other settings... e.g., network_id, subnet_id if needed
  }

  // Default limits, adjust as needed
  limits = {
    max_units_per_hour      = 10
    max_units_per_execution = 100
  }
}

// Manage DataSphere Community via IaC (Example)
resource "yandex_datasphere_community" "main" {
  name      = "plastinka-community" // Consider parameterizing
}

// S3 Bucket for Code Bundles and Models
resource "yandex_storage_bucket" "datasphere_artifacts" {
  // Use environment variables or tfvars for bucket name to ensure uniqueness
  bucket = "${var.yc_folder_id}-ds-artifacts-${random_id.bucket_suffix.hex}" // Example naming convention

  // Configure access policies, versioning etc. as needed
  // Example: Private bucket
  acl    = "private"

  folder_id = var.yc_folder_id // Use variable for folder ID
}

// Random suffix for bucket name uniqueness
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

// TODO: Add resources for IAM roles, Service Accounts, etc.
// TODO: Configure backend state storage (e.g., S3)
// TODO: Add resources for S3 bucket, IAM roles, etc. 