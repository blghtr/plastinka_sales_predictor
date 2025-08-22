# Terraform Infrastructure for Plastinka Sales Predictor

This directory contains a declarative description of Yandex Cloud resources required for the DataSphere project.

> **Note**: This infrastructure is part of the comprehensive Plastinka Sales Predictor system. See the [main README](../../README.md) for a full system overview and getting started guide.

## What is Created

This Terraform configuration deploys the necessary Yandex Cloud infrastructure, including:

- **DataSphere Community**: An organizational unit for grouping projects.
- **DataSphere Project**: A DataSphere project with configured resource limits.
- **Automatic `.env` file generation**: Terraform automatically creates or updates the `.env` file in the project root, populating it with API keys and resource IDs needed by the FastAPI application.

---

## Quick Start

Follow these steps to deploy the infrastructure.

### Prerequisites

- You have an active Yandex Cloud account.
- You have installed and configured your tool of choice:
  - [Terraform](https://developer.hashicorp.com/terraform/install) (v1.0+)
  - or [OpenTofu](https://opentofu.org/docs/intro/install/)
- You have obtained your Yandex Cloud `cloud_id`, `folder_id`, and `organization_id`.

### 1. Prepare Variables

Navigate to the environment directory and create a `terraform.tfvars` file from the example.

```bash
cd deployment/infrastructure/envs/prod
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` and fill in your `yc_cloud_id`, `yc_folder_id`, and `yc_organization_id`.

### 2. Set Authentication Token

For Terraform to authenticate with Yandex Cloud, export your OAuth token as an environment variable.

```bash
export TF_VAR_yc_token="your-yc-oauth-token"
```

### 3. Initialize and Apply

Run the standard Terraform commands to deploy the resources.

```bash
# Initialize the backend and providers
terraform init

# (Optional) Preview the changes
terraform plan

# Apply the configuration
terraform apply
```

This command will provision the cloud resources and populate the `.env` file in the project root, ensuring the API can connect to DataSphere.

---

## 🔄 Using Existing Resources

You can connect existing DataSphere resources to Terraform management without recreating them.

### How to Use

1.  **Get the IDs** of your existing DataSphere Community and Project.
2.  In the `terraform.tfvars` file, uncomment and set the `existing_community_id` and `existing_project_id` variables.
3.  Run `terraform apply`. Terraform will adopt these resources instead of creating new ones.

You can check which resources were adopted and which were created using the `terraform output import_status` command.