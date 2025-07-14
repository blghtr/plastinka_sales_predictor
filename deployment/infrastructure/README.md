# Terraform Infrastructure for Plastinka Sales Predictor

This directory contains a declarative description of Yandex Cloud resources required for the DataSphere project, with automatic creation and configuration of a service account.

> **Note**: This infrastructure is part of the comprehensive Plastinka Sales Predictor system. See the [main README](../../README.md) for a full system overview.

## Structure

```
modules/                    # Reusable modules
  datasphere_community/     # YC DataSphere Community
  datasphere_project/       # YC DataSphere Project  
  service_account/          # YC Service Account with IAM roles

envs/
  prod/                     # Configuration for production (single environment)
    main.tf                 # Main resource configuration
    variables.tf            # Variable definitions
    outputs.tf              # Output values
    terraform.tfvars        # Variable values (not in VCS)
    terraform.tfvars.example # Example variable file

versions.tf                 # Global Terraform and provider version constraints
```

## What is Created

This Terraform configuration deploys the necessary Yandex Cloud infrastructure for the Plastinka Sales Predictor, including:

- **DataSphere Service Account**: A service account with the minimum necessary roles for working with DataSphere, Object Storage, and other cloud resources.
- **DataSphere Community**: An organizational unit for grouping projects.
- **DataSphere Project**: A DataSphere project with configured resource limits and绑定 to a service account.
- **Automatic generation of `.env` file and API keys**: Upon the first application, Terraform will automatically create or update the `.env` file in the project root, and generate and add the necessary API keys for interaction with the FastAPI application.

For a more detailed description of the system components, including the ML module and FastAPI application, please refer to the [main README](../../README.md).

## Quick Start

To quickly deploy the infrastructure, follow these steps:

### 1. Prepare Variables

Navigate to the `deployment/infrastructure/envs/prod` directory and copy the `terraform.tfvars.example` file to `terraform.tfvars`:

```bash
cd deployment/infrastructure/envs/prod
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`, specifying your `yc_cloud_id`, `yc_folder_id`, and `yc_organization_id`.

For Terraform authentication, use the `TF_VAR_yc_token` environment variable:

```bash
export TF_VAR_yc_token="your-oauth-token"
terraform apply
```

### 2. Initialization and Application

```bash
terraform init
terraform plan
terraform apply
```

This command will automatically configure the `yc CLI` profile, generate `sa-key.json`, and populate the `.env` file in the project root with the necessary environment variables, including API keys for the FastAPI application. Ensure that Python 3.x is installed in the environment where `terraform apply` is run.


## 🔄 Using Existing Resources

**New Feature!** You can now connect existing DataSphere infrastructure to Terraform management without recreating it.

### When to Use
- ✅ You already have configured DataSphere resources
- ✅ Need to migrate a project under Terraform management
- ✅ Want to work in a hybrid mode (part existing, part new)

### Quick Connection of Existing Resources

1. **Get IDs of existing resources:**
```bash
# Service Account
yc iam service-account list --format json | jq -r '.[] | select(.name=="datasphere-sa-prod") | .id'

# Community  
yc datasphere community list --format json | jq -r '.[] | select(.name=="prod-ds-community") | .id'

# Project
yc datasphere project list --community-id YOUR_COMMUNITY_ID --format json | jq -r '.[] | select(.name=="prod-ds-project") | .id'
```

2. **Add to terraform.tfvars:**
```hcl
# Core variables
yc_cloud_id        = "your-cloud-id"
yc_folder_id       = "your-folder-id"
yc_organization_id = "your-org-id"

# Using existing resources
existing_service_account_id = "your-existing-service-account-id"
existing_community_id       = "your-existing-community-id"
existing_project_id         = "your-existing-project-id"
# existing_static_key_id - if you already have a static key and want to use it
```

3. **Apply the configuration:**
```bash
terraform init
terraform plan  # Verify - no new resources are created!
terraform apply
```

### How it Works?

We use a **pure Data Source approach** instead of import:
- ✅ **Simplicity** - no need for complex imports
- ✅ **Reliability** - standard Terraform capabilities
- ✅ **Flexibility** - can combine new and existing resources
- ✅ **Security** - existing resources are not modified
- ✅ **Compatibility** - works with any Terraform version

### Detailed Documentation
📖 **[Full Guide](envs/prod/IMPORT_GUIDE.md)** - detailed instructions, usage scenarios, troubleshooting.

### Resource Status Check
```bash
# See which resources are existing and which are new
terraform output import_status

# Example output:
# {
#   "community": "existing",
#   "project": "created", 
#   "service_account": "existing",
#   "static_key": "existing"
# }
```

## Retrieving Data After Apply

After successfully applying Terraform, you can retrieve important output data necessary for further system configuration and operation. This data includes IDs of created resources and access keys.

### Main Output Values:
```bash
# DataSphere summary information
terraform output datasphere_summary

# DataSphere project ID
terraform output datasphere_project_id

# Service Account data
terraform output service_account_id
terraform output service_account_name

# Access keys (sensitive)
terraform output -raw static_access_key_id
terraform output -raw static_secret_key
```

## Integration with the Main System

After infrastructure creation, it automatically integrates with the FastAPI application. The API application uses environment variables to connect to DataSphere. Job management, monitoring, and result retrieval are performed via API endpoints that interact with DataSphere.

## Additional Information

For detailed information on API integration, monitoring, logging, and security aspects, please refer to the [README.md in the `deployment` directory](../README.md).