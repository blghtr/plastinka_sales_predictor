# Terraform Infrastructure for Plastinka Sales Predictor

This directory contains a declarative description of Yandex Cloud resources required for the DataSphere project.

> **Note**: This infrastructure is part of the comprehensive Plastinka Sales Predictor system. See the [main README](../../README.md) for a full system overview.

## Structure

```
modules/                    # Reusable modules
  datasphere_community/     # YC DataSphere Community
  datasphere_project/       # YC DataSphere Project  

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

- **DataSphere Community**: An organizational unit for grouping projects.
- **DataSphere Project**: A DataSphere project with configured resource limits.
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

This command will populate the `.env` file in the project root with the necessary environment variables, including API keys for the FastAPI application. Ensure that Python 3.x is installed in the environment where `terraform apply` is run.


## 🔄 Using Existing Resources

**New Feature!** You can now connect existing DataSphere infrastructure to Terraform management without recreating it.

### When to Use
- ✅ You already have configured DataSphere resources
- ✅ Need to migrate a project under Terraform management
- ✅ Want to work in a hybrid mode (part existing, part new)

### Quick Connection of Existing Resources

1. **Get IDs of existing resources:**
```bash
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
existing_community_id       = "your-existing-community-id"
existing_project_id         = "your-existing-project-id"
```

3. **Apply the configuration:**
```bash
terraform init
terraform plan  # Verify - no new resources are created!
terraform apply
```

### Resource Status Check
```bash
# See which resources are existing and which are new
terraform output import_status

# Example output:
# {
#   "community": "existing",
#   "project": "created"
# }
```

## Retrieving Data After Apply

After successfully applying Terraform, you can retrieve important output data necessary for further system configuration and operation. This data includes IDs of created resources.

### Main Output Values:
```bash
# DataSphere summary information
terraform output datasphere_summary

# DataSphere project ID
terraform output datasphere_project_id
```

## Integration with the Main System

After infrastructure creation, it automatically integrates with the FastAPI application. The API application uses environment variables to connect to DataSphere. Job management, monitoring, and result retrieval are performed via API endpoints that interact with DataSphere.

## Additional Information

For detailed information on API integration, monitoring, logging, and security aspects, please refer to the [README.md in the `deployment` directory](../README.md).
