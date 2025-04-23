# Deployment Guide for Cloud Functions

This document provides instructions for deploying the Plastinka Sales Predictor cloud functions to Yandex Cloud.

## Prerequisites

- Yandex Cloud CLI (`yc`) installed and configured
- Yandex Cloud account with appropriate permissions
- Python 3.10 or higher
- Required environment variables set (or `.env` file)

## Required Environment Variables

The following environment variables are required for deployment:

| Variable | Description |
|----------|-------------|
| `YANDEX_CLOUD_FOLDER_ID` | Your Yandex Cloud folder ID |
| `YANDEX_CLOUD_SERVICE_ACCOUNT_ID` | Your Yandex Cloud service account ID |
| `YANDEX_CLOUD_ACCESS_KEY` | Access key for Yandex Cloud storage |
| `YANDEX_CLOUD_SECRET_KEY` | Secret key for Yandex Cloud storage |
| `CLOUD_CALLBACK_AUTH_TOKEN` | Authentication token for API callbacks |

Optional environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `FASTAPI_CALLBACK_BASE_URL` | Base URL for API callbacks | `http://localhost:8000` |
| `YANDEX_CLOUD_BUCKET` | Cloud storage bucket name | `plastinka-ml-data` |
| `YANDEX_CLOUD_STORAGE_ENDPOINT` | Storage endpoint URL | `https://storage.yandexcloud.net` |
| `YANDEX_CLOUD_REGION` | Cloud region | `ru-central1` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CLOUD_FUNCTION_MEMORY` | Memory limit for functions (MB) | `512` |
| `CLOUD_FUNCTION_TIMEOUT` | Function timeout (seconds) | `300` |
| `ENABLE_ROLLBACK` | Enable automatic rollback | `false` |
| `ENABLE_CONCURRENT_DEPLOY` | Enable concurrent deployment | `false` |
| `VERIFY_DEPLOYMENT` | Verify function after deployment | `false` |

You can use the `check_environment.py` script to validate your environment setup:

```bash
python deployment/scripts/check_environment.py
```

## Basic Deployment

To deploy all cloud functions with default settings:

```bash
python deployment/scripts/deploy_cloud_functions.py
```

This will:

1. Check if all required dependencies are installed
2. Validate environment variables
3. Create ZIP packages for cloud functions
4. Deploy the functions to Yandex Cloud
5. Report the deployment status

## Advanced Deployment Options

### Deploy Specific Function Type

To deploy only the training or prediction function:

```bash
python deployment/scripts/deploy_cloud_functions.py --function-type training
# or
python deployment/scripts/deploy_cloud_functions.py --function-type prediction
```

### Configure Memory and Timeout

Adjust memory allocation and execution timeout:

```bash
python deployment/scripts/deploy_cloud_functions.py --memory 1024 --timeout 600
```

### Use Environment File

Use variables from a specific .env file:

```bash
python deployment/scripts/deploy_cloud_functions.py --env-file .env.production
```

### Automatic Rollback

Enable automatic rollback on deployment failure:

```bash
python deployment/scripts/deploy_cloud_functions.py --rollback-on-failure
```

### Skip ZIP Generation

To skip generating new ZIP files (use existing ones):

```bash
python deployment/scripts/deploy_cloud_functions.py --skip-zipgen
```

### Concurrent Deployment

Deploy multiple functions concurrently for faster deployment:

```bash
python deployment/scripts/deploy_cloud_functions.py --concurrent
```

### Dry Run Mode

Preview what would be deployed without actually deploying:

```bash
python deployment/scripts/deploy_cloud_functions.py --dry-run
```

### Deployment Verification

Verify that functions are active after deployment:

```bash
python deployment/scripts/deploy_cloud_functions.py --verify
```

## Deployment Process Flow

The enhanced deployment script follows this process:

1. **Preparation Phase**:
   - Validate environment
   - Parse command-line arguments
   - Check dependencies

2. **ZIP Creation Phase**:
   - Create temporary directory
   - Copy function code
   - Add requirements file
   - Create ZIP archive

3. **Deployment Phase**:
   - Store previous version (for potential rollback)
   - Check if function exists
   - Create or update function
   - Create new function version

4. **Verification Phase** (if enabled):
   - Check function status
   - Wait for activation
   - Retrieve function URL

5. **Summary Phase**:
   - Report deployment status for each function
   - Exit with appropriate code

## Troubleshooting

### Command not found: yc

Ensure Yandex Cloud CLI is installed and in your PATH.

### Missing environment variables

Use the `check_environment.py` script to identify and set missing variables.

### ZIP creation fails

Check file permissions and available disk space.

### Deployment timeout

Try increasing the timeout value with `--timeout` option.

### Concurrent deployment issues

If you encounter issues with concurrent deployment, try using sequential deployment instead.

## Advanced Configuration

### Custom Requirements File

Specify a custom requirements file for cloud functions:

```bash
python deployment/scripts/deploy_cloud_functions.py --requirements-file custom_requirements.txt
```

### Modifying Deployment Settings

For persistent changes to deployment settings, edit the environment variables in your `.env` file or system environment. 