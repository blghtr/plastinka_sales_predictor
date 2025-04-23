# Environment Variables Documentation

This document describes all environment variables used by the Plastinka Sales Predictor application. These environment variables control application behavior across different environments like development, testing, and production.

## Setting Environment Variables

Environment variables can be set in multiple ways:

1. **System environment variables**: Set using operating system methods
2. **`.env` file**: Create a `.env` file in the project root directory
3. **Command-line arguments**: Some variables can be overridden via command-line args in deployment scripts

## Core Application Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `APP_ENV` | Application environment | `development` | `production`, `testing` |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | `http://localhost:3000` | `https://app.example.com,https://admin.example.com` |
| `LOG_LEVEL` | Application logging level | `INFO` | `DEBUG`, `WARNING`, `ERROR` |

## Database Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DATABASE_PATH` | Path to SQLite database file | `deployment/data/plastinka.db` | `/var/data/plastinka.db` |

## Cloud Storage Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `YANDEX_CLOUD_STORAGE_ENDPOINT` | Endpoint URL for cloud storage | `https://storage.yandexcloud.net` | - |
| `YANDEX_CLOUD_REGION` | Region for cloud storage | `ru-central1` | - |
| `YANDEX_CLOUD_BUCKET` | Bucket name for ML data | `plastinka-ml-data` | `my-custom-bucket` |
| `YANDEX_CLOUD_ACCESS_KEY` | Access key for cloud storage | - | `AKIAIOSFODNN7EXAMPLE` |
| `YANDEX_CLOUD_SECRET_KEY` | Secret key for cloud storage | - | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |

## Cloud Functions Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `YANDEX_CLOUD_FOLDER_ID` | Yandex Cloud folder ID | - | `b1gvmob95yysaplct532` |
| `YANDEX_CLOUD_SERVICE_ACCOUNT_ID` | Service account ID | - | `aje123mnd82jfk2lf10j` |
| `YANDEX_CLOUD_API_KEY` | API key for cloud functions | - | `AQVN1XbUfO8qA15TdJqU-miAXoFql1sPvEz9` |
| `CLOUD_FUNCTION_MEMORY` | Memory limit for cloud functions (MB) | `512` | `1024`, `2048` |
| `CLOUD_FUNCTION_TIMEOUT` | Timeout for cloud functions (seconds) | `300` | `600` |
| `ENABLE_ROLLBACK` | Enable automatic rollback on deployment failure | `false` | `true` |

## API Callback Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `FASTAPI_CALLBACK_BASE_URL` | Base URL for cloud function callbacks | `http://localhost:8000` | `https://api.example.com` |
| `CLOUD_CALLBACK_AUTH_TOKEN` | Authentication token for callbacks | - | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` |

## Upload Limits

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MAX_UPLOAD_SIZE` | Maximum size for file uploads (bytes) | `52428800` (50MB) | `104857600` (100MB) |

## Deployment Script Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CLOUD_ENV_FILE` | Path to environment file for cloud functions | `.env` | `/path/to/cloud.env` |
| `CLOUD_REQUIREMENTS_FILE` | Path to requirements file for cloud functions | `deployment/scripts/cloud_requirements.txt` | `/path/to/requirements.txt` |

## Example .env File

```
# API Configuration
APP_ENV=development
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Database Configuration
DATABASE_PATH=deployment/data/plastinka.db

# Logging
LOG_LEVEL=INFO

# Cloud Integration
YANDEX_CLOUD_STORAGE_ENDPOINT=https://storage.yandexcloud.net
YANDEX_CLOUD_REGION=ru-central1
YANDEX_CLOUD_BUCKET=plastinka-ml-data
YANDEX_CLOUD_ACCESS_KEY=your_access_key_here
YANDEX_CLOUD_SECRET_KEY=your_secret_key_here

# Yandex Cloud Functions
YANDEX_CLOUD_FOLDER_ID=your_folder_id_here
YANDEX_CLOUD_SERVICE_ACCOUNT_ID=your_service_account_id_here
YANDEX_CLOUD_API_KEY=your_api_key_here
CLOUD_FUNCTION_MEMORY=512
CLOUD_FUNCTION_TIMEOUT=300
ENABLE_ROLLBACK=true

# API Callbacks
FASTAPI_CALLBACK_BASE_URL=http://localhost:8000
CLOUD_CALLBACK_AUTH_TOKEN=your_auth_token_here

# Upload Limits
MAX_UPLOAD_SIZE=52428800
```

## Security Considerations

- Never commit sensitive credentials to version control
- Restrict access to environment files containing credentials
- In production, use proper secret management services where available
- Regularly rotate authentication tokens and API keys 