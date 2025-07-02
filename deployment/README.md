# Plastinka Sales Predictor API

A FastAPI application for predicting vinyl record sales using time series forecasting, with cloud-based computation via Yandex DataSphere.

## Overview

This API provides endpoints for:

1. **Data Upload** - Process and store raw sales and stock data
2. **Model Training & Prediction** - Train TiDE models and generate predictions in one step
3. **Reports** - Generate analysis reports with visualizations

All operations are handled as asynchronous jobs with status tracking and cloud computation offloading.

## 🔐 Authentication Setup

### Service Account Authentication (Recommended for Production)

For production deployments, it's recommended to use service account authentication through YC CLI profiles instead of OAuth tokens.

#### 1. Get Service Account Information from Terraform

```bash
cd deployment/infrastructure/envs/prod
terraform output service_account_id
terraform output -raw static_access_key_id  
terraform output -raw static_secret_key
```

#### 2. Create Service Account Key

```bash
# Get the service account ID from terraform output
SA_ID=$(terraform output -raw service_account_id)

# Create JSON key for the service account
yc iam key create --service-account-id $SA_ID --output sa-key.json
```

#### 3. Configure YC CLI Profile

```bash
# Create a new YC CLI profile for DataSphere
yc config profile create datasphere-prod

# Set the service account key
yc config set service-account-key sa-key.json

# Verify the profile configuration
yc config list
```

#### 4. Configure Required Environment Variables

Create a `.env` file with the following **REQUIRED** variables:

```bash
# Required DataSphere Configuration
DATASPHERE_PROJECT_ID=$(terraform output -raw datasphere_project_id)
DATASPHERE_FOLDER_ID=your-folder-id
DATASPHERE_YC_PROFILE=datasphere-prod

# Required API Security
API_X_API_KEY=your-api-key

# Optional: Authentication method (defaults to "auto")
DATASPHERE_AUTH_METHOD=yc_profile
```

#### 5. Test the Configuration

```bash
# Start the API to test the configuration
cd deployment
python run.py
```

Check the logs for authentication method confirmation:
```
INFO - DataSphere client initialized with YC profile: datasphere-prod
```

### OAuth Token Authentication (Legacy/Development)

For development or if you prefer OAuth tokens:

```bash
export DATASPHERE_AUTH_METHOD="oauth_token"
export DATASPHERE_OAUTH_TOKEN="your-oauth-token"
# YC profile settings are ignored in this mode
```

### Auto-Detection Mode (Default)

The system will automatically choose the best authentication method:
1. **YC Profile** (if `DATASPHERE_YC_PROFILE` is set)
2. **OAuth Token** (if `DATASPHERE_OAUTH_TOKEN` is set and no YC profile)
3. **SDK Auto-detection** (fallback to DataSphere SDK's built-in logic)

```bash
export DATASPHERE_AUTH_METHOD="auto"  # This is the default
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         Plastinka Sales Predictor API                       │
│                                                                             │
│  ┌───────────────┐    ┌───────────────┐    ┌────────────┐    ┌───────────┐  │
│  │  Client App   │────│  FastAPI      │────│  Database  │────│ File      │  │
│  │  (Web UI)     │    │  Application  │    │  (SQLite)  │    │ Storage   │  │
│  └───────────────┘    └───────────────┘    └────────────┘    └───────────┘  │
│                              │                                               │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Background Task Processing                       │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐                   ┌───────────┐  │   │
│  │  │ Data Upload │  │ Train+Predict│                  │  Reports  │  │   │
│  │  │ Processing  │  │   Jobs      │                  │           │  │   │
│  │  └─────────────┘  └─────────────┘                  └───────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Yandex Cloud Integration                            │
│                                                                              │
│                      ┌────────────────────────────────┐                      │
│                      │          DataSphere            │                      │
│                      │  - ML Compute                  │                      │
│                      │  - Job Runner                  │                      │
│                      │  - Project Storage (temporary) │                      │
│                      │                                │                      │
│                      └────────────────────────────────┘                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## System Requirements

- Python 3.10+
- SQLite3
- 2GB+ RAM
- 1GB+ disk space
- Yandex Cloud account and credentials

## Setup Instructions

### 1. Check Environment Prerequisites

```bash
# Verify your environment setup
python deployment/scripts/check_environment.py
```

This will generate a `.env.template` file that you can fill with your values.

### 2. Set Environment Variables

The `check_environment.py` script will generate a `.env.template` file that you can use as a starting point.

#### Required Variables (must be set in .env file):

Create a `.env` file with these **REQUIRED** variables:
```bash
# DataSphere Configuration (REQUIRED)
DATASPHERE_PROJECT_ID=your-project-id
DATASPHERE_FOLDER_ID=your-folder-id
DATASPHERE_YC_PROFILE=datasphere-prod

# API Security (REQUIRED)
API_X_API_KEY=your-api-key
```

Alternatively, you can export these as environment variables:
```bash
export DATASPHERE_PROJECT_ID="your-project-id"
export DATASPHERE_FOLDER_ID="your-folder-id" 
export DATASPHERE_YC_PROFILE="datasphere-prod"
export API_X_API_KEY="your-api-key"
```

#### Optional Variables (have defaults):
```bash
# Authentication Method (defaults to "auto")
DATASPHERE_AUTH_METHOD=auto

# For legacy OAuth authentication (optional)
DATASPHERE_OAUTH_TOKEN=your-oauth-token

# API Configuration (optional, defaults shown)
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
MAX_UPLOAD_SIZE=52428800
APP_ENV=development

# Data Storage (optional, defaults shown)
DATA_ROOT_DIR=~/.plastinka_sales_predictor
```

### 3. Deploy Cloud Infrastructure (Optional)

If you need to set up the cloud infrastructure:

```bash
cd deployment/infrastructure
terraform init
terraform apply
```

### 4. Run the Application

```bash
cd deployment
python run.py
```

For development with auto-reload:
```bash
python run.py --reload
```

The API will be available at http://localhost:8000

## API Endpoints

### Job Management

- `POST /api/v1/jobs/data-upload` - Upload and process sales data
  - Accepts Excel (.xlsx, .xls) and CSV files for stock and sales data
  - Supports automatic format detection and encoding handling
  - Performs validation and preprocessing
  - Stores structured data in the database

- `POST /api/v1/jobs/training` - Train a model and generate predictions
  - Uses active parameter set configuration
  - Executes training in Yandex DataSphere
  - Generates predictions during training
  - Downloads and stores model locally
  - Stores predictions in the database
  - Records model metadata in database

- `POST /api/v1/jobs/reports` - Generate reports
  - Creates HTML reports with visualizations
  - Supports various report types and formats
  - Uses previously generated predictions

- `GET /api/v1/jobs/{job_id}` - Get job status and results
  - Returns current status, progress, and result data
  - Provides error information if job failed

- `GET /api/v1/jobs` - List all jobs
  - Supports filtering by type and status
  - Includes pagination

### Model Parameter Management

- `GET /api/v1/model-params` - List model parameters
- `POST /api/v1/model-params` - Create new parameter set
- `PUT /api/v1/model-params/{id}/activate` - Activate parameter set

### Admin & Health Monitoring

- `GET /health` - API health check
- `GET /health/system` - System resource statistics
- `GET /health/retry-stats` - Operation retry statistics

### General

- `GET /` - API information
- `GET /docs` - API documentation (SwaggerUI)

## Data Flow

1. **Data Upload**:
   - Upload Excel (.xlsx, .xls) or CSV files with stock and sales data
   - Automatic file format detection and encoding handling (UTF-8, Windows-1251, CP1252)
   - Automatic CSV separator detection (comma, semicolon)
   - Data is validated, processed, and stored in SQLite database
   - Records are mapped to multi-index structure for time series analysis

2. **Training & Prediction** (Combined process):
   - Data is prepared and uploaded directly to Yandex DataSphere project storage
   - Training job is submitted to Yandex DataSphere
   - TiDE model is trained with two-phase approach:
     1. First training with early stopping to determine optimal epochs
     2. Final training on full dataset with optimized parameters
   - Predictions are generated immediately after training
   - Both model and predictions are downloaded to the local server
   - Model file is stored locally and metadata in the database
   - Predictions are stored in the database for later reporting
   - No separate prediction API call is needed
   - DataSphere project storage automatically cleans up temporary files

3. **Reporting**:
   - Uses stored prediction data to generate HTML reports
   - Multiple report types available based on data views
   - Reports are stored as files for later access

## Database Schema

The application uses SQLite with a star schema design:

### Dimension Tables
- `dim_multiindex_mapping` - Central dimension table for vinyl records

### Fact Tables
- `fact_sales` - Historical sales records
- `fact_stock` - Inventory snapshots
- `fact_prices` - Historical pricing data
- `fact_stock_changes` - Inventory movement records
- `fact_predictions` - Generated sales predictions

### Job & Model Tables
- `jobs` - Tracks all asynchronous operations
- `models` - Stores trained model metadata and local file paths
- `parameter_sets` - Model configuration parameters
- Various result tables for different job types

## Directory Structure

```
deployment/
├── app/
│   ├── api/                  # API routes and request/response models
│   │   ├── admin.py          # Admin endpoints (system info, cleanup)
│   │   ├── health.py         # Health check endpoints
│   │   ├── jobs.py           # Job management endpoints
│   │   └── models_configs.py # Model parameter management
│   ├── db/                   # Database schema, operations, and data retention logic
│   │   ├── database.py       # Core database operations
│   │   ├── schema.py         # Database table definitions
│   │   ├── data_retention.py # Automated data cleanup
│   │   └── feature_storage.py # Feature engineering data storage
│   ├── services/             # Business logic services
│   │   ├── auth.py           # Authentication service
│   │   ├── data_processor.py # Data processing and validation
│   │   ├── datasphere_service.py # Yandex DataSphere integration
│   │   └── report_service.py # Report generation service
│   ├── utils/                # Utility functions
│   │   ├── error_handling.py # Error handling and retry logic
│   │   ├── file_validation.py # File upload validation
│   │   ├── retry.py          # Retry mechanisms
│   │   └── validation.py     # Data validation utilities
│   ├── config.py             # Configuration management (Pydantic settings)
│   ├── logger_config.py      # Centralized logging configuration
│   └── main.py               # FastAPI application setup and main entry point
├── datasphere/
│   ├── client.py             # Yandex DataSphere client for API interactions
│   └── prepare_datasets.py   # Scripts for preparing datasets for DataSphere jobs
├── infrastructure/           # Terraform infrastructure as code
│   ├── modules/              # Reusable Terraform modules
│   │   ├── datasphere_community/ # DataSphere community setup
│   │   ├── datasphere_project/   # DataSphere project configuration
│   │   └── service_account/      # Service account and IAM setup
│   └── envs/
│       └── prod/             # Production environment configuration
├── scripts/                  # Deployment and utility scripts
│   └── check_environment.py  # Script to validate environment setup
└── run.py                    # Main runner script for the API application
```

**Data Storage Structure** (created at runtime based on `DATA_ROOT_DIR`):
```
~/.plastinka_sales_predictor/    # Default data root directory
├── database/
│   └── plastinka.db            # SQLite database file
├── models/                     # Storage for trained model artifacts (e.g., .onnx files)
├── datasphere_input/           # Prepared datasets for DataSphere jobs
├── datasphere_output/          # Downloaded results from DataSphere
├── predictions/                # Saved prediction outputs
├── reports/                    # Generated reports
├── logs/                       # Application logs
```

## Troubleshooting

### Common Issues

- **Database errors**: Check that SQLite is available and the data directory is writable
- **Missing dependencies**: Ensure all requirements are installed with `pip install -e ".[deployment]"`
- **File permission errors**: Check permissions on data directories (controlled by `DATA_ROOT_DIR`)
- **Cloud errors**: Verify your Yandex Cloud credentials and network connectivity
- **Missing environment variables**: Run the check_environment.py script to verify your setup

### Logs

Application logs are stored in `{DATA_ROOT_DIR}/logs/app.log` (default: `~/.plastinka_sales_predictor/logs/`)

Log levels can be configured via the `API_LOG_LEVEL` environment variable:
- `DEBUG` - Detailed debugging information
- `INFO` - General operational messages (default)
- `WARNING` - Warning messages
- `ERROR` - Error messages only

### Environment Validation

Run the environment checker to validate your setup:

```bash
python deployment/scripts/check_environment.py .env.template
```

This script will:
- Check Python version compatibility
- Verify required dependencies
- Validate environment variables
- Test database connectivity
- Check file system permissions
- Generate a `.env.template` file

### Performance Optimization

**Database Performance:**
- Regular database maintenance via `/api/v1/admin/cleanup`
- Monitor database size and query performance
- Use indexed queries for large datasets

**File Storage:**
- Configure appropriate `DATA_ROOT_DIR` location
- Monitor disk space usage
- Use data retention policies to manage storage

**API Performance:**
- Monitor job queue length
- Use appropriate timeout settings for DataSphere jobs
- Configure CORS origins for production deployment

### Security Considerations

**Authentication:**
- Set strong `CLOUD_CALLBACK_AUTH_TOKEN` for DataSphere callbacks
- Use HTTPS in production deployments
- Configure appropriate CORS origins

**Data Protection:**
- Secure file upload validation
- Database access controls
- Proper error message sanitization

**Cloud Security:**
- Rotate Yandex Cloud credentials regularly
- Use IAM roles with minimal required permissions
- Monitor cloud resource usage 