# Plastinka Sales Predictor API

A FastAPI application for predicting vinyl record sales using time series forecasting, with cloud-based computation via Yandex DataSphere.

## Overview

This API provides endpoints for:

1. **Data Upload** - Process and store raw sales and stock data
2. **Model Training & Prediction** - Train TiDE models and generate predictions in one step
3. **Reports** - Generate analysis reports with visualizations

All operations are handled as asynchronous jobs with status tracking and cloud computation offloading.

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

```bash
# Required Variables
export YANDEX_CLOUD_ACCESS_KEY="your-access-key"
export YANDEX_CLOUD_SECRET_KEY="your-secret-key"
export YANDEX_CLOUD_FOLDER_ID="your-folder-id"
export YANDEX_CLOUD_API_KEY="your-api-key"
export CLOUD_CALLBACK_AUTH_TOKEN="your-callback-token"

# Optional Variables (defaults shown)
export YANDEX_CLOUD_BUCKET="plastinka-ml-data"
export YANDEX_CLOUD_STORAGE_ENDPOINT="https://storage.yandexcloud.net"
export YANDEX_CLOUD_REGION="ru-central1"
export MAX_UPLOAD_SIZE="52428800"  # 50MB
export APP_ENV="development"  # Options: development, testing, production
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
  - Accepts Excel files for stock and sales data
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
   - Upload Excel files with stock and sales data
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
- `dim_price_categories` - Price range categories
- `dim_styles` - Music style hierarchy

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
│   ├── api/                  # API routes for different endpoints
│   ├── db/                   # Database schema and operations
│   ├── models/               # API data models (Pydantic)
│   ├── services/             # Business logic services
│   ├── utils/                # Utility functions
│   ├── config.py             # Configuration management
│   └── main.py               # FastAPI application setup
├── datasphere/
│   ├── client.py             # Yandex DataSphere client
│   ├── prepare_datasets.py   # Dataset preparation for ML
│   └── datasphere_job/       # Code executed in DataSphere
│       ├── train_and_predict.py  # Model training and prediction
│       └── requirements.txt      # Dependencies for cloud execution
├── data/                     # Data storage (created at runtime)
│   ├── uploads/              # Temporary storage for uploads
│   ├── predictions/          # Saved predictions
│   └── reports/              # Generated reports
├── infrastructure/           # Terraform IaC for cloud resources
├── logs/                     # Application logs
├── scripts/                  # Deployment and utility scripts
│   └── check_environment.py  # Environment validation
└── run.py                    # Runner script for the API
```

## Troubleshooting

### Common Issues

- **Database errors**: Check that SQLite is available and the `data` directory is writable
- **Missing dependencies**: Ensure all requirements are installed with `pip install -r requirements.txt`
- **File permission errors**: Check permissions on the `data` and `logs` directories
- **Cloud errors**: Verify your Yandex Cloud credentials and network connectivity
- **Missing environment variables**: Run the check_environment.py script to verify your setup

### Logs

Application logs are stored in `deployment/logs/app.log`

### Environment Validation

Run the environment checker to validate your setup:

```bash
python deployment/scripts/check_environment.py
``` 