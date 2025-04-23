# Plastinka Sales Predictor API

A FastAPI application for predicting vinyl record sales using time series forecasting, with cloud-based computation via Yandex Cloud Functions.

## Overview

This API provides endpoints for:

1. **Data Upload** - Process and store raw sales and stock data
2. **Model Training** - Train time series forecasting models in the cloud
3. **Predictions** - Generate sales predictions using trained models
4. **Reports** - Generate various analysis reports

All operations are handled as asynchronous jobs with status tracking and cloud computation offloading.

## System Requirements

- Python 3.10+
- SQLite3
- 2GB+ RAM
- 1GB+ disk space
- Yandex Cloud account and credentials

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd plastinka_sales_predictor
```

### 2. Set Environment Variables

```bash
# Required Variables
export YANDEX_CLOUD_ACCESS_KEY="your-access-key"
export YANDEX_CLOUD_SECRET_KEY="your-secret-key"
export YANDEX_CLOUD_FOLDER_ID="your-folder-id"
export YANDEX_CLOUD_API_KEY="your-api-key"
export CLOUD_CALLBACK_AUTH_TOKEN="your-callback-token"

# Optional Variables
export YANDEX_CLOUD_BUCKET="plastinka-ml-data"
export YANDEX_CLOUD_STORAGE_ENDPOINT="https://storage.yandexcloud.net"
export YANDEX_CLOUD_REGION="ru-central1"
```

### 3. Run the Application

```bash
cd deployment
python run.py --reload
```

The API will be available at http://localhost:8000

## API Endpoints

### Job Management

- `POST /api/v1/jobs/data-upload` - Upload and process sales data
- `POST /api/v1/jobs/training` - Train a new model
- `POST /api/v1/jobs/prediction` - Generate predictions
- `POST /api/v1/jobs/reports` - Generate reports
- `GET /api/v1/jobs/{job_id}` - Get job status and results
- `GET /api/v1/jobs` - List all jobs

### General

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - API documentation (SwaggerUI)

## Data Flow with Cloud Integration

1. **Data Upload**:
   - Upload Excel files with stock and sales data
   - Data is processed and stored in SQLite database
   - Feature engineering for time series analysis

2. **Training** (Cloud-Based):
   - Data is prepared and uploaded to Yandex Cloud Storage
   - Training job is submitted to Yandex Cloud Functions
   - Model is trained in the cloud and stored in Cloud Storage
   - Training metrics are retrieved and stored in the database

3. **Prediction** (Cloud-Based):
   - Input data is uploaded to Yandex Cloud Storage
   - Prediction job is submitted to Yandex Cloud Functions
   - Predictions are generated in the cloud
   - Results are downloaded and stored in the database

4. **Reports**:
   - Uses stored data to generate HTML reports
   - Multiple report types available

## Data Storage

The application uses SQLite for data storage with the following structure:

- **Dimension tables** - Store reference data (artists, albums, etc.)
- **Fact tables** - Store measurements (stock, sales, prices)
- **Job tables** - Track job status and results

## Cloud Integration

This application integrates with Yandex Cloud to offload computation:

- **Cloud Functions** - Execute CPU-intensive model training and prediction tasks
- **Cloud Storage** - Store datasets, models, and results

### Deploying Cloud Functions

```bash
python deployment/scripts/deploy_cloud_functions.py --function-type all --folder-id YOUR_FOLDER_ID --service-account-id YOUR_SERVICE_ACCOUNT_ID
```

Options:
- `--concurrent` - Deploy functions in parallel
- `--dry-run` - Test deployment without actual execution
- `--verify` - Verify function after deployment
- `--rollback-on-failure` - Enable automatic rollback if deployment fails

## Directory Structure

```
deployment/
├── app/
│   ├── api/                  # API routes
│   ├── db/                   # Database utilities
│   ├── models/               # API models (Pydantic)
│   ├── services/             # Business logic
│   ├── cloud_integration/    # Yandex Cloud integration
│   ├── utils/                # Utility functions
│   └── main.py               # Application entry point
├── data/                     # Data storage (created at runtime)
│   ├── uploads/              # Temporary storage for uploads
│   ├── predictions/          # Saved predictions
│   └── reports/              # Generated reports
├── logs/                     # Application logs
├── scripts/                  # Deployment scripts
│   ├── deploy_cloud_functions.py  # Cloud function deployment
│   └── check_environment.py       # Environment validation
├── tests/                    # Test suite
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
└── run.py                    # Runner script
```

## Troubleshooting

### Common Issues

- **Database errors**: Check that SQLite is available and the `data` directory is writable
- **Missing dependencies**: Ensure all requirements are installed with `pip install -r requirements.txt`
- **File permission errors**: Check permissions on the `data` and `logs` directories
- **Cloud errors**: Verify your Yandex Cloud credentials and network connectivity
- **Missing environment variables**: Check that all required cloud variables are set

### Logs

Application logs are stored in `deployment/logs/app.log`

### Environment Validation

Run the environment checker to validate your setup:

```bash
python deployment/scripts/check_environment.py
``` 