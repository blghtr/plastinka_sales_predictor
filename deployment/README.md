# Plastinka Sales Predictor API

A FastAPI application for predicting vinyl record sales using time series forecasting, with cloud-based computation via Yandex DataSphere.

## 📚 Documentation

This API is a core component of the Plastinka Sales Predictor system. For a high-level overview of the entire project, including its ML module and infrastructure, please refer to the [main project README](../README.md).

This document focuses specifically on the API application, its usage, and internal architecture.

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



## 📚 Documentation

This API is a core component of the Plastinka Sales Predictor system. For a high-level overview of the entire project, including its ML module and infrastructure, please refer to the [main project README](../README.md).

This document focuses specifically on the API application, its usage, and internal architecture.

#

## Troubleshooting

For general troubleshooting steps, common issues, and logging information, please refer to the [main project README](../README.md).