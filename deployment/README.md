# Plastinka Sales Predictor API

A FastAPI application for predicting vinyl record sales using time series forecasting.

## Overview

This API provides endpoints for:

1. **Data Upload** - Process and store raw sales and stock data
2. **Model Training** - Train time series forecasting models
3. **Predictions** - Generate sales predictions using trained models
4. **Reports** - Generate various analysis reports

All operations are handled as asynchronous jobs with status tracking.

## System Requirements

- Python 3.10+
- SQLite3
- 2GB+ RAM
- 1GB+ disk space

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd plastinka_sales_predictor
```

### 2. Run the Application

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

## Data Flow

1. **Data Upload**:
   - Upload Excel files with stock and sales data
   - Data is processed and stored in SQLite database
   - Feature engineering for time series analysis

2. **Training**:
   - Model is trained using the processed dataset
   - Training metrics are stored
   - Model is saved for future prediction

3. **Prediction**:
   - Uses stored features and trained model
   - Generates sales forecast
   - Saves results to database

4. **Reports**:
   - Uses stored data to generate HTML reports
   - Multiple report types available

## Data Storage

The application uses SQLite for data storage with the following structure:

- **Dimension tables** - Store reference data (artists, albums, etc.)
- **Fact tables** - Store measurements (stock, sales, prices)
- **Job tables** - Track job status and results

## Directory Structure

```
deployment/
├── app/
│   ├── api/             # API routes
│   ├── db/              # Database utilities
│   ├── models/          # API models (Pydantic)
│   ├── services/        # Business logic
│   └── main.py          # Application entry point
├── data/                # Data storage (created at runtime)
│   ├── uploads/         # Temporary storage for uploads
│   ├── predictions/     # Saved predictions
│   └── reports/         # Generated reports
├── logs/                # Application logs
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
└── run.py               # Runner script
```

## Troubleshooting

### Common Issues

- **Database errors**: Check that SQLite is available and the `data` directory is writable
- **Missing dependencies**: Ensure all requirements are installed with `pip install -r requirements.txt`
- **File permission errors**: Check permissions on the `data` and `logs` directories

### Logs

Application logs are stored in `deployment/logs/app.log` 