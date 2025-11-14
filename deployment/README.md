# Plastinka Sales Predictor API

A FastAPI application for predicting vinyl record sales using time series forecasting, with cloud-based computation via Yandex DataSphere.

## 📚 High-Level Documentation

This document focuses specifically on the API application. For a high-level overview of the entire project, the system architecture, and a getting started guide, please refer to the **[main project README](../README.md)**.

---

## ⚙️ API Usage and Business Logic

This section describes the core principles and workflow for interacting with the API.

### Data Requirements

Correct and efficient system operation depends on the quality of the input data.

-   **File Format**: All uploaded files must be in `.xlsx` or `.csv` format. The system automatically handles various encodings (UTF-8, Windows-1251) and CSV separators (comma, semicolon).
-   **`stock_file`**: This file must contain current stock levels. **Crucially**, it must be generated for the date immediately following the last transaction date in the `sales_files`. For example, if the last sale occurred on March 25, 2025, the stock file must reflect the state at the beginning of March 26, 2025.
-   **`sales_files`**: These files should contain all transactions that the business considers a sale and that affect stock levels. The file order during upload does not matter.

### The Three Laws of Data Management

To ensure data integrity, the system operates on three core principles:

1.  **The Law of Global Monotonicity**: The database must contain a complete, unbroken sequence of monthly data. There should be no "gaps" between the first and last month of data. The `GET /health` endpoint helps diagnose violations of this law.
2.  **The Law of Local Monotonicity**: Data uploaded in a single `POST /api/v1/jobs/data-upload` request must also be monotonic, with months following each other without gaps.
3.  **The Law of Snapshot Coherence**: The `stock_file` must always represent the stock state *after* all transactions from the accompanying `sales_files` have occurred.

### Standard Monthly Workflow (API Interaction)

This is the primary sequence of API calls for the monthly forecasting cycle.

1.  **Upload Data (`POST /api/v1/jobs/data-upload`)**
    -   Send `stock_file` and one or more `sales_files` as `multipart/form-data`.
    -   The call returns a `job_id`.

2.  **Monitor Data Processing (`GET /api/v1/jobs/{job_id}`)**
    -   Poll this endpoint until the job `status` is `completed`. This usually takes about 10 seconds. If it's `failed`, check the `error` field.

3.  **Check System Health (`GET /health`)**
    -   After data upload, call this endpoint.
    -   Ensure `database` status is not `unhealthy` (which would indicate a violation of Global Monotonicity).
    -   Check `active_model_metric` status. If `degraded`, consider running a tuning job.

4.  **Run Training (`POST /api/v1/jobs/training`)**
    -   This initiates a training and prediction job in Yandex DataSphere. It's a long-running process (~2 hours).
    -   The call returns a new `job_id` for the training task.

5.  **Monitor Training (`GET /api/v1/jobs/{job_id}`)**
    -   Poll this endpoint to track the training progress.

6.  **Run Tuning (Optional, `POST /api/v1/jobs/tuning`)**
    -   If model metrics are degraded, run this job to find better hyperparameters.
    -   This is a very long-running process (1-5+ hours).
    -   After successful tuning, you **must** run a new training job (Step 4) to apply the new parameters.

7.  **Get Report (`POST /api/v1/jobs/reports`)**
    -   Once training is complete, you can request a forecast report.
    -   Specify `report_type` and `prediction_month` in the JSON body.
    -   The API returns the report data directly in the response.

---

## Directory Structure

```
deployment/
├── app/
│   ├── api/                  # API routes and request/response models
│   ├── db/                   # Database schema, operations, and data retention
│   ├── services/             # Business logic services
│   └── utils/                # Utility functions
├── datasphere/
│   ├── client.py             # Yandex DataSphere client
│   └── prepare_datasets.py   # Scripts for preparing datasets for jobs
├── infrastructure/           # Terraform IaC
└── run.py                    # Main runner script for the API application
```

## Data Storage

The application uses PostgreSQL database for data storage. Trained models, logs, and other artifacts are stored in a local data storage directory (default: `~/.plastinka_sales_predictor/`).
