# Plastinka Sales Predictor - Architecture Document

## System Overview

The Plastinka Sales Predictor architecture implements a hybrid model that separates the API layer from the computational-intensive ML tasks. The system uses FastAPI for client interactions and Yandex Cloud Functions for the resource-intensive machine learning operations.

## Architecture Diagram

```
+-------------------+        +-------------------------+        +------------------------+
|                   |        |                         |        |                        |
|   Client Apps     | <----> |  FastAPI Application    | <----> |  Yandex Cloud Storage  |
|                   |        |                         |        |                        |
+-------------------+        +-------------------------+        +------------------------+
                                       ^    ^                              ^
                                       |    |                              |
                                       v    v                              v
                +----------------------+    +----------------------+    +------------------------+
                |                      |    |                      |    |                        |
                | Yandex Cloud Function|    | Yandex Cloud Function|    |  Database (SQLite)     |
                |   (Training)         |    |   (Prediction)       |    |                        |
                |                      |    |                      |    |                        |
                +----------------------+    +----------------------+    +------------------------+
```

## Component Description

### 1. Client Applications
- External applications or services that consume the Plastinka Sales Predictor API
- Sends requests for training models, making predictions, and retrieving results
- Interacts exclusively with the FastAPI application

### 2. FastAPI Application
- Serves as the central interface for all client interactions
- Handles authentication, request validation, and response formatting
- Manages communication with cloud functions and database
- Implements asynchronous job processing for long-running operations
- Provides status tracking and result retrieval for cloud function executions
- Endpoints:
  - `/api/v1/models` - Model management
  - `/api/v1/training` - Training job management 
  - `/api/v1/prediction` - Prediction job management
  - `/api/v1/status` - Job status monitoring
  - `/api/v1/results` - Result retrieval

### 3. Yandex Cloud Functions
- Executes computation-intensive ML tasks in a serverless environment
- Scales automatically based on demand
- Isolated from the API service for better resource utilization
- Two primary function types:
  - **Training Function**: Handles model training and hyperparameter tuning
  - **Prediction Function**: Performs forecasting based on trained models

### 4. Yandex Cloud Storage
- Stores large datasets, trained models, and prediction results
- Serves as the data exchange medium between the API and cloud functions
- Provides secure, temporary URLs for data access

### 5. Database (SQLite)
- Stores metadata about jobs, models, and results
- Tracks status of cloud function executions
- Maintains references to objects in cloud storage
- Schemas include:
  - `models` - Trained model metadata
  - `training_jobs` - Training job status and parameters
  - `prediction_jobs` - Prediction job status and parameters
  - `cloud_function_logs` - Detailed logs of cloud function executions

## Data Flow

### Training Flow
1. Client submits a training request with parameters and dataset reference
2. FastAPI validates the request and creates a training job record in the database
3. FastAPI uploads the dataset to cloud storage if not already present
4. FastAPI initiates the training cloud function with job parameters and data references
5. Cloud function retrieves the dataset from cloud storage
6. Cloud function executes training and uploads the model to cloud storage
7. Cloud function updates job status throughout the process via API callbacks
8. FastAPI updates the database with job status and results
9. Client can poll or receive notifications about job status
10. When complete, client can access the trained model or results

### Prediction Flow
1. Client submits a prediction request with parameters and model reference
2. FastAPI validates the request and creates a prediction job record in the database
3. FastAPI initiates the prediction cloud function with job parameters and model reference
4. Cloud function retrieves the model from cloud storage
5. Cloud function executes prediction and uploads results to cloud storage
6. Cloud function updates job status throughout the process via API callbacks
7. FastAPI updates the database with job status and results
8. Client can poll or receive notifications about job status
9. When complete, client can access the prediction results

## Error Handling and Recovery

1. **Job Monitoring**:
   - All cloud function executions are tracked with detailed status updates
   - Any failures are logged with error details and stack traces

2. **Retry Mechanism**:
   - Transient failures in cloud functions are automatically retried with exponential backoff
   - API implements a dead-letter queue for jobs that repeatedly fail

3. **Data Consistency**:
   - All data transfers between components use checksums to verify integrity
   - Cloud storage operations use atomic patterns to prevent partial updates

4. **Resource Management**:
   - Cloud functions implement timeouts and progress tracking
   - Large datasets/models are processed in chunks to avoid memory issues

## Security Considerations

1. **Authentication**:
   - All API endpoints require authentication
   - Cloud function invocations use secure API keys

2. **Data Protection**:
   - Sensitive data in cloud storage uses encryption at rest
   - Cloud function communication uses encrypted channels
   - Temporary URLs for data access have short expiration times

3. **Access Control**:
   - API implements role-based access control
   - Cloud functions have minimal permission sets

## Scalability

1. **Horizontal Scaling**:
   - FastAPI application can be deployed across multiple instances
   - Cloud functions automatically scale based on demand

2. **Performance Optimization**:
   - Large data transfers use cloud storage as intermediary
   - Database queries are optimized with appropriate indexes
   - Long-running operations are executed asynchronously

## Monitoring and Logging

1. **Application Metrics**:
   - API request volumes, response times, and error rates
   - Database query performance and connection pool utilization

2. **Cloud Function Metrics**:
   - Execution times, memory usage, and error rates
   - Throttling events and cold start frequencies

3. **Business Metrics**:
   - Model training times and quality metrics
   - Prediction accuracy and processing volumes

## Deployment Topology

The system is designed for deployment in the following configuration:

1. **FastAPI Application**:
   - Deployed as a containerized application
   - Can be scaled horizontally based on demand

2. **Yandex Cloud Functions**:
   - Deployed as serverless functions in Yandex Cloud
   - Automatically scaled by the cloud platform

3. **Database**:
   - Initially deployed as SQLite for simplicity
   - Can be migrated to a managed database service for production

4. **Cloud Storage**:
   - Uses Yandex Object Storage service
   - Organized with separate buckets for datasets, models, and results 