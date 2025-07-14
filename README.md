# Plastinka Sales Predictor

A comprehensive machine learning system for predicting vinyl record sales using advanced time-series forecasting with cloud-based deployment capabilities.

## 🎯 Overview

Plastinka Sales Predictor is a production-ready ML system designed to provide accurate vinyl record sales forecasts. It combines:

- **🧠 Advanced ML Models**: Utilizing state-of-the-art time-series forecasting models.
- **☁️ Cloud Integration**: Seamless integration with cloud platforms for scalable model training and deployment.
- **🚀 Production API**: A robust FastAPI-based REST API for interacting with the forecasting system.
- **🏗️ Infrastructure as Code**: Automated infrastructure management for consistent and reproducible deployments.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Plastinka Sales Predictor System                         │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   ML Module     │    │   FastAPI App   │    │  Infrastructure │          │
│  │                 │    │                 │    │                 │          │
│  │ • TiDE Model    │    │ • REST API      │    │ • Terraform     │          │
│  │ • Data Prep     │    │ • Job Queue     │    │ • DataSphere    │          │
│  │ • Metrics       │    │ • Database      │    │ • Service Acc   │          │
│  │ • Training      │    │ • File Storage  │    │ • Monitoring    │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│           │                       │                       │                 │
│           └───────────────────────┼───────────────────────┘                 │
│                                   │                                         │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   Yandex DataSphere   │
                        │                       │
                        │ • ML Compute          │
                        │ • Model Training      │
                        │ • Prediction Gen      │
                        │ • Resource Scaling    │
                        └───────────────────────┘
```

## 🔧 Key Components

### 1. ML Module (`plastinka_sales_predictor/`)
- **TiDE Architecture**: State-of-the-art time-series dense encoder for forecasting
- **Custom Metrics**: MIWS (Mean Interval Width Score) and MIC (Mean Interval Coverage) for specialized evaluation
- **Data Processing**: Advanced preprocessing for vinyl sales data with intermittent demand patterns
- **Quantile Regression**: Probabilistic forecasts with confidence intervals
- **Hyperparameter Tuning**: Ray Tune integration for automated optimization

### 2. FastAPI Application (`deployment/`)
- **Asynchronous API**: Non-blocking job processing with status tracking
- **Database Integration**: SQLite with star schema for efficient data storage
- **File Management**: Excel upload processing and model artifact storage
- **Cloud Integration**: Seamless DataSphere job submission and monitoring
- **Error Handling**: Comprehensive retry logic and failure recovery

### 3. Infrastructure (`deployment/infrastructure/`)
- **Terraform Modules**: Reusable infrastructure components
- **DataSphere Projects**: Automated ML compute environment setup
- **Service Accounts**: Proper IAM configuration for cloud access
- **Resource Management**: Cost-effective resource allocation and cleanup

## 🏗️ Project Structure

```
plastinka_sales_predictor/
├── plastinka_sales_predictor/          # Core ML module
│   ├── __init__.py                     # Module exports
│   ├── data_preparation.py             # Data processing & dataset creation
│   ├── model.py                        # TiDE model implementation
│   ├── metrics.py                      # Custom MIWS/MIC metrics
│   ├── training_utils.py               # Training orchestration
│   ├── tuning_utils.py                 # Hyperparameter optimization
│   └── datasphere_job/                 # DataSphere execution scripts
│       ├── train_and_predict.py        # Main training script
│       └── config.yaml                 # DataSphere job configuration
├── deployment/                         # FastAPI application
│   ├── app/                           # Application core
│   │   ├── api/                       # REST API endpoints
│   │   ├── db/                        # Database schema & operations
│   │   ├── services/                  # Business logic services
│   │   └── utils/                     # Utility functions
│   ├── datasphere/                    # DataSphere integration
│   ├── infrastructure/                # Terraform IaC
│   └── scripts/                       # Deployment utilities
├── tests/                             # Comprehensive test suite
```

## 📊 Key Features

### Advanced Metrics
- **MIWS (Mean Interval Width Score)**: Measures prediction interval efficiency
- **MIC (Mean Interval Coverage)**: Evaluates interval coverage accuracy
- **Zero-Inflation Handling**: Specialized metrics for intermittent demand patterns

### Model Architecture
- **TiDE (Time-series Dense Encoder)**: Cutting-edge architecture for time-series forecasting
- **Quantile Regression**: Full distribution predictions with uncertainty quantification
- **Multi-variate Features**: Integration of stock levels, pricing, and categorical features

### Production Features
- **Asynchronous Processing**: Non-blocking job execution with status tracking
- **Automatic Scaling**: Cloud-based compute with on-demand resource allocation
- **Data Retention**: Configurable cleanup policies for production data management
- **Monitoring**: Comprehensive logging and health monitoring


## 📚 Documentation

This project is composed of several key components, each with its own detailed documentation:

- **ML Module (`plastinka_sales_predictor/`)**: Contains the core machine learning logic, including data preparation, model implementation, metrics, and training utilities.
- **FastAPI Application (`deployment/`)**: Provides the API endpoints for data ingestion, model training, prediction, and reporting. For detailed information on API usage, data flow, and application-specific troubleshooting, refer to the [Deployment README](deployment/README.md).
- **Infrastructure (`deployment/infrastructure/`)**: Describes the Terraform configurations for managing cloud resources, including DataSphere projects and service accounts. For comprehensive guidance on setting up and managing the cloud infrastructure, refer to the [Infrastructure README](deployment/infrastructure/README.md).

## 📦 Installation

1.  **Install `uv`**:
    This project uses `uv` for Python package management. Follow the official instructions to install it:
    ```bash
    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Install Infrastructure Tools**:
    You will need either OpenTofu or Terraform to manage the cloud infrastructure.
    - **OpenTofu**: Follow the installation guide at [https://opentofu.org/docs/intro/install/](https://opentofu.org/docs/intro/install/).
    - **Terraform**: Follow the installation guide at [https://developer.hashicorp.com/terraform/install](https://developer.hashicorp.com/terraform/install).

3.  **Sync Project Dependencies**:
    Once `uv` is installed, create a virtual environment and sync the dependencies defined in `pyproject.toml`:
    ```bash
    uv sync
    ```
    ```

4.  **Set Up Infrastructure**:
    For detailed instructions on configuring and deploying the cloud infrastructure, please refer to the **[Infrastructure README](deployment/infrastructure/README.md)**.

## 🚀 Usage

For detailed instructions on using the core ML functionalities or the FastAPI application, please refer to the respective component documentation:

- **[Deployment README](deployment/README.md)**: For API setup and usage.
- **[Infrastructure README](deployment/infrastructure/README.md)**: For cloud infrastructure setup.

## 🧪 Testing

Comprehensive test suites are available for both the ML module and the FastAPI application. Refer to the individual component documentation for instructions on running tests.

## 📄 License

This project is licensed under the [MIT License](LICENSE).



