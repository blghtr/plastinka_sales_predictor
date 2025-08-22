# Plastinka Sales Predictor

A comprehensive machine learning system for predicting vinyl record sales using advanced time-series forecasting with cloud-based deployment capabilities.

## 🎯 Overview

Plastinka Sales Predictor is a production-ready ML system designed to provide accurate vinyl sales forecasts. It combines:

- **🧠 Advanced ML Models**: Utilizing state-of-the-art time-series forecasting models (TiDE).
- **☁️ Cloud Integration**: Seamless integration with Yandex DataSphere for scalable model training and deployment.
- **🚀 Production API**: A robust FastAPI-based REST API for interacting with the forecasting system.
- **🏗️ Infrastructure as Code**: Automated infrastructure management with Terraform for consistent and reproducible deployments.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Plastinka Sales Predictor System                         │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   ML Module     │    │   FastAPI App   │    │  Infrastructure │          │
│  │ (ML Code)       │    │ (Orchestrator)  │    │ (IaC)           │          │
│  │ • TiDE Model    │    │ • REST API      │    │ • Terraform     │          │
│  │ • Data Prep     │    │ • Job Queue     │    │ • DataSphere    │          │
│  │ • Metrics       │    │ • Database      │    │ • Monitoring    │          │
│  │ • Training      │    │ • File Storage  │    │                 │          │
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

### Компоненты и их взаимодействие

-   **FastAPI App (`deployment/`)**: Центральный компонент-оркестратор. Он предоставляет REST API для взаимодействия с пользователем, управляет базой данных (метаданные, задачи, результаты) и инициирует задачи (тренировка, тюнинг) в Yandex DataSphere.
-   **ML Module (`plastinka_sales_predictor/`)**: Это не отдельный сервис, а Python-пакет, содержащий всю логику машинного обучения (подготовка данных, архитектура модели TiDE, метрики). Этот код упаковывается и выполняется непосредственно в облачной среде Yandex DataSphere.
-   **Infrastructure (`deployment/infrastructure/`)**: Инфраструктура как код (IaC) на базе Terraform. Эти конфигурации описывают и создают все необходимые облачные ресурсы в Yandex Cloud, включая проект DataSphere и права доступа. Это компонент этапа развертывания.

---

## 🚀 Getting Started

This guide provides the steps to set up the project from scratch.

### 1. Prerequisites

- **Python & `uv`**: Ensure you have Python 3.x installed. This project uses `uv` for package management. Install it with:
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Windows
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Terraform / OpenTofu**: Install either [Terraform](https://developer.hashicorp.com/terraform/install) or [OpenTofu](https://opentofu.org/docs/intro/install/) to manage cloud infrastructure.
- **Yandex Cloud Account**: You need a Yandex Cloud account with an organization and folder.

### 2. Set Up Infrastructure

Terraform automatically creates the cloud resources and a `.env` file in the project root with the necessary API keys.

```bash
# 1. Navigate to the infrastructure configuration directory
cd deployment/infrastructure/envs/prod

# 2. Create a variables file from the example
cp terraform.tfvars.example terraform.tfvars

# 3. Edit terraform.tfvars with your Yandex Cloud IDs
# (yc_cloud_id, yc_folder_id, yc_organization_id)

# 4. Set your Yandex Cloud OAuth token as an environment variable
export TF_VAR_yc_token="your-yc-oauth-token"

# 5. Initialize and apply the Terraform configuration
terraform init
terraform apply
```
This will create the necessary DataSphere resources and populate the `.env` file at the project root.

### 3. Install Dependencies

Once the infrastructure is ready, install the Python dependencies.

```bash
# From the project root
uv sync
```

### 4. Run the Application

Start the FastAPI server:

```bash
python deployment/run.py
```
The API will be available at `http://127.0.0.1:8000`, and the documentation can be found at `http://127.0.0.1:8000/docs`.

---

## 🔄 Core Workflow

The system is designed around a monthly operational cycle. This workflow is the primary use case for the API.

1.  **Upload Data**: At the beginning of a new month, upload the sales data for the past month and the current stock levels.
    - `POST /api/v1/jobs/data-upload`
2.  **Monitor Upload Job**: Track the status of the data processing job.
    - `GET /api/v1/jobs/{job_id}`
3.  **Check System Health (Optional but Recommended)**: Verify that the data is consistent and the system is ready for training.
    - `GET /health`
4.  **Trigger Model Training**: Start a new training job in Yandex DataSphere. This process trains the model on the complete, updated dataset and generates predictions for the next period.
    - `POST /api/v1/jobs/training`
5.  **Monitor Training Job**: Track the training process, which can take a significant amount of time (e.g., ~2 hours).
    - `GET /api/v1/jobs/{job_id}`
6.  **Trigger Hyperparameter Tuning (If Needed)**: If the model performance degrades over time (indicated by the `/health` endpoint), run a tuning job to find better hyperparameters. After tuning, you must re-run the training job (Step 4).
    - `POST /api/v1/jobs/tuning`
7.  **Retrieve Report**: Once training is complete, fetch the prediction report.
    - `POST /api/v1/jobs/reports`

---

## 📚 Documentation

This project is composed of several key components, each with its own detailed documentation.

- **[Deployment README](deployment/README.md)**: **(START HERE FOR API USAGE)** Detailed guide on the FastAPI application, including API endpoints, data requirements, business logic, and practical examples.
- **[Infrastructure README](deployment/infrastructure/README.md)**: Comprehensive guide on setting up and managing the cloud infrastructure with Terraform.
- **ML Module (`plastinka_sales_predictor/`)**: The core Python package containing the ML code (model, data processing, etc.). The code is the primary documentation.

## 🧪 Testing

To run the test suite:
```bash
pytest
```

## 📄 License

This project is licensed under the [CC BY-NC-SA 4.0 License](LICENSE).