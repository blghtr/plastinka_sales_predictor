# Terraform Infrastructure for Plastinka Sales Predictor

Эта директория содержит декларативное описание ресурсов Yandex Cloud, необходимых для работы DataSphere-проекта с автоматическим созданием и настройкой service account.

## Структура

```
modules/                    # Переиспользуемые модули
  datasphere_community/     # YC DataSphere Community
  datasphere_project/       # YC DataSphere Project  
  service_account/          # YC Service Account с IAM ролями

envs/
  prod/                     # Конфигурация для prod (единственное окружение)
    main.tf                 # Основная конфигурация ресурсов
    variables.tf            # Определения переменных
    outputs.tf              # Выходные значения
    terraform.tfvars        # Значения переменных (не в VCS)
    terraform.tfvars.example # Пример файла переменных


versions.tf                 # Глобальные ограничения версий Terraform и провайдеров
```

## Что создаётся

### 1. DataSphere Service Account (`datasphere-sa-prod`)
Service account с необходимыми ролями для работы с DataSphere:
- `datasphere.user` - базовый доступ к DataSphere
- `datasphere.communities.developer` - разработка и запуск проектов
- `storage.admin` - доступ к Object Storage
- `compute.admin` - управление вычислительными ресурсами
- `vpc.user` - доступ к сетевым ресурсам

### 2. DataSphere Community (`prod-ds-community`)
Организационная единица для группировки проектов с метками:
- `env = "prod"`
- `project = "plastinka-sales-predictor"`

### 3. DataSphere Project (`prod-ds-project`)
Проект DataSphere с конфигурацией:
- Привязка к созданному service account
- Ограничения ресурсов: 20 единиц/час, 200 единиц/выполнение


### 4. DataSphere Job Configuration
Готовая конфигурация `config.yaml` с:
- Manual Python 3.10.13 окружение
- Зависимости из `requirements.txt`
- Настроенные пути и индексы PyPI
- Готовая команда выполнения

## Быстрый старт

### 1. Подготовка переменных
```powershell
cd deployment\infrastructure\envs\prod
cp terraform.tfvars.example terraform.tfvars
```

Отредактируйте `terraform.tfvars`:
```hcl
yc_token           = "your-oauth-token" (безопаснее: сохраните в переменную среды TF_yc_token)
yc_cloud_id        = "your-cloud-id"
yc_folder_id       = "your-folder-id"
yc_organization_id = "your-organization-id"
```

### 2. Инициализация и применение
```powershell
terraform init
terraform plan
terraform apply
```

## Получение данных после применения

### Основные выходные значения:
```powershell
# Сводная информация о DataSphere
terraform output datasphere_summary

# ID проекта DataSphere
terraform output datasphere_project_id

# Service Account данные
terraform output service_account_id
terraform output service_account_name

# Ключи доступа (sensitive)
terraform output -raw static_access_key_id
terraform output -raw static_secret_key


```

## Использование Service Account

### В коде приложения
```python
from deployment.datasphere.client import DataSphereClient

client = DataSphereClient(
    service_account_key_id="<static_access_key_id>",
    service_account_key="<static_secret_key>",
    folder_id="<folder_id>"
)
```

### В переменных окружения
```powershell
$env:YC_SERVICE_ACCOUNT_KEY_ID = terraform output -raw static_access_key_id
$env:YC_SERVICE_ACCOUNT_KEY = terraform output -raw static_secret_key
$env:YC_FOLDER_ID = "your-folder-id"
```

## DataSphere Job Configuration

Проект содержит готовую конфигурацию `config.yaml`:

**Ресурсы:**
- Compute instance: `c1.4`
- Входные данные: `plastinka_sales_predictor/datasphere_job/input.zip`
- Выходные данные: `output.zip`

**Python окружение:**
- Manual Python 3.10.13
- Зависимости из `plastinka_sales_predictor/requirements.txt`
- PyTorch index: `https://download.pytorch.org/whl/cu118`
- Локальные пути: `plastinka_sales_predictor`

**Команда выполнения:**
```bash
python -m plastinka_sales_predictor.datasphere_job.train_and_predict --input ${INPUT} --output ${OUTPUT}
```

## Безопасность

### Защита чувствительных данных
- ✅ `sensitive = true` для всех ключей в outputs
- ✅ `prevent_destroy = true` для проекта (защита от случайного удаления)
- ✅ OAuth-токен не сохраняется в state файле
- ✅ Service Account создаётся с минимально необходимыми правами

### Рекомендации
- Не храните `terraform.tfvars` в VCS
- Используйте `-raw` флаг для получения sensitive значений
- Регулярно ротируйте static keys
- Мониторьте использование ресурсов через метки

## State Management

- State хранится локально в файле `terraform.tfstate`
- Создаются автоматические backup файлы
- При необходимости можно настроить удалённый backend

## Требования

- **Terraform** >= 1.6.0
- **Yandex Cloud Provider** ~> 0.109
- Права доступа: `datasphere.user`, `resource-manager.editor`


## Запуск DataSphere Job

1. **Подготовка данных**: создайте архив с входными данными
2. **Загрузка конфигурации**: используйте готовый `config.yaml`
3. **Запуск**: через DataSphere UI или API
4. **Результаты**: получите `output.zip` с результатами предсказаний 