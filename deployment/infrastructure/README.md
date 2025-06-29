# Terraform Infrastructure for Plastinka Sales Predictor

Эта директория содержит декларативное описание ресурсов Yandex Cloud, необходимых для работы DataSphere-проекта с автоматическим созданием и настройкой service account.

## Структура

```
modules/                # Переиспользуемые модули
  datasphere_community/ # YC DataSphere Community
  datasphere_project/   # YC DataSphere Project
  service_account/      # YC Service Account с IAM ролями

envs/
  prod/                 # Конфигурация для prod (единственное окружение)
    main.tf
    variables.tf
    terraform.tfvars
    terraform.tfvars.example
    config.yaml.tpl     # Шаблон конфигурации DataSphere

versions.tf             # Глобальные ограничения версий Terraform и провайдеров
```

## Что создаётся

1. **DataSphere Service Account** - для работы с DataSphere:
   - `datasphere.user` - базовый доступ к DataSphere
   - `datasphere.communities.developer` - разработка и запуск проектов
   - `storage.admin` - доступ к Object Storage
   - `compute.admin` - управление вычислительными ресурсами
   - `vpc.user` - доступ к сетевым ресурсам

2. **DataSphere Community** - организационная единица для проектов

3. **DataSphere Project** - привязанный к service account проект

4. **DataSphere Configuration** - автоматически генерируемый `config_standard.yaml`

## Быстрый старт

1. Установите Terraform >= 1.6.
2. Перейдите в каталог окружения prod:
   ```powershell
   cd deployment\infrastructure\envs\prod
   ```
3. Скопируйте `terraform.tfvars.example` в `terraform.tfvars` и заполните значения.
4. Инициализируйте рабочую директорию:
   ```powershell
   terraform init
   ```
5. Посмотрите план изменений:
   ```powershell
   terraform plan
   ```
6. Примените изменения:
   ```powershell
   terraform apply
   ```

## Получение данных после применения

После применения конфигурации вы получите:

```powershell
# DataSphere Service Account данные
terraform output service_account_id
terraform output service_account_name
terraform output -raw static_access_key_id
terraform output -raw static_secret_key

# DataSphere проект и конфигурация
terraform output datasphere_project_id
terraform output datasphere_config_path
terraform output datasphere_summary
```

## Использование Service Account

### В коде приложения
Используйте полученные ключи для настройки DataSphere клиента:

```python
from deployment.datasphere.client import DataSphereClient

client = DataSphereClient(
    service_account_key_id="<static_access_key_id>",
    service_account_key="<static_secret_key>",
    folder_id="<folder_id>"
)
```

### В переменных окружения
```bash
export YC_SERVICE_ACCOUNT_KEY_ID="<static_access_key_id>"
export YC_SERVICE_ACCOUNT_KEY="<static_secret_key>"
export YC_FOLDER_ID="<folder_id>"
```

## DataSphere Job Configuration

Terraform автоматически генерирует конфигурацию DataSphere Job в файл `config_standard.yaml` со следующими возможностями:

- **Python 3.8** окружение
- **Автоматическая установка зависимостей** из requirements списка
- **Настроенные переменные окружения**
- **Готовая команда выполнения**

Конфигурация включает все необходимые Python пакеты:
- pandas, numpy, scikit-learn
- fastapi, uvicorn
- onnx, onnxruntime
- matplotlib, seaborn, plotly
- И другие

## Безопасность

* `prevent_destroy = true` установлен для проекта, чтобы случайный `destroy` не стёр историю экспериментов.
* OAuth-токен и ключи не должны попадать в VCS. Используйте переменные окружения или secrets-менеджер.
* Static keys помечены как `sensitive` в outputs - используйте `-raw` флаг для получения значений.
* Service Account создаётся с минимально необходимыми правами для работы DataSphere.

## State Management

State хранится локально (файл `terraform.tfstate` в каталоге). При необходимости можно настроить удалённый backend.

## Расширение

* Для дополнительных окружений создайте каталог `envs/<env_name>` с собственными параметрами.
* Добавьте модуль `artifacts_bucket`, если потребуется управлять S3-хранилищем для моделей.
* Настройте дополнительные роли в `folder_roles` или `cloud_roles` по необходимости.

## Запуск DataSphere Job

После применения Terraform:

1. Загрузите сгенерированный `config_standard.yaml` в DataSphere проект
2. Подготовьте входные данные в формате `input.zip`
3. Запустите Job через DataSphere UI или API
4. Получите результаты в `output.zip` 