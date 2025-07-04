# Terraform Infrastructure for Plastinka Sales Predictor

Эта директория содержит декларативное описание ресурсов Yandex Cloud, необходимых для работы DataSphere-проекта с автоматическим созданием и настройкой service account.

> **Примечание**: Данная инфраструктура является частью комплексной системы Plastinka Sales Predictor. См. [основной README](../../README.md) для полного обзора системы.

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
- Автоматическая интеграция с FastAPI приложением

### 4. DataSphere Job Configuration
Готовая конфигурация `config.yaml` с:
- Manual Python 3.10.13 окружение
- Зависимости из `requirements.txt`
- Настроенные пути и индексы PyPI
- Готовая команда выполнения

## Быстрый старт

### 1. Подготовка переменных

```bash
cd deployment/infrastructure/envs/prod
cp terraform.tfvars.example terraform.tfvars
```

#### Настройка terraform.tfvars:
Отредактируйте `terraform.tfvars` (только эти переменные):
```hcl
yc_cloud_id        = "your-cloud-id-here"
yc_folder_id       = "your-folder-id-here"
yc_organization_id = "your-organization-id-here"
```

#### Настройка OAuth токена для Terraform:
OAuth токен НЕ должен быть в `terraform.tfvars`. Используйте один из способов:

**Способ 1 - Переменная окружения (рекомендуется):**
```bash
export TF_VAR_yc_token="your-oauth-token"
terraform apply
```

**Способ 2 - Передача напрямую:**
```bash
terraform apply -var="yc_token=your-oauth-token"
```

### 2. Инициализация и применение
```bash
terraform init
terraform plan
terraform apply
```

## Получение данных после применения

### Основные выходные значения:
```bash
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

### В переменных окружения для FastAPI приложения

#### Обязательные переменные для .env файла:

**Рекомендуемый способ - создать .env файл:**
```bash
# Создать .env файл с обязательными переменными
echo "DATASPHERE_PROJECT_ID=$(terraform output -raw datasphere_project_id)" >> ../.env
echo "DATASPHERE_FOLDER_ID=your-folder-id" >> ../.env
echo "DATASPHERE_YC_PROFILE=datasphere-prod" >> ../.env
echo "API_X_API_KEY=your-api-key" >> ../.env
```

**Альтернативно - через переменные окружения:**
```bash
# Bash - установка обязательных переменных
export DATASPHERE_PROJECT_ID=$(terraform output -raw datasphere_project_id)
export DATASPHERE_FOLDER_ID="your-folder-id"
export DATASPHERE_YC_PROFILE="datasphere-prod"
export API_X_API_KEY="your-api-key"
```

#### Опциональные переменные:
```bash
# Только если используется OAuth аутентификация (не рекомендуется для продакшена)
export DATASPHERE_OAUTH_TOKEN="your-oauth-token"
export DATASPHERE_AUTH_METHOD="oauth_token"
```


## Интеграция с основной системой

После создания инфраструктуры она автоматически интегрируется с FastAPI приложением:

1. **Автоматическое обнаружение**: API приложение использует переменные окружения для подключения
2. **Управление заданиями**: Задания отправляются через API endpoints (`/api/v1/jobs/training`)
3. **Мониторинг**: Статус заданий отслеживается через DataSphere API
4. **Результаты**: Обученные модели и предсказания автоматически загружаются в локальное хранилище

## Мониторинг и логирование

### DataSphere мониторинг
- Логи выполнения заданий доступны через DataSphere UI
- Метрики использования ресурсов автоматически собираются
- Уведомления о состоянии заданий через callback API

### Локальное логирование
```bash
# Просмотр логов FastAPI приложения
tail -f ~/.plastinka_sales_predictor/logs/app.log

# Фильтрация логов DataSphere
grep "DataSphere" ~/.plastinka_sales_predictor/logs/app.log
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
- Используйте переменные окружения вместо хардкода в конфигурации

## State Management

- State хранится локально в файле `terraform.tfstate`
- Создаются автоматические backup файлы
- При необходимости можно настроить удалённый backend
- **Важно**: Не добавляйте state файлы в VCS (уже включены в .gitignore)

## Требования

- **Terraform** >= 1.6.0
- **Yandex Cloud Provider** ~> 0.109
- Права доступа: `datasphere.user`, `resource-manager.editor`
- **Yandex Cloud CLI** (опционально, для упрощения аутентификации)

## Устранение неполадок

### Частые проблемы

**Ошибки аутентификации:**
```bash
# Проверить валидность токена

yc config profile activate default
```

**Недостаточные права:**
- Убедитесь, что у пользователя есть роль `resource-manager.editor`
- Проверьте, что организация указана корректно

**Конфликты ресурсов:**
```bash
# Проверить существующие ресурсы
terraform state list
terraform show
```

### Очистка ресурсов

**Удаление всей инфраструктуры:**
```bash
terraform destroy
```

**Частичная очистка:**
```bash
# Удалить только проект DataSphere
terraform destroy -target=module.datasphere_project
```

## Запуск DataSphere Job

1. **Подготовка данных**: FastAPI приложение автоматически создаёт архив с входными данными
2. **Загрузка конфигурации**: используется готовый `config.yaml` из проекта
3. **Запуск**: через API endpoint `/api/v1/jobs/training`
4. **Результаты**: автоматическая загрузка `output.zip` с результатами предсказаний в локальное хранилище

Подробнее см. [API документацию](../README.md#api-endpoints). 