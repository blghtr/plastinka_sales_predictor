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

### 4. Автоматическая генерация `.env` файла и API ключей
При первом применении Terraform автоматически создаст или обновит файл `.env` в корне проекта, а также сгенерирует и добавит необходимые API ключи для взаимодействия с FastAPI приложением.

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

## 🔄 Использование существующих ресурсов

**Новая возможность!** Теперь можно подключить существующую инфраструктуру DataSphere к управлению Terraform без её пересоздания.

### Когда использовать
- ✅ У вас уже есть настроенные DataSphere ресурсы
- ✅ Нужно мигрировать проект под управление Terraform
- ✅ Хотите работать в смешанном режиме (часть существующая, часть новая)

### Быстрое подключение существующих ресурсов

1. **Получите ID существующих ресурсов:**
```bash
# Service Account
yc iam service-account list --format json | jq -r '.[] | select(.name=="datasphere-sa-prod") | .id'

# Community  
yc datasphere community list --format json | jq -r '.[] | select(.name=="prod-ds-community") | .id'

# Project
yc datasphere project list --community-id YOUR_COMMUNITY_ID --format json | jq -r '.[] | select(.name=="prod-ds-project") | .id'
```

2. **Добавьте в terraform.tfvars:**
```hcl
# Основные переменные
yc_cloud_id        = "your-cloud-id"
yc_folder_id       = "your-folder-id"
yc_organization_id = "your-org-id"

# Использование существующих ресурсов
existing_service_account_id = "your-existing-service-account-id"
existing_community_id       = "your-existing-community-id"
existing_project_id         = "your-existing-project-id"
# existing_static_key_id - если у вас уже есть статический ключ и вы хотите его использовать
```

3. **Примените конфигурацию:**
```bash
terraform init
terraform plan  # Проверьте - новые ресурсы НЕ создаются!
terraform apply
```

### Как это работает?

Мы используем **чистый Data Source подход** вместо импорта:
- ✅ **Простота** - нет необходимости в сложных импортах
- ✅ **Надежность** - стандартные возможности Terraform
- ✅ **Гибкость** - можно комбинировать новые и существующие ресурсы
- ✅ **Безопасность** - существующие ресурсы не модифицируются
- ✅ **Совместимость** - работает с любой версией Terraform

### Подробная документация
📖 **[Полное руководство](envs/prod/IMPORT_GUIDE.md)** - детальные инструкции, сценарии использования, устранение проблем.

### Проверка статуса ресурсов
```bash
# Посмотреть какие ресурсы существующие, а какие новые
terraform output import_status

# Пример вывода:
# {
#   "community": "existing",
#   "project": "created", 
#   "service_account": "existing",
#   "static_key": "existing"
# }
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

**Рекомендуемый способ - автоматически сгенерировать .env файл:**
Terraform автоматически создаст или обновит файл `.env` в корне проекта с необходимыми переменными.

**Альтернативно - через переменные окружения (для ручной настройки или CI/CD):**
```bash
# Bash - установка обязательных переменных
export DATASPHERE_PROJECT_ID=$(terraform output -raw datasphere_project_id)
export DATASPHERE_FOLDER_ID="your-folder-id" # Замените на ваш folder_id
export DATASPHERE_YC_PROFILE="datasphere-prod"
export API_ADMIN_API_KEY="your-admin-api-key" # Будет сгенерирован автоматически при первом запуске Terraform
export API_X_API_KEY="your-x-api-key" # Будет сгенерирован автоматически при первом запуске Terraform
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
- Не храните `terraform.tfvars` в системе контроля версий.
- Используйте переменные окружения для чувствительных данных, таких как `TF_VAR_yc_token`.
- Регулярно проверяйте и обновляйте версии Terraform и провайдеров.
- Ограничивайте доступ к файлам с чувствительными данными.