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

Эта Terraform конфигурация разворачивает необходимую инфраструктуру Yandex Cloud для работы Plastinka Sales Predictor, включая:

- **DataSphere Service Account**: Сервисный аккаунт с минимально необходимыми ролями для работы с DataSphere, Object Storage и другими облачными ресурсами.
- **DataSphere Community**: Организационная единица для группировки проектов.
- **DataSphere Project**: Проект DataSphere с настроенными лимитами ресурсов и привязкой к сервисному аккаунту.
- **Автоматическая генерация `.env` файла и API ключей**: При первом применении Terraform автоматически создаст или обновит файл `.env` в корне проекта, а также сгенерирует и добавит необходимые API ключи для взаимодействия с FastAPI приложением.

Для более подробного описания компонентов системы, включая ML модуль и FastAPI приложение, обратитесь к [основному README](../../README.md).

## Быстрый старт

Для быстрого развертывания инфраструктуры выполните следующие шаги:

### 1. Подготовка переменных

Перейдите в директорию `deployment/infrastructure/envs/prod` и скопируйте файл `terraform.tfvars.example` в `terraform.tfvars`:

```bash
cd deployment/infrastructure/envs/prod
cp terraform.tfvars.example terraform.tfvars
```

Отредактируйте `terraform.tfvars`, указав ваши `yc_cloud_id`, `yc_folder_id` и `yc_organization_id`.

Для аутентификации Terraform используйте переменную окружения `TF_VAR_yc_token`:

```bash
export TF_VAR_yc_token="your-oauth-token"
terraform apply
```

### 2. Инициализация и применение

```bash
terraform init
terraform plan
terraform apply
```

Эта команда автоматически настроит профиль `yc CLI`, сгенерирует `sa-key.json` и заполнит файл `.env` в корне проекта необходимыми переменными окружения, включая API ключи для FastAPI приложения. Убедитесь, что Python 3.x установлен в окружении, где запускается `terraform apply`.


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

После успешного применения Terraform, вы можете получить важные выходные данные, необходимые для дальнейшей настройки и работы системы. Эти данные включают ID созданных ресурсов и ключи доступа.

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

## Интеграция с основной системой

После создания инфраструктуры она автоматически интегрируется с FastAPI приложением. API приложение использует переменные окружения для подключения к DataSphere. Управление заданиями, мониторинг и получение результатов осуществляется через API endpoints, которые взаимодействуют с DataSphere.

## Дополнительная информация

Для получения подробной информации об интеграции API, мониторинге, логировании и аспектах безопасности, пожалуйста, обратитесь к [README.md в директории `deployment`](../README.md).