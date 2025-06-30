import json
import logging
import os
import traceback
import uuid
from datetime import date, datetime
from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from deployment.app.utils.validation import ValidationError as AppValidationError

# Configure more detailed logging
logger = logging.getLogger("plastinka.errors")

# Enable detailed error responses in development environment
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
INCLUDE_EXCEPTION_DETAILS = ENVIRONMENT.lower() in ["development", "testing"]


def json_default_serializer(obj):
    """
    JSON serializer for objects not serializable by default json code
    """
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class RetryableError(Exception):
    """Exception class for errors that should be automatically retried."""

    def __init__(
        self,
        message: str,
        code: str = "retryable_error",
        retry_after: int | None = None,
        max_retries: int | None = None,
        original_exception: Exception | None = None,
    ):
        self.message = message
        self.code = code
        self.retry_after = retry_after  # Seconds to wait before retry
        self.max_retries = max_retries  # Maximum number of retries
        self.original_exception = original_exception
        super().__init__(message)


class ErrorDetail:
    """Error detail model for consistent error responses."""

    def __init__(
        self,
        message: str,
        code: str = "internal_error",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
        exception: Exception | None = None,
        request_id: str | None = None,
        retry_info: dict[str, Any] | None = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        self.exception = exception
        self.request_id = request_id or str(uuid.uuid4())
        self.retry_info = retry_info

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON response."""
        error_dict = {
            "error": {
                "message": self.message,
                "code": self.code,
                "request_id": self.request_id,
                "details": self.details,
            }
        }

        # Add retry information if available
        if self.retry_info:
            error_dict["error"]["retry_info"] = self.retry_info

        # Add exception details in development mode
        if INCLUDE_EXCEPTION_DETAILS and self.exception:
            error_dict["error"]["exception"] = {
                "type": type(self.exception).__name__,
                "traceback": traceback.format_exception(
                    type(self.exception), self.exception, self.exception.__traceback__
                )
                if self.exception.__traceback__
                else None,
            }

        return error_dict

    def log_error(self, request: Request | None = None):
        """Log error with consistent structure."""
        log_data = {
            "request_id": self.request_id,
            "error_code": self.code,
            "error_message": self.message,
            "status_code": self.status_code,
        }

        # Add request data if available
        if request:
            log_data["method"] = request.method
            log_data["url"] = str(request.url)
            log_data["client"] = request.client.host if request.client else None
            log_data["headers"] = dict(request.headers)

        # Add exception details
        if self.exception:
            log_data["exception_type"] = type(self.exception).__name__
            log_data["exception_msg"] = str(self.exception)

        # Add retry information if available
        if self.retry_info:
            log_data["retry_info"] = self.retry_info

        # Use the custom JSON serializer for logging
        if self.status_code >= 500:
            logger.error(
                json.dumps(log_data, default=json_default_serializer),
                exc_info=self.exception,
            )
        elif self.status_code >= 400:
            logger.warning(json.dumps(log_data, default=json_default_serializer))
        else:
            logger.info(json.dumps(log_data, default=json_default_serializer))


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle validation errors from FastAPI.
    """
    error_details = []

    for error in exc.errors():
        error_details.append(
            {
                "loc": error.get("loc", []),
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            }
        )

    error = ErrorDetail(
        message="Validation error",
        code="validation_error",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={"errors": error_details},
        exception=exc,
    )

    error.log_error(request)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error.to_dict()
    )


async def app_validation_exception_handler(
    request: Request, exc: AppValidationError
) -> JSONResponse:
    """
    Handle application-specific validation errors.
    """
    error = ErrorDetail(
        message=exc.message,
        code="validation_error",
        status_code=status.HTTP_400_BAD_REQUEST,
        details=exc.details,
        exception=exc,
    )

    error.log_error(request)

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content=error.to_dict()
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle HTTPException from FastAPI.
    """
    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        detail = exc.detail
        headers = getattr(exc, "headers", None)

        error = ErrorDetail(
            message=str(detail),
            code=f"http_{status_code}",
            status_code=status_code,
            exception=exc,
        )

        error.log_error(request)

        response = JSONResponse(status_code=status_code, content=error.to_dict())

        if headers:
            for name, value in headers.items():
                response.headers[name] = value

        return response

    # If it's not an HTTPException, pass to generic handler
    return await generic_exception_handler(request, exc)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all other exceptions.
    """
    error = ErrorDetail(
        message="Internal server error",
        code="internal_error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={"error": str(exc)},
        exception=exc,
    )

    error.log_error(request)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error.to_dict()
    )


async def retryable_exception_handler(
    request: Request, exc: RetryableError
) -> JSONResponse:
    """
    Handle retryable errors.
    """
    retry_info = {}
    if exc.retry_after is not None:
        retry_info["retry_after"] = exc.retry_after
    if exc.max_retries is not None:
        retry_info["max_retries"] = exc.max_retries

    error = ErrorDetail(
        message=exc.message,
        code=exc.code,
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        exception=exc.original_exception or exc,
        retry_info=retry_info,
    )

    error.log_error(request)

    response = JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=error.to_dict()
    )

    if exc.retry_after is not None:
        response.headers["Retry-After"] = str(exc.retry_after)

    return response


def configure_error_handlers(app):
    """
    Configure exception handlers for the FastAPI application.
    """
    # Import here to avoid circular imports
    from fastapi import HTTPException

    # Add exception handlers
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(AppValidationError, app_validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RetryableError, retryable_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Error handlers configured")

    return app
