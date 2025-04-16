from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import logging
from typing import Dict, Any, Optional, List, Union

from app.utils.validation import ValidationError as AppValidationError

logger = logging.getLogger("plastinka.errors")

class ErrorDetail:
    """Error detail model for consistent error responses."""
    
    def __init__(
        self, 
        message: str, 
        code: str = "internal_error",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "error": {
                "message": self.message,
                "code": self.code,
                "details": self.details
            }
        }


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle validation errors from FastAPI.
    """
    error_details = []
    
    for error in exc.errors():
        error_details.append({
            "loc": error.get("loc", []),
            "msg": error.get("msg", ""),
            "type": error.get("type", "")
        })
    
    logger.warning(f"Validation error: {error_details}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorDetail(
            message="Validation error",
            code="validation_error",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"errors": error_details}
        ).to_dict()
    )


async def app_validation_exception_handler(request: Request, exc: AppValidationError) -> JSONResponse:
    """
    Handle application-specific validation errors.
    """
    logger.warning(f"Application validation error: {exc.message}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorDetail(
            message=exc.message,
            code="validation_error",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=exc.details
        ).to_dict()
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all other exceptions.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorDetail(
            message="Internal server error",
            code="internal_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": str(exc)}
        ).to_dict()
    )


def configure_error_handlers(app):
    """
    Configure exception handlers for the FastAPI application.
    """
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(AppValidationError, app_validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("Error handlers configured")
    
    return app 