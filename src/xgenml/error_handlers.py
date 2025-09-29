# /src/xgenml/error_handlers.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime
import logging

# XgenMLException import 추가
from src.xgenml.exceptions import XgenMLException

logger = logging.getLogger(__name__)

def create_error_response(
    error_code: str, 
    message: str, 
    status_code: int = 500,
    details: dict = None
) -> JSONResponse:
    """표준화된 에러 응답 생성"""
    
    content = {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    if details:
        content["error"]["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )

async def xgenml_exception_handler(request: Request, exc: XgenMLException):
    """XgenML 커스텀 예외 핸들러"""
    logger.error(f"XgenML error: {exc.error_code} - {exc.message}", extra={"details": exc.details})
    
    # 에러 타입에 따른 HTTP 상태 코드 매핑
    status_map = {
        "DataLoadError": 400,
        "ModelNotFoundError": 404,
        "ValidationError": 422,
        "TrainingError": 500,
        "PredictionError": 500,
        "ConfigurationError": 500,
    }
    
    status_code = status_map.get(exc.error_code, 500)
    
    return create_error_response(
        error_code=exc.error_code,
        message=exc.message,
        status_code=status_code,
        details=exc.details
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic 검증 에러 핸들러"""
    logger.warning(f"Validation error: {exc.errors()}")
    
    return create_error_response(
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        status_code=422,
        details={"validation_errors": exc.errors()}
    )

async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 핸들러"""
    logger.error(f"Unhandled exception: {type(exc).__name__} - {str(exc)}", exc_info=True)
    
    return create_error_response(
        error_code="INTERNAL_ERROR",
        message="An internal server error occurred",
        status_code=500,
        details={"exception_type": type(exc).__name__}
    )