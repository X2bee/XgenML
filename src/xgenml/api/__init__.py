# /src/xgenml/api/__init__.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

# 에러 핸들러 import
from ..exceptions import XgenMLException
from ..error_handlers import (
    xgenml_exception_handler,
    validation_exception_handler, 
    general_exception_handler
)

# 분리된 라우터들 import
from .training_router import router as training_router
from .serving_router import router as serving_router
from .cache_router import router as cache_router
from .model_catalog_router import router as catalog_router

def create_app() -> FastAPI:
    """FastAPI 앱 팩토리 함수"""
    
    app = FastAPI(
        title="XgenML API",
        description="AutoML platform for training and serving machine learning models",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 에러 핸들러 등록
    app.add_exception_handler(XgenMLException, xgenml_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # CORS 미들웨어 추가
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8080", 'https://xgen.x2bee.com'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(training_router, prefix="/api")      # /api/training/*
    app.include_router(serving_router, prefix="/api")       # /api/predict/*
    app.include_router(cache_router, prefix="/api")         # /api/cache/*
    app.include_router(catalog_router, prefix="/api")       # /api/models/*

    # 헬스체크 엔드포인트
    @app.get("/")
    async def root():
        return {
            "message": "XgenML API is running",
            "version": "1.0.0",
            "docs": "/docs",
            "endpoints": {
                "training": "/api/training",
                "prediction": "/api/predict", 
                "cache": "/api/cache",
                "catalog": "/api/models"
            }
        }

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "xgenml-api"}

    return app

# 기본 앱 인스턴스 생성
app = create_app()