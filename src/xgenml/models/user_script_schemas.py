"""
User Script Management Schemas
사용자 스크립트 관리를 위한 Pydantic 스키마
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4
import re


# ========================
# User Script Metadata
# ========================
class UserScriptMetadata(BaseModel):
    """사용자 스크립트 메타데이터"""
    name: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_]+$')
    display_name: str = Field(..., max_length=256)
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$')  # SemVer
    description: str = Field(default="", max_length=1000)
    tags: List[str] = Field(default_factory=list)
    task: str  # classification, regression, forecasting 등

    @field_validator('task')
    @classmethod
    def validate_task(cls, v: str) -> str:
        allowed_tasks = [
            'classification',
            'regression',
            'forecasting',
            'clustering',
            'anomaly_detection',
            'timeseries'
        ]
        if v not in allowed_tasks:
            raise ValueError(f"task must be one of {allowed_tasks}")
        return v


# ========================
# User Script Run Config
# ========================
class UserScriptRunConfig(BaseModel):
    """스크립트 실행 설정"""
    dataset_uri: str
    target_column: str
    feature_columns: List[str] = Field(default_factory=list, max_length=1000)
    artifact_dir: str
    random_seed: int = 42


# ========================
# Artifact
# ========================
class Artifact(BaseModel):
    """아티팩트 정보"""
    name: str
    path: str
    size_bytes: int
    type: str  # "model", "metadata", "plot", "log"
    created_at: datetime
    checksum: Optional[str] = None  # SHA256 체크섬


# ========================
# User Script Result
# ========================
class UserScriptResult(BaseModel):
    """스크립트 실행 결과"""
    metrics: Dict[str, float]
    warnings: List[str]
    errors: List[str]
    artifacts: List[Artifact]
    model: Optional[Any] = None  # 훈련된 모델 객체 (선택사항)

    class Config:
        arbitrary_types_allowed = True  # 모델 객체 허용


# ========================
# Resource Usage
# ========================
class ResourceUsage(BaseModel):
    """리소스 사용량"""
    max_memory_mb: float
    cpu_time_seconds: float


# ========================
# Script Execution Result
# ========================
class ScriptExecutionResult(BaseModel):
    """전체 스크립트 실행 결과"""
    stdout: List[str]
    stderr: List[str]
    result: Optional[UserScriptResult] = None
    duration_seconds: float
    started_at: datetime
    finished_at: datetime
    exit_code: int
    resource_usage: Optional[ResourceUsage] = None


# ========================
# Draft Model
# ========================
class Draft(BaseModel):
    """드래프트 데이터 모델"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., max_length=256)
    script_path: str = Field(..., max_length=512)
    content: str = Field(..., max_length=1048576)  # 1MB
    task: str
    run_config: UserScriptRunConfig
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_validation_status: Optional[str] = None  # "passed", "failed", null
    last_validation_at: Optional[datetime] = None
    user_id: Optional[str] = None

    @field_validator('task')
    @classmethod
    def validate_task(cls, v: str) -> str:
        allowed_tasks = [
            'classification',
            'regression',
            'forecasting',
            'clustering',
            'anomaly_detection',
            'timeseries'
        ]
        if v not in allowed_tasks:
            raise ValueError(f"task must be one of {allowed_tasks}")
        return v


# ========================
# API Request/Response Models
# ========================

# Validation
class ScriptValidateRequest(BaseModel):
    """스크립트 검증 요청"""
    script_path: str = Field(..., max_length=512)
    content: str = Field(..., max_length=1048576)  # 1MB


class ValidationMessage(BaseModel):
    """검증 메시지"""
    level: str  # "info", "warning", "error"
    message: str
    code: Optional[str] = None


class ScriptValidateResponse(BaseModel):
    """스크립트 검증 응답"""
    is_valid: bool
    messages: List[ValidationMessage]
    metadata: Optional[Dict[str, Any]] = None
    validated_at: datetime


# Execute
class ScriptExecuteRequest(BaseModel):
    """스크립트 실행 요청"""
    script_path: str = Field(..., max_length=512)
    content: str = Field(..., max_length=1048576)
    run_config: UserScriptRunConfig


class ScriptExecuteResponse(BaseModel):
    """스크립트 실행 응답"""
    stdout: List[str]
    stderr: List[str]
    result: Optional[UserScriptResult] = None
    duration_seconds: float
    started_at: datetime
    finished_at: datetime
    exit_code: int
    resource_usage: Optional[ResourceUsage] = None


# Register
class ScriptRegisterRequest(BaseModel):
    """스크립트 등록 요청"""
    script_path: str = Field(..., max_length=512)
    content: str = Field(..., max_length=1048576)


class CatalogEntry(BaseModel):
    """카탈로그 엔트리"""
    name: str
    display_name: str
    tags: List[str]
    script_path: str
    version: str
    task: str
    description: str


class ScriptRegisterResponse(BaseModel):
    """스크립트 등록 응답"""
    catalog_entry: CatalogEntry
    message: str
    registered_at: datetime


# Draft CRUD
class DraftCreateRequest(BaseModel):
    """드래프트 생성 요청"""
    name: str = Field(..., max_length=256)
    script_path: str = Field(..., max_length=512)
    content: str = Field(..., max_length=1048576)
    task: str
    run_config: UserScriptRunConfig


class DraftUpdateRequest(BaseModel):
    """드래프트 수정 요청 (부분 업데이트)"""
    name: Optional[str] = Field(None, max_length=256)
    script_path: Optional[str] = Field(None, max_length=512)
    content: Optional[str] = Field(None, max_length=1048576)
    run_config: Optional[Dict[str, Any]] = None


class DraftCloneRequest(BaseModel):
    """드래프트 복제 요청"""
    name: Optional[str] = Field(None, max_length=256)


class DraftListResponse(BaseModel):
    """드래프트 목록 응답"""
    drafts: List[Dict[str, Any]]  # content 필드 제외
    total: int
    limit: int
    offset: int
    has_more: bool


class DraftDeleteResponse(BaseModel):
    """드래프트 삭제 응답"""
    message: str
    deleted_id: str


# ========================
# Error Response
# ========================
class ErrorDetail(BaseModel):
    """에러 상세 정보"""
    message: str
    detail: Optional[str] = None
    code: str


class ErrorResponse(BaseModel):
    """통일된 에러 응답"""
    error: ErrorDetail
