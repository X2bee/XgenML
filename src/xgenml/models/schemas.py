from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

# 기존 스키마 정의 (그대로 유지)
class HyperparameterConfig(BaseModel):
    enable_hpo: bool = False
    n_trials: int = Field(default=50, ge=10, le=500)
    timeout_minutes: Optional[int] = Field(default=None, ge=1, le=120)
    param_spaces: Optional[Dict[str, Dict[str, Any]]] = None

class TrainRequest(BaseModel):
    model_id: str
    task: str = "classification"
    test_size: float = 0.2
    validation_size: float = 0.1
    use_cv: bool = False
    cv_folds: int = 5
    hf_repo: str
    hf_filename: str
    hf_revision: Optional[str] = None
    target_column: str
    feature_columns: Optional[List[str]] = None
    model_names: List[str]
    overrides: Optional[Dict[str, Any]] = None
    
    # HPO 설정 추가
    hpo_config: Optional[HyperparameterConfig] = None

class TrainResponse(BaseModel):
    run_id: str
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None

class TrainingStatusResponse(BaseModel):
    status: str
    progress: Optional[float] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None

class ExperimentResponse(BaseModel):
    id: str
    name: str
    lifecycle_stage: str
    creation_time: Optional[int] = None
