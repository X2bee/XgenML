# /src/xgenml/core/model_provider.py
import importlib
from typing import Dict, Any, Optional, List
from .model_catalog import (
    CATALOG, 
    validate_model_name, 
    get_model_info,
    check_model_requirements,
    get_models_by_task
)
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)


def create_estimator(task: str, name: str, override: Dict[str, Any] | None = None):
    """
    카탈로그 기반 모델 생성
    
    Args:
        task: 태스크 타입 (classification, regression, timeseries, anomaly_detection, clustering)
        name: 모델 이름
        override: 기본 파라미터 오버라이드
    
    Returns:
        학습 가능한 estimator 객체
    
    Raises:
        ValueError: 알 수 없는 태스크 또는 모델
        ImportError: 필요한 패키지 미설치
    """
    # 카탈로그에서 모델 스펙 찾기
    spec = next((m for m in CATALOG.get(task, []) if m["name"] == name), None)
    if spec is None:
        available_tasks = list(CATALOG.keys())
        available_models = [m["name"] for m in CATALOG.get(task, [])]
        raise ValueError(
            f"모델 '{name}'이(가) task '{task}' 카탈로그에 없습니다.\n"
            f"사용 가능한 태스크: {available_tasks}\n"
            f"'{task}' 태스크의 사용 가능한 모델: {available_models}"
        )
    
    # 필요한 패키지 확인
    req_check = check_model_requirements(task, name)
    if not req_check["available"]:
        missing = req_check["missing_packages"]
        raise ImportError(
            f"모델 '{name}'에 필요한 패키지가 설치되지 않았습니다: {missing}\n"
            f"설치 명령: pip install {' '.join(missing)}"
        )
    
    # 모델 클래스 동적 임포트
    module_path, cls_name = spec["cls"].rsplit(".", 1)
    
    try:
        module = importlib.import_module(module_path)
        Estimator = getattr(module, cls_name)
        logger.info(f"모델 클래스 로드 성공: {spec['cls']}")
    except ImportError as e:
        raise ImportError(
            f"모듈 '{module_path}'를 임포트할 수 없습니다: {e}\n"
            f"필요한 패키지를 설치했는지 확인하세요."
        )
    except AttributeError as e:
        raise ImportError(
            f"모듈 '{module_path}'에서 클래스 '{cls_name}'를 찾을 수 없습니다: {e}"
        )
    
    # 파라미터 병합 (기본값 + 오버라이드)
    params = {**spec.get("default", {}), **(override or {})}
    
    logger.info(f"모델 생성: {name} (task: {task})")
    logger.info(f"파라미터: {params}")
    
    try:
        estimator = Estimator(**params)
    except Exception as e:
        raise ValueError(
            f"모델 '{name}' 생성 실패. 파라미터: {params}\n"
            f"에러: {e}"
        )
    
    return estimator


def get_model_default_params(task: str, name: str) -> Dict[str, Any]:
    """모델의 기본 파라미터 조회"""
    if task not in CATALOG:
        raise ValueError(f"Unknown task: {task}")
    
    spec = next((m for m in CATALOG[task] if m["name"] == name), None)
    if spec is None:
        raise ValueError(f"Model '{name}' not found in task '{task}'")
    
    return spec.get("default", {})


def list_available_models(task: Optional[str] = None) -> Dict[str, List[str]]:
    """
    사용 가능한 모델 목록
    
    Args:
        task: 특정 태스크만 조회 (None이면 전체)
    
    Returns:
        {task: [model_names]} 형태의 딕셔너리
    """
    if task:
        if task not in CATALOG:
            raise ValueError(f"Unknown task: {task}")
        return {task: [m["name"] for m in CATALOG[task]]}
    
    return {
        task_name: [m["name"] for m in models]
        for task_name, models in CATALOG.items()
    }


def get_model_description(task: str, name: str) -> str:
    """모델 설명 조회"""
    model_info = get_model_info(task, name)
    if not model_info:
        return "No description available"
    return model_info.get("description", "No description available")


def validate_models_for_task(task: str, model_names: List[str]) -> Dict[str, Any]:
    """
    여러 모델이 태스크에 유효한지 검증
    
    Returns:
        {
            "valid": [유효한 모델들],
            "invalid": [유효하지 않은 모델들],
            "missing_packages": {model: [packages]}
        }
    """
    result = {
        "valid": [],
        "invalid": [],
        "missing_packages": {}
    }
    
    for name in model_names:
        if not validate_model_name(task, name):
            result["invalid"].append(name)
            continue
        
        req_check = check_model_requirements(task, name)
        if not req_check["available"]:
            result["invalid"].append(name)
            result["missing_packages"][name] = req_check["missing_packages"]
        else:
            result["valid"].append(name)
    
    return result