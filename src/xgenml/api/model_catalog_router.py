# /src/xgenml/api/model_catalog_router.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any

from ..core.model_catalog import (
    get_all_models,
    get_models_by_task, 
    get_model_info,
    get_models_by_tag,
    validate_model_name
)

router = APIRouter(prefix="/models", tags=["Model Catalog"])

@router.get("/catalog")
def get_model_catalog(
    task: Optional[str] = Query(None, description="Filter by task type (classification/regression)"),
    tag: Optional[str] = Query(None, description="Filter by model tag"),
    include_details: bool = Query(True, description="Include model descriptions and metadata")
):
    """
    사용 가능한 모든 모델의 카탈로그를 반환합니다.
    
    - **task**: 특정 태스크 타입으로 필터링 (classification 또는 regression)
    - **tag**: 특정 태그로 필터링 (예: ensemble, boosting)
    - **include_details**: 모델 설명과 메타데이터 포함 여부
    """
    try:
        if tag:
            # 태그로 필터링
            catalog = get_models_by_tag(tag)
        elif task:
            # 태스크로 필터링
            if task not in ["classification", "regression"]:
                raise HTTPException(
                    status_code=400, 
                    detail="Task must be either 'classification' or 'regression'"
                )
            models = get_models_by_task(task)
            catalog = {task: models}
        else:
            # 전체 카탈로그
            catalog = get_all_models()
        
        # 세부 정보 제거 옵션
        if not include_details:
            simplified_catalog = {}
            for task_name, models in catalog.items():
                simplified_catalog[task_name] = [
                    {"name": model["name"], "cls": model["cls"]} 
                    for model in models
                ]
            catalog = simplified_catalog
        
        return {
            "catalog": catalog,
            "total_models": sum(len(models) for models in catalog.values()),
            "tasks": list(catalog.keys()),
            "filter_applied": {
                "task": task,
                "tag": tag,
                "include_details": include_details
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model catalog: {str(e)}")

@router.get("/catalog/{task}")
def get_models_for_task(task: str):
    """특정 태스크의 사용 가능한 모델들을 반환합니다."""
    
    if task not in ["classification", "regression"]:
        raise HTTPException(
            status_code=400,
            detail="Task must be either 'classification' or 'regression'"
        )
    
    try:
        models = get_models_by_task(task)
        return {
            "task": task,
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models for task {task}: {str(e)}")

@router.get("/catalog/{task}/{model_name}")
def get_model_details(task: str, model_name: str):
    """특정 모델의 상세 정보를 반환합니다."""
    
    if task not in ["classification", "regression"]:
        raise HTTPException(
            status_code=400,
            detail="Task must be either 'classification' or 'regression'"
        )
    
    try:
        model_info = get_model_info(task, model_name)
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found for task '{task}'"
            )
        
        return {
            "task": task,
            "model": model_info,
            "usage_example": {
                "training_request": {
                    "model_names": [model_name],
                    "task": task,
                    "overrides": {
                        model_name: {
                            "n_estimators": "adjust based on your data size",
                            "learning_rate": "tune for optimal performance"
                        }
                    }
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model details: {str(e)}")

@router.get("/tags")
def get_available_tags():
    """사용 가능한 모든 모델 태그를 반환합니다."""
    try:
        all_tags = set()
        catalog = get_all_models()
        
        for task, models in catalog.items():
            for model in models:
                tags = model.get("tags", [])
                all_tags.update(tags)
        
        return {
            "tags": sorted(list(all_tags)),
            "count": len(all_tags)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available tags: {str(e)}")

@router.post("/validate")
def validate_model_selection(request: Dict[str, Any]):
    """
    모델 선택이 유효한지 검증합니다.
    
    Request body:
    {
        "task": "classification",
        "model_names": ["xgboost", "random_forest"],
        "overrides": {"xgboost": {"n_estimators": 500}}
    }
    """
    try:
        task = request.get("task")
        model_names = request.get("model_names", [])
        overrides = request.get("overrides", {})
        
        if not task:
            raise HTTPException(status_code=400, detail="Task is required")
        
        if task not in ["classification", "regression"]:
            raise HTTPException(
                status_code=400,
                detail="Task must be either 'classification' or 'regression'"
            )
        
        if not model_names:
            raise HTTPException(status_code=400, detail="At least one model name is required")
        
        validation_results = {
            "valid": True,
            "task": task,
            "validated_models": [],
            "invalid_models": [],
            "warnings": []
        }
        
        available_models = [m["name"] for m in get_models_by_task(task)]
        
        for model_name in model_names:
            if validate_model_name(task, model_name):
                model_info = get_model_info(task, model_name)
                validation_results["validated_models"].append({
                    "name": model_name,
                    "status": "valid",
                    "default_params": model_info.get("default", {}),
                    "override_params": overrides.get(model_name, {})
                })
            else:
                validation_results["invalid_models"].append(model_name)
                validation_results["valid"] = False
        
        if validation_results["invalid_models"]:
            validation_results["warnings"].append(
                f"Invalid models for {task}: {validation_results['invalid_models']}. "
                f"Available models: {available_models}"
            )
        
        return validation_results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model validation failed: {str(e)}")