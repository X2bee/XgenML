# /src/xgenml/api/cache_router.py
from fastapi import APIRouter, HTTPException
import logging

from ..services.model_cache import model_cache

router = APIRouter(prefix="/cache", tags=["Cache Management"])
logger = logging.getLogger("uvicorn.error")

@router.get("/stats")
def get_cache_stats():
    """모델 캐시 통계 조회"""
    try:
        return model_cache.get_stats()
    except Exception as e:
        logger.error(f"Failed to get cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_id}")
def invalidate_model_cache(model_id: str):
    """특정 모델 캐시 무효화"""
    try:
        model_cache.invalidate(model_id)
        return {"message": f"Cache invalidated for model {model_id}"}
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear")
def clear_all_cache():
    """전체 모델 캐시 클리어"""
    try:
        model_cache.clear()
        return {"message": "All model cache cleared"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}")
def get_model_cache_info(model_id: str):
    """특정 모델의 캐시 정보 조회"""
    try:
        stats = model_cache.get_stats()
        model_info = stats.get("models", {}).get(model_id)
        
        if not model_info:
            return {
                "model_id": model_id,
                "cached": False,
                "message": "Model not in cache"
            }
        
        return {
            "model_id": model_id,
            "cached": True,
            "cache_info": model_info
        }
    except Exception as e:
        logger.error(f"Failed to get model cache info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))