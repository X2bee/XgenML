# /src/xgenml/api/serving_router.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import json
import logging

from ..services.serving_service import ServingService
from ..services.batch_serving_service import batch_serving_service

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = logging.getLogger("uvicorn.error")

serving_service = ServingService()

@router.post("/{model_id}")
def predict_single(model_id: str, records: List[Dict[str, Any]]):
    """단일/소량 예측"""
    try:
        result = serving_service.predict(model_id, records)
        return result
        
    except Exception as e:
        logger.error(f"Single prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/batch")
def predict_batch(model_id: str, records: List[Dict[str, Any]]):
    """동기 배치 예측 (작은 배치용)"""
    try:
        result = batch_serving_service.predict_batch_sync(model_id, records)
        return result
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/batch/stream")
async def predict_batch_stream(model_id: str, records: List[Dict[str, Any]]):
    """스트리밍 배치 예측 (대용량 배치용)"""
    
    async def generate_predictions():
        async for batch_result in batch_serving_service.predict_batch_stream(model_id, records):
            yield f"data: {json.dumps(batch_result)}\n\n"
    
    return StreamingResponse(
        generate_predictions(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@router.get("/{model_id}/info")
def get_model_info(model_id: str):
    """모델 정보 조회"""
    try:
        return serving_service.get_model_info(model_id)
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/validate")
def validate_input_schema(model_id: str, sample_record: Dict[str, Any]):
    """입력 스키마 검증 (실제 예측 없이)"""
    try:
        return serving_service.validate_input_schema(model_id, sample_record)
    except Exception as e:
        logger.error(f"Schema validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))