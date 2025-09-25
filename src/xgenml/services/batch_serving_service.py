# /src/xgenml/services/batch_serving_service.py
import asyncio
import pandas as pd
from typing import List, Dict, Any, AsyncGenerator
import logging
from .serving_service import ServingService

logger = logging.getLogger(__name__)

class BatchServingService:
    def __init__(self, batch_size: int = 1000):
        self.serving_service = ServingService()
        self.batch_size = batch_size
    
    async def predict_batch_stream(
        self, 
        model_id: str, 
        records: List[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """스트리밍 방식으로 배치 예측 수행"""
        
        total_records = len(records)
        logger.info(f"Starting batch prediction for {total_records} records with batch size {self.batch_size}")
        
        # 모델과 스키마 한 번만 로드
        model, feature_names = self.serving_service.get_model_and_schema(model_id)
        
        processed = 0
        for i in range(0, total_records, self.batch_size):
            batch_records = records[i:i + self.batch_size]
            batch_df = pd.DataFrame.from_records(batch_records)
            
            # 스키마 검증 및 정렬
            if feature_names:
                missing = [c for c in feature_names if c not in batch_df.columns]
                if missing:
                    error_msg = f"Missing features for model '{model_id}': {missing}"
                    yield {
                        "error": error_msg,
                        "batch_index": i // self.batch_size,
                        "processed": processed
                    }
                    continue
                batch_df = batch_df[feature_names]
            
            try:
                # 예측 수행
                predictions = model.predict(batch_df)
                probabilities = None
                
                if hasattr(model, "predict_proba"):
                    try:
                        probabilities = model.predict_proba(batch_df)
                    except Exception:
                        pass
                
                # 결과 준비
                batch_result = {
                    "batch_index": i // self.batch_size,
                    "batch_size": len(batch_records),
                    "predictions": predictions.tolist(),
                    "processed": processed + len(batch_records),
                    "total": total_records,
                    "progress": min(100.0, (processed + len(batch_records)) / total_records * 100)
                }
                
                if probabilities is not None:
                    batch_result["probabilities"] = probabilities.tolist()
                
                processed += len(batch_records)
                yield batch_result
                
                # CPU 양보 (다른 요청들이 처리될 수 있도록)
                await asyncio.sleep(0)
                
            except Exception as e:
                logger.error(f"Batch prediction failed for batch {i // self.batch_size}: {str(e)}")
                yield {
                    "error": str(e),
                    "batch_index": i // self.batch_size,
                    "processed": processed
                }
        
        logger.info(f"Batch prediction completed: {processed}/{total_records} records processed")
    
    def predict_batch_sync(self, model_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """동기 방식 배치 예측 (작은 배치용)"""
        
        if len(records) > 10000:
            raise ValueError("For large batches (>10k records), use async streaming prediction")
        
        model, feature_names = self.serving_service.get_model_and_schema(model_id)
        df = pd.DataFrame.from_records(records)
        
        # 스키마 검증
        if feature_names:
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                raise ValueError(f"Missing features for model '{model_id}': {missing}")
            df = df[feature_names]
        
        predictions = model.predict(df)
        result = {"predictions": predictions.tolist()}
        
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(df)
                result["probabilities"] = probabilities.tolist()
            except Exception:
                pass
        
        return result

# 글로벌 배치 서빙 서비스 인스턴스
batch_serving_service = BatchServingService(batch_size=1000)