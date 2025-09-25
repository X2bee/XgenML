# /src/xgenml/services/serving_service.py (전체 업데이트)
import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import mlflow
import joblib
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder

from .model_cache import model_cache
from ..exceptions import ModelNotFoundError, ValidationError, PredictionError

logger = logging.getLogger(__name__)


class ServingService:
    """
    MLflow Model Registry를 소스 오브 트루스로 사용한 서빙 서비스.
    라벨 인코딩 지원 포함.
    """

    def __init__(self) -> None:
        self.client = MlflowClient()

    def _get_prod_version(self, model_name: str):
        """Model Registry에서 현재 Production 버전을 반환."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            for v in versions:
                if v.current_stage == "Production":
                    return v
            raise RuntimeError(f"No Production version for model '{model_name}'")
        except Exception as e:
            logger.error(f"Failed to get production version for model '{model_name}': {str(e)}")
            raise

    def _load_feature_names_from_tags(self, mv) -> List[str]:
        """모델 버전 태그에서 feature_names를 JSON으로 읽음."""
        try:
            tags = mv.tags or {}
            if "feature_names" in tags:
                try:
                    feature_names = json.loads(tags["feature_names"])
                    if isinstance(feature_names, list):
                        logger.debug(f"Loaded {len(feature_names)} feature names from model version tags")
                        return feature_names
                    else:
                        logger.warning("feature_names in tags is not a list")
                        return []
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse feature_names from tags: {e}")
                    return []
            return []
        except Exception as e:
            logger.warning(f"Error loading feature names from tags: {e}")
            return []

    def _load_feature_names_from_manifest_artifact(self, run_id: str) -> List[str]:
        """백업 경로: run 아티팩트('manifest/manifest.json')에서 feature_names를 복원."""
        try:
            logger.debug(f"Attempting to load feature names from manifest artifact for run {run_id}")
            tmp_dir = "/tmp"
            local_path = self.client.download_artifacts(run_id, "manifest/manifest.json", tmp_dir)
            
            with open(local_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            
            feats = manifest.get("feature_names") or []
            if isinstance(feats, list):
                logger.debug(f"Loaded {len(feats)} feature names from manifest artifact")
                return feats
            else:
                logger.warning("feature_names in manifest is not a list")
                return []
                
        except Exception as e:
            logger.warning(f"Failed to load feature names from manifest artifact: {e}")
            return []

    def _get_feature_names(self, mv) -> List[str]:
        """feature_names는 태그에서 우선 읽고, 없으면 run 아티팩트(manifest)에서 복구."""
        feats = self._load_feature_names_from_tags(mv)
        if feats:
            return feats
        
        logger.info("Feature names not found in tags, trying manifest artifact")
        return self._load_feature_names_from_manifest_artifact(mv.run_id)

    def _load_label_encoder_from_run(self, run_id: str) -> Optional[LabelEncoder]:
        """Run에서 라벨 인코더 로드"""
        try:
            # 라벨 인코딩 사용 여부 확인
            run = self.client.get_run(run_id)
            if run.data.params.get("label_encoded") == "True":
                logger.info(f"Loading label encoder from run {run_id}")
                
                # 라벨 인코더 아티팩트 다운로드
                tmp_dir = "/tmp"
                try:
                    encoder_path = self.client.download_artifacts(
                        run_id, "preprocessing", tmp_dir
                    )
                    
                    # 인코더 파일 찾기
                    for file in os.listdir(encoder_path):
                        if file.endswith("_label_encoder.pkl"):
                            encoder_file = os.path.join(encoder_path, file)
                            label_encoder = joblib.load(encoder_file)
                            logger.info(f"Successfully loaded label encoder with classes: {label_encoder.classes_}")
                            return label_encoder
                
                except Exception as download_error:
                    logger.warning(f"Failed to download label encoder artifacts: {download_error}")
        
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load label encoder from run {run_id}: {e}")
            return None

    def _get_label_info_from_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Run에서 라벨 인코딩 정보 추출"""
        try:
            run = self.client.get_run(run_id)
            params = run.data.params
            
            if params.get("label_encoded") == "True":
                return {
                    "original_classes": json.loads(params.get("original_classes", "[]")),
                    "label_mapping": json.loads(params.get("label_mapping", "{}"))
                }
            return None
        except Exception as e:
            logger.warning(f"Failed to get label info from run {run_id}: {e}")
            return None

    def get_model_and_schema(self, model_id: str):
        """Production 버전 모델과 feature_names를 반환. 캐시 우선 사용."""
        try:
            # 캐시에서 먼저 조회
            cached_result = model_cache.get(model_id)
            if cached_result:
                logger.debug(f"Using cached model for {model_id}")
                return cached_result
            
            # 캐시 미스 - MLflow에서 로드
            logger.info(f"Loading model {model_id} from MLflow (cache miss)")
            
            # Production 버전 조회
            mv = self._get_prod_version(model_id)
            
            # 모델 로드
            try:
                model = mlflow.sklearn.load_model(f"models:/{model_id}/Production")
                logger.info(f"Successfully loaded model {model_id} from MLflow Model Registry")
            except Exception as e:
                logger.error(f"Failed to load model from MLflow: {str(e)}")
                raise
            
            # 피처 이름 로드
            feature_names = self._get_feature_names(mv)
            if not feature_names:
                logger.warning(f"No feature names found for model {model_id}")
            else:
                logger.info(f"Loaded {len(feature_names)} feature names for model {model_id}")
            
            # 캐시에 저장
            model_cache.put(model_id, model, feature_names)
            
            return model, feature_names
            
        except Exception as e:
            if "No Production version" in str(e) or "RESOURCE_DOES_NOT_EXIST" in str(e):
                raise ModelNotFoundError(
                    f"No production model found for '{model_id}'. Please ensure the model is trained and registered.",
                    details={
                        "model_id": model_id,
                        "suggestion": "Train a model first using the /api/training/sync endpoint"
                    }
                )
            else:
                raise ModelNotFoundError(
                    f"Failed to load model '{model_id}': {str(e)}",
                    details={
                        "model_id": model_id, 
                        "original_error": str(e),
                        "error_type": type(e).__name__
                    }
                )

    def predict(self, model_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """JSON 레코드 배열을 입력받아 예측 결과를 반환. 라벨 디코딩 포함."""
        if not records:
            raise ValidationError(
                "Empty records list provided",
                details={"model_id": model_id, "records_count": 0}
            )
        
        try:
            # 모델과 스키마 로드
            model, feature_names = self.get_model_and_schema(model_id)
            
            # Production 버전에서 라벨 인코더와 정보 로드
            mv = self._get_prod_version(model_id)
            label_encoder = self._load_label_encoder_from_run(mv.run_id)
            label_info = self._get_label_info_from_run(mv.run_id)
            
            # DataFrame 변환
            try:
                df = pd.DataFrame.from_records(records)
                logger.debug(f"Created DataFrame with shape {df.shape} from {len(records)} records")
            except Exception as e:
                raise ValidationError(
                    f"Failed to convert records to DataFrame: {str(e)}",
                    details={
                        "model_id": model_id,
                        "records_count": len(records),
                        "sample_record": records[0] if records else None
                    }
                )

            # 컬럼 검증/정렬
            if feature_names:
                provided_features = df.columns.tolist()
                missing = [c for c in feature_names if c not in df.columns]
                extra = [c for c in df.columns if c not in feature_names]
                
                if missing:
                    raise ValidationError(
                        f"Missing required features for model '{model_id}'",
                        details={
                            "model_id": model_id,
                            "missing_features": missing,
                            "required_features": feature_names,
                            "provided_features": provided_features,
                            "suggestion": "Ensure all required features are present in your input data"
                        }
                    )
                
                if extra:
                    logger.info(f"Extra features provided (will be ignored): {extra}")
                
                df = df[feature_names]
                logger.debug(f"Filtered DataFrame to required features: {feature_names}")
            else:
                logger.warning(f"No feature schema available for model {model_id}, using all provided features")

            # 예측 수행
            try:
                preds = model.predict(df)
                logger.debug(f"Generated {len(preds)} predictions")
                
                # 라벨 디코딩 (문자열 라벨로 복원)
                if label_encoder is not None:
                    try:
                        preds_decoded = label_encoder.inverse_transform(preds)
                        result = {
                            "predictions": preds_decoded.tolist(),
                            "predictions_encoded": preds.tolist(),  # 숫자 버전도 함께 제공
                            "label_info": {
                                "used_encoding": True,
                                "original_classes": label_info["original_classes"] if label_info else label_encoder.classes_.tolist(),
                                "label_mapping": label_info["label_mapping"] if label_info else None
                            }
                        }
                        logger.debug("Successfully decoded predictions to original labels")
                    except Exception as decode_error:
                        logger.warning(f"Label decoding failed: {decode_error}, returning encoded predictions")
                        result = {
                            "predictions": preds.tolist(),
                            "label_info": {"used_encoding": True, "decoding_failed": True}
                        }
                else:
                    result = {
                        "predictions": preds.tolist(),
                        "label_info": {"used_encoding": False}
                    }

                # 분류 모델의 경우 확률까지 제공
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(df)
                        result["probabilities"] = proba.tolist()
                        
                        # 클래스 이름도 함께 제공
                        if label_encoder is not None:
                            result["class_names"] = label_encoder.classes_.tolist()
                        
                        logger.debug("Added prediction probabilities")
                    except Exception as prob_e:
                        logger.warning(f"Failed to get probabilities: {str(prob_e)}")

                return result
                
            except Exception as e:
                raise PredictionError(
                    f"Model prediction failed: {str(e)}",
                    details={
                        "model_id": model_id,
                        "input_shape": df.shape,
                        "feature_names": feature_names,
                        "original_error": str(e)
                    }
                )
            
        except (ModelNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in predict method: {str(e)}", exc_info=True)
            raise PredictionError(
                f"Prediction failed for model '{model_id}': {str(e)}",
                details={
                    "model_id": model_id,
                    "original_error": str(e),
                    "error_type": type(e).__name__
                }
            )

    # 나머지 메서드들은 기존과 동일...
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """모델 정보 조회 (라벨 인코딩 정보 포함)"""
        try:
            mv = self._get_prod_version(model_id)
            feature_names = self._get_feature_names(mv)
            label_info = self._get_label_info_from_run(mv.run_id)
            
            # 캐시 상태 확인
            cache_info = model_cache.get_stats()
            is_cached = model_id in cache_info.get("models", {})
            
            info = {
                "model_id": model_id,
                "version": mv.version,
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "creation_timestamp": mv.creation_timestamp,
                "last_updated_timestamp": mv.last_updated_timestamp,
                "feature_names": feature_names,
                "feature_count": len(feature_names) if feature_names else 0,
                "cache_status": {
                    "is_cached": is_cached,
                    "cache_stats": cache_info.get("models", {}).get(model_id) if is_cached else None
                }
            }
            
            # 라벨 인코딩 정보 추가
            if label_info:
                info["label_encoding"] = {
                    "used": True,
                    "original_classes": label_info["original_classes"],
                    "label_mapping": label_info["label_mapping"]
                }
            else:
                info["label_encoding"] = {"used": False}
            
            return info
            
        except Exception as e:
            if "No Production version" in str(e):
                raise ModelNotFoundError(
                    f"No production model found for '{model_id}'",
                    details={"model_id": model_id}
                )
            else:
                raise ModelNotFoundError(
                    f"Failed to get model info for '{model_id}': {str(e)}",
                    details={"model_id": model_id, "original_error": str(e)}
                )