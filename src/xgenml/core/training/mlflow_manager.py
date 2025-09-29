# /src/xgenml/core/training/mlflow_manager.py
import os
import json
import tempfile
import joblib
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.xgenml.utils.logger_config import setup_logger

logger = setup_logger(__name__)

PRIMARY_METRIC = {"classification": "accuracy", "regression": "r2"}


class MLflowManager:
    """MLflow 관련 작업 담당 클래스"""
    
    def __init__(self, experiment_name: str, artifact_root: Optional[str] = None):
        self.experiment_name = experiment_name
        self.artifact_root = artifact_root or os.getenv(
            "MLFLOW_DEFAULT_ARTIFACT_ROOT", "s3://mlflow-artifacts"
        )
        self.client = MlflowClient()
        self._setup_experiment()
    
    def _setup_experiment(self):
        """MLflow 실험 설정"""
        logger.info(f"\n🔧 MLflow 실험 설정: {self.experiment_name}")
        
        try:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=f"{self.artifact_root}/{self.experiment_name}"
            )
            logger.info(f"✅ 새 실험 생성: {self.experiment_name}")
        except Exception:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id
            logger.info(f"✅ 기존 실험 사용: {self.experiment_name}")
        
        mlflow.set_experiment(experiment_id=self.experiment_id)
    
    def log_model_training(
        self,
        run_name: str,
        estimator,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        X_train,
        y_pred_test,
        execution_id: str,
        data_source_info: Dict[str, Any],
        label_encoding_info: Optional[Dict[str, Any]] = None,
        hpo_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """모델 학습 결과를 MLflow에 로깅"""
        with mlflow.start_run(run_name=run_name):
            run_id = mlflow.active_run().info.run_id
            run_info = mlflow.active_run().info
            logger.info(f"MLflow Run ID: {run_id}")
            logger.info(f"Artifact URI: {run_info.artifact_uri}")
            
            # 기본 파라미터 로깅
            self._log_basic_params(execution_id, data_source_info, params)
            
            # 라벨 인코딩 정보 로깅
            if label_encoding_info and label_encoding_info.get("used"):
                self._log_label_encoding(run_id, label_encoding_info)
            
            # HPO 정보 로깅
            if hpo_results:
                self._log_hpo_results(hpo_results)
            
            # 메트릭 로깅
            self._log_metrics(metrics)
            
            # 모델 저장
            model_saved = self._save_model(estimator, X_train, y_pred_test)
            
            return run_id, model_saved
    
    def _log_basic_params(
        self, execution_id: str, data_source_info: Dict[str, Any], params: Dict[str, Any]
    ):
        """기본 파라미터 로깅"""
        mlflow.log_param("execution_id", execution_id)
        mlflow.log_param("data_source_type", data_source_info["source_type"])
        
        if data_source_info["source_type"] == "mlflow":
            mlflow.log_param("source_mlflow_run_id", data_source_info["mlflow_run_id"])
        else:
            mlflow.log_param("source_hf_repo", data_source_info["hf_repo"])
        
        mlflow.log_params(params)
    
    def _log_label_encoding(self, run_id: str, label_info: Dict[str, Any]):
        """라벨 인코딩 정보 로깅"""
        mlflow.log_param("label_encoded", True)
        mlflow.log_param("original_classes", json.dumps(label_info["original_classes"], ensure_ascii=False))
        mlflow.log_param("label_mapping", json.dumps(label_info["label_mapping"], ensure_ascii=False))
        
        # 라벨 인코더 저장
        encoder_path = f"/tmp/{run_id}_label_encoder.pkl"
        joblib.dump(label_info["encoder"], encoder_path)
        mlflow.log_artifact(encoder_path, artifact_path="preprocessing")
        os.unlink(encoder_path)
        
        logger.info("📋 라벨 인코딩 정보를 MLflow에 저장했습니다.")
    
    def _log_hpo_results(self, hpo_results: Dict[str, Any]):
        """HPO 결과 로깅"""
        best_params = hpo_results.get('best_params', {})
        mlflow.log_params({f"hpo_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("hpo_best_score", hpo_results.get('best_score', 0.0))
        mlflow.log_metric("hpo_n_trials", hpo_results.get('n_trials', 0))
        mlflow.log_param("hpo_enabled", True)
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """메트릭 로깅"""
        logger.info("메트릭 로깅 중...")
        metric_count = 0
        
        for mtype, d in metrics.items():
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{mtype}_{k}", float(v))
                    metric_count += 1
        
        logger.info(f"메트릭 로깅 완료 ({metric_count}개)")
    
    def _save_model(self, estimator, X_train, y_pred_test) -> bool:
        """모델 저장"""
        logger.info("모델 아티팩트 저장 중 (MinIO)...")
        
        try:
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, y_pred_test)
            input_example = X_train.iloc[:3] if hasattr(X_train, 'iloc') else X_train[:3]
            
            mlflow.sklearn.log_model(
                estimator,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=None,
                await_registration_for=0
            )
            logger.info("✅ 모델 아티팩트 저장 완료 (signature + input_example)")
            return True
            
        except Exception as signature_error:
            logger.warning(f"⚠️ Signature 포함 저장 실패: {signature_error}")
            
            try:
                mlflow.sklearn.log_model(
                    estimator,
                    artifact_path="model",
                    registered_model_name=None,
                    await_registration_for=0
                )
                logger.info("✅ 모델 아티팩트 저장 완료 (기본)")
                return True
                
            except Exception as basic_error:
                logger.error(f"❌ 모델 저장 실패: {basic_error}")
                return False
    
    def register_best_model(
        self,
        model_name: str,
        best: Dict[str, Any],
        feature_names: List[str],
        version_tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """베스트 모델을 Model Registry에 등록"""
        logger.info(f"🏷️  Model Registry 등록 시작: {model_name}")
        
        try:
            # MLflow 서버 연결 테스트
            self.client.search_experiments(max_results=1)
            logger.info("✅ MLflow 서버 연결 성공")
            
            run_id = best["run_id"]
            run = self.client.get_run(run_id)
            source = f"{run.info.artifact_uri}/model"
            
            # 등록 모델 보장
            try:
                self.client.get_registered_model(model_name)
                logger.info(f"✅ 기존 등록 모델 발견: {model_name}")
            except Exception:
                self.client.create_registered_model(
                    model_name,
                    description=f"Auto-generated model registry for {model_name}"
                )
                logger.info("✅ 새 등록 모델 생성 완료")
            
            # 버전 생성
            mv = self.client.create_model_version(
                name=model_name, source=source, run_id=run_id
            )
            logger.info(f"✅ 모델 버전 생성 완료: v{mv.version}")
            
            # 태그 저장
            self._set_model_version_tags(model_name, mv.version, best, feature_names, version_tags)
            
            # Production 승격
            self.client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(f"🎉 모델 v{mv.version} Production 승격 완료!")
            
            return mv.version
            
        except Exception as e:
            logger.error(f"❌ Model Registry 등록 실패: {str(e)}")
            logger.error(f"상세 스택트레이스:\n{traceback.format_exc()}")
            return None
    
    def _set_model_version_tags(
        self,
        model_name: str,
        version: str,
        best: Dict[str, Any],
        feature_names: List[str],
        version_tags: Optional[Dict[str, str]]
    ):
        """모델 버전 태그 설정"""
        # Feature names 저장
        feature_names_json = json.dumps(feature_names, ensure_ascii=False)
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="feature_names",
            value=feature_names_json,
        )
        
        # 버전 관리 태그
        task = best.get("task", "classification")
        primary_metric = PRIMARY_METRIC.get(task, "accuracy")
        
        version_info_tags = {
            "created_at": datetime.now().isoformat(),
            "training_run_id": best["run_id"],
            "algorithm": best.get("algorithm", "unknown"),
            "best_metric": json.dumps({
                "name": primary_metric,
                "value": best.get("metrics", {}).get("test", {}).get(primary_metric, 0.0)
            }),
        }
        
        if version_tags:
            version_info_tags.update(version_tags)
        
        for tag_key, tag_value in version_info_tags.items():
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key=tag_key,
                value=str(tag_value),
            )
        
        logger.info(f"✅ 태그 저장 완료: {len(feature_names)}개 피처 + {len(version_info_tags)}개 버전 태그")
    
    def save_manifest(
    self,
    manifest: Dict[str, Any],
    run_id: str
    ) -> bool:
        """Manifest를 MLflow 아티팩트로 저장"""
        logger.info("베스트 run에 manifest 저장 중 (MinIO)...")
        tmp_manifest = None

        try:
            # ✅ MLmodel S3 경로 추가
            run = self.client.get_run(run_id)
            artifact_uri = run.info.artifact_uri  # 예: s3://mlflow-artifacts/model_cls-test3/...
            mlmodel_path = f"{artifact_uri}/model/MLmodel"
            manifest["best_model_s3_path"] = mlmodel_path

            tmp_manifest = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
            with open(tmp_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            self.client.log_artifact(run_id, tmp_manifest, artifact_path="manifest")
            logger.info(f"✅ Manifest 저장 완료 (MinIO): {mlmodel_path}")
            return True

        except Exception as e:
            logger.error(f"⚠️  Manifest 저장 실패: {e}")
            return False

        finally:
            if tmp_manifest and os.path.exists(tmp_manifest):
                os.unlink(tmp_manifest)