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
import numpy as np
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
        self._log_s3_configuration()
        self._setup_experiment()

    def _log_s3_configuration(self) -> None:
        required_keys = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "MLFLOW_S3_ENDPOINT_URL",
        ]
        missing = [key for key in required_keys if not os.environ.get(key)]
        if missing:
            logger.warning(
                "필수 MinIO 환경변수가 누락되어 있습니다: %s", ", ".join(missing)
            )
        else:
            logger.info(
                "MLflow S3 endpoint: %s",
                os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
            )
    
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
        hpo_results: Optional[Dict[str, Any]] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> tuple[str, bool]:
        """모델 학습 결과를 MLflow에 로깅"""
        import pandas as pd
        
        with mlflow.start_run(run_name=run_name):
            run_id = mlflow.active_run().info.run_id
            run_info = mlflow.active_run().info
            logger.info(f"MLflow Run ID: {run_id}")
            logger.info(f"Artifact URI: {run_info.artifact_uri}")
            
            # ✅ X_train을 명확하게 DataFrame으로 변환 (한 번만)
            if input_schema and input_schema.get('feature_names'):
                if not isinstance(X_train, pd.DataFrame):
                    X_train = pd.DataFrame(X_train, columns=input_schema['feature_names'])
                    logger.info(f"✅ X_train을 DataFrame으로 변환 (features: {len(input_schema['feature_names'])})")
                else:
                    # 컬럼명 정렬 확인
                    if list(X_train.columns) != input_schema['feature_names']:
                        X_train = X_train[input_schema['feature_names']]
                        logger.info(f"✅ X_train 컬럼 순서 재정렬")
            
            # 기본 파라미터 로깅
            self._log_basic_params(execution_id, data_source_info, params)
            
            # 스키마 로깅 (아티팩트)
            if input_schema or output_schema:
                self._log_schemas(run_id, input_schema, output_schema)
            
            # 라벨 인코딩 정보 로깅
            if label_encoding_info and label_encoding_info.get("used"):
                self._log_label_encoding(run_id, label_encoding_info)
            
            # HPO 정보 로깅
            if hpo_results:
                self._log_hpo_results(hpo_results)
            
            # 메트릭 로깅
            self._log_metrics(metrics)
            
            # ✅ 모델 저장 (이미 DataFrame으로 변환된 X_train 전달)
            model_saved = self._save_model(
                estimator=estimator,
                X_train=X_train,  # 이미 DataFrame
                y_pred_test=y_pred_test,
                input_schema=input_schema,
                output_schema=output_schema
            )
            
            # ✅ 모델 저장 검증
            if model_saved:
                logger.info("✅ 모델 아티팩트 저장 성공")
                # MLmodel 파일 존재 확인
                try:
                    self.client.download_artifacts(run_id, "model/MLmodel", "/tmp")
                    logger.info("✅ MLmodel 파일 확인됨")
                except Exception as e:
                    logger.error(f"❌ MLmodel 파일 없음: {e}")
                    model_saved = False
            
            return run_id, model_saved

    def _log_schemas(
    self, 
    run_id: str, 
    input_schema: Optional[Dict[str, Any]], 
    output_schema: Optional[Dict[str, Any]]
    ):
        """입출력 스키마 정보를 아티팩트로만 로깅"""
        logger.info("📊 Input/Output 스키마 로깅 중...")
        
        if input_schema:
            schema_path = f"/tmp/{run_id}_input_schema.json"
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(input_schema, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(schema_path, artifact_path="schema")
            os.unlink(schema_path)
            logger.info(f"  ✓ Input schema: {input_schema.get('n_features', 0)} features")
        
        if output_schema:
            schema_path = f"/tmp/{run_id}_output_schema.json"
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(output_schema, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(schema_path, artifact_path="schema")
            os.unlink(schema_path)
            logger.info(f"  ✓ Output schema: {output_schema.get('type', 'unknown')}")
        
        logger.info("✅ 스키마 로깅 완료")

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
    
    def _save_model(self, estimator, X_train, y_pred_test, input_schema=None, output_schema=None):
        """모델 저장 - 로컬 저장 후 MLflow 서버를 통해 업로드"""
        import mlflow  # ← 이거 추가!
        import mlflow.sklearn
        from mlflow.models import infer_signature
        import pandas as pd
        import numpy as np
        import tempfile
        import shutil
        import os
        
        logger.info("============================================================")
        logger.info("모델 아티팩트 저장 시작")
        logger.info("============================================================")
        
        active_run = mlflow.active_run()
        if not active_run:
            logger.error("Active run 없음")
            return False
        
        run_id = active_run.info.run_id
        
        try:
            # Input/Output example
            input_example = X_train.iloc[:min(5, len(X_train))].copy()
            
            if hasattr(estimator, 'predict_proba') and output_schema and output_schema.get('type') == 'classification':
                output_example = estimator.predict_proba(input_example)
                logger.info(f"predict_proba 사용 (shape: {output_example.shape})")
            else:
                output_example = estimator.predict(input_example)
                logger.info(f"predict 사용 (shape: {np.array(output_example).shape})")
            
            # Signature
            signature = infer_signature(input_example, output_example)
            logger.info(f"✅ Signature 생성 성공")
            logger.info(f"   Input schema: {signature.inputs}")
            logger.info(f"   Output schema: {signature.outputs}")
            
            # 임시 디렉토리에 모델 저장
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, "model")
            
            try:
                # 로컬 파일로 저장
                mlflow.sklearn.save_model(
                    sk_model=estimator,
                    path=model_path,
                    signature=signature,
                    input_example=input_example,
                )
                
                logger.info(f"✅ 로컬 저장 완료: {model_path}")
                
                # 저장된 파일 확인
                model_files = []
                for root, dirs, files in os.walk(model_path):
                    for f in files:
                        rel_path = os.path.relpath(os.path.join(root, f), model_path)
                        model_files.append(rel_path)
                
                logger.info(f"저장된 파일: {model_files[:5]}...")  # 처음 5개만 로깅
                
                # MLflow API를 통해 업로드
                logger.info("MLflow 서버를 통해 S3 업로드 중...")
                mlflow.log_artifacts(model_path, artifact_path="model")
                
                logger.info("업로드 완료, 검증 대기 중...")
                
                # S3 동기화 대기
                import time
                time.sleep(5)
                
                # 검증
                artifacts = self.client.list_artifacts(run_id, "model")
                if artifacts:
                    artifact_paths = [a.path for a in artifacts]
                    logger.info(f"✅ 업로드 검증 성공: {artifact_paths}")
                    
                    # MLmodel 파일 확인
                    if any('MLmodel' in p for p in artifact_paths):
                        logger.info("✅ MLmodel 파일 확인됨")
                        logger.info("============================================================")
                        logger.info("✅ 모델 저장 완료")
                        logger.info("============================================================")
                        return True
                    else:
                        logger.error("❌ MLmodel 파일 없음")
                        return False
                else:
                    logger.error("❌ 아티팩트 목록 비어있음")
                    return False
                    
            finally:
                # 임시 디렉토리 삭제
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            logger.error(f"❌ 모델 저장 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_custom_signature(
    self, 
    input_example, 
    output_example, 
    input_schema, 
    output_schema
    ):
        """Custom signature 생성 - MLflow Schema 형식"""
        from mlflow.types.schema import Schema, ColSpec
        from mlflow.models.signature import ModelSignature
        import pandas as pd
        import numpy as np
        
        logger.info("수동 Signature 생성 중...")
        
        # Input Schema 생성
        if isinstance(input_example, pd.DataFrame):
            input_cols = []
            for col in input_example.columns:
                dtype = input_example[col].dtype
                mlflow_dtype = self._map_to_mlflow_dtype(str(dtype))
                input_cols.append(ColSpec(mlflow_dtype, str(col)))
            input_schema_obj = Schema(input_cols)
            logger.info(f"  Input: {len(input_cols)} columns")
        else:
            raise ValueError("input_example must be DataFrame")
        
        # Output Schema 생성
        output_array = np.array(output_example)
        if output_array.ndim == 1:
            from mlflow.types import DataType
            output_schema_obj = Schema([ColSpec(DataType.double, "prediction")])
            logger.info(f"  Output: single column (1D)")
        elif output_array.ndim == 2:
            from mlflow.types import DataType
            n_outputs = output_array.shape[1]
            output_cols = [ColSpec(DataType.double, f"output_{i}") 
                        for i in range(n_outputs)]
            output_schema_obj = Schema(output_cols)
            logger.info(f"  Output: {n_outputs} columns (2D)")
        else:
            raise ValueError(f"Unsupported output dimensions: {output_array.ndim}")
        
        return ModelSignature(inputs=input_schema_obj, outputs=output_schema_obj)


    def _map_to_mlflow_dtype(self, dtype_str):
        """Python dtype을 MLflow dtype으로 매핑"""
        from mlflow.types import DataType
        
        mapping = {
            'int': DataType.long,
            'int64': DataType.long,
            'int32': DataType.integer,
            'float': DataType.double,
            'float64': DataType.double,
            'float32': DataType.float,
            'double': DataType.double,
            'object': DataType.string,
            'str': DataType.string,
            'string': DataType.string,
            'bool': DataType.boolean,
            'datetime64': DataType.datetime,
        }
        
        return mapping.get(dtype_str.lower(), DataType.double)

    def register_best_model(
        self,
        model_name: str,
        best: Dict[str, Any],
        feature_names: List[str],
        version_tags: Optional[Dict[str, str]] = None,
        require_signature: bool = True,
        include_schema_uri_tags: bool = True,  # ✅ 기본값 True로 변경
    ) -> Optional[str]:
        """베스트 모델을 Model Registry에 등록"""
        logger.info("=" * 80)
        logger.info(f"Model Registry 등록: {model_name}")
        logger.info("=" * 80)
        
        try:
            run_id = best["run_id"]
            run = self.client.get_run(run_id)
            
            # ✅ 1. MLmodel 파일 존재 확인
            try:
                temp_dir = tempfile.mkdtemp()
                mlmodel_path = self.client.download_artifacts(
                    run_id, "model/MLmodel", temp_dir
                )
                with open(mlmodel_path, 'r') as f:
                    mlmodel_content = f.read()
                
                logger.info("✅ MLmodel 파일 확인됨")
                
                # ✅ 2. Signature 존재 확인
                has_signature = ("signature:" in mlmodel_content and 
                            "inputs:" in mlmodel_content and 
                            "outputs:" in mlmodel_content)
                
                if not has_signature:
                    msg = "❌ MLmodel에 유효한 signature가 없습니다"
                    if require_signature:
                        logger.error(f"{msg} - 등록 중단")
                        return None
                    else:
                        logger.warning(f"{msg} - 계속 진행")
                else:
                    logger.info("✅ Signature 확인됨")
                    # Signature 내용 로깅
                    import yaml
                    mlmodel_dict = yaml.safe_load(mlmodel_content)
                    if 'signature' in mlmodel_dict:
                        logger.info(f"   Signature: {mlmodel_dict['signature']}")
                
                import shutil
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                logger.error(f"❌ MLmodel 파일 확인 실패: {e}")
                if require_signature:
                    return None
            
            # 3. 등록 모델 생성/확인
            try:
                self.client.get_registered_model(model_name)
                logger.info(f"✅ 기존 모델 사용: {model_name}")
            except Exception:
                self.client.create_registered_model(
                    model_name,
                    description=f"ML model: {model_name}"
                )
                logger.info(f"✅ 새 모델 생성: {model_name}")
            
            # 4. 모델 버전 생성
            source = f"{run.info.artifact_uri}/model"
            mv = self.client.create_model_version(
                name=model_name,
                source=source,
                run_id=run_id
            )
            logger.info(f"✅ 모델 버전 생성: v{mv.version}")
            
            # 5. 태그 설정
            self._set_model_version_tags(
                model_name=model_name,
                version=mv.version,
                best=best,
                feature_names=feature_names,
                version_tags=version_tags,
                include_schema_uri_tags=include_schema_uri_tags
            )
            
            # 6. Production 승격
            self.client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Production",
                archive_existing_versions=True
            )
            logger.info(f"🎉 Production 승격 완료: v{mv.version}")
            logger.info("=" * 80)
            
            return mv.version
            
        except Exception as e:
            logger.error(f"❌ Registry 등록 실패: {e}")
            logger.error(traceback.format_exc())
            return None

    def _has_signature_in_mlmodel(self, run_id: str) -> bool:
        """해당 run의 model/MLmodel 내부에 signature 블록이 존재하는지 간단 검증"""
        import shutil
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="mlmodel_")
            local_path = self.client.download_artifacts(run_id, "model/MLmodel", tmp_dir)
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()
            has_sig = "signature:" in content or ("inputs:" in content and "outputs:" in content)
            logger.info(f"MLmodel signature 존재 여부: {has_sig}")
            return has_sig
        except Exception as e:
            logger.warning(f"MLmodel 서명 확인 실패(run={run_id}): {e}")
            return False
        finally:
            if tmp_dir:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
    
    def _set_model_version_tags(
        self,
        model_name: str,
        version: str,
        best: Dict[str, Any],
        feature_names: List[str],
        version_tags: Optional[Dict[str, str]],
        include_schema_uri_tags: bool = False,
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

        if include_schema_uri_tags:
            run = self.client.get_run(best["run_id"])
            schema_uri = f"{run.info.artifact_uri}/schema"
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key="input_schema_uri",
                value=f"{schema_uri}/input_schema.json",
            )
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key="output_schema_uri",
                value=f"{schema_uri}/output_schema.json",
            )
            logger.info("✅ Schema URI 태그 추가 완료")
    
    def save_manifest(
    self,
    manifest: Dict[str, Any],
    run_id: str
    ) -> bool:
        """Manifest를 MLflow 아티팩트로 저장
        
        Args:
            manifest: 저장할 manifest 딕셔너리
            run_id: MLflow run ID
        
        Returns:
            bool: 저장 성공 여부
        """
        logger.info("베스트 run에 manifest 저장 중 (MinIO)...")
        tmp_manifest = None

        try:
            run = self.client.get_run(run_id)
            artifact_uri = run.info.artifact_uri
            
            # 학습 시 저장된 원본 모델 경로
            model_dir_path = f"{artifact_uri}/model"
            manifest["best_model_s3_path"] = model_dir_path
            manifest["best_model_mlmodel_path"] = f"{model_dir_path}/MLmodel"
            
            # Model Registry 경로 안내 추가
            manifest["model_registry_info"] = {
                "note": (
                    "Model Registry 등록 시 MLflow가 자동으로 models/m-{hash} 경로에 "
                    "복사본을 생성합니다. 원본 모델은 best_model_s3_path에 있습니다."
                ),
                "training_artifact_path": model_dir_path,
                "registry_path_pattern": f"s3://mlflow-artifacts/{manifest.get('model_id', 'model')}/models/m-{{version_hash}}/artifacts/"
            }
            
            logger.info(f"Training artifact path: {model_dir_path}")

            # 직렬화 불가 항목 제거/치환
            sanitized = self._sanitize_for_json(manifest)

            # 임시 파일 생성 및 저장
            tmp_manifest = tempfile.NamedTemporaryFile(
                mode='w',
                delete=False,
                suffix=".json",
                encoding='utf-8'
            )
            json.dump(sanitized, tmp_manifest, ensure_ascii=False, indent=2)
            tmp_manifest.close()
            
            # Run context 안에서 아티팩트 저장
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(tmp_manifest.name, artifact_path="manifest")
            
            logger.info(f"Manifest 저장 완료: {model_dir_path}")
            return True

        except Exception as e:
            logger.error(f"Manifest 저장 실패: {e}")
            logger.error(f"상세:\n{traceback.format_exc()}")
            return False

        finally:
            if tmp_manifest and os.path.exists(tmp_manifest.name):
                try:
                    os.unlink(tmp_manifest.name)
                except Exception:
                    pass


    def _sanitize_for_json(self, obj: Any) -> Any:
        """JSON 직렬화 가능한 형태로 변환
        - dict/list/tuple 재귀 처리
        - numpy/pandas 등 기본 타입으로 변환 시도
        - 직렬화 불가 객체는 문자열(repr)로 치환
        - 특수 키 처리: 'encoder' 키는 제거(아티팩트로 별도 저장됨)
        """
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        # encoder 라벨은 제거
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in {"encoder"}:  # 직렬화 불가, 아티팩트로 이미 저장됨
                    continue
                out[k] = self._sanitize_for_json(v)
            return out
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Series,)):
            return obj.tolist()
        elif isinstance(obj, (pd.DataFrame,)):
            return obj.to_dict(orient="list")
        else:
            try:
                json.dumps(obj)
                return obj
            except Exception:
                return repr(obj)
