# /src/xgenml/core/train_many.py
import os
import uuid
import time
import mlflow
from typing import List, Dict, Any, Optional
from datetime import datetime

from .training.data_loader import DataLoader
from .training.model_trainer import ModelTrainer
from .training.mlflow_manager import MLflowManager
from .model_catalog import (
    get_models_by_task,
    get_primary_metric,
    validate_model_name,
    check_model_requirements,
    get_available_tasks
)
from ..services.hyperparameter_optimization import HyperparameterOptimizer
from ..utils.logger_config import setup_logger
from ..utils.file_utils import TempDirectoryManager

logger = setup_logger(__name__)


def train_from_hf(
    model_id: str,
    task: str,
    # HuggingFace 관련
    hf_repo: Optional[str] = None,
    hf_filename: Optional[str] = None,
    # /src/xgenml/core/train_many.py (계속)
    hf_revision: Optional[str] = None,
    # MLflow 관련
    use_mlflow_dataset: bool = False,
    mlflow_run_id: Optional[str] = None,
    mlflow_artifact_path: str = "dataset",
    # 데이터 관련
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    use_cv: bool = False,
    cv_folds: int = 5,
    # MLflow 실험 관련
    mlflow_experiment: Optional[str] = None,
    artifact_base_uri: Optional[str] = None,
    storage_ctor_kwargs: Optional[Dict[str, Any]] = None,
    # HPO 관련
    hpo_config: Optional[Dict[str, Any]] = None,
    # 태스크별 설정
    task_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    전처리 완료 데이터 가정 / 여러 모델 일괄 학습·평가·로깅
    
    Args:
        model_id: 모델 식별자
        task: 태스크 타입 (classification, regression, timeseries, anomaly_detection, clustering)
        hf_repo: HuggingFace 레포지토리
        hf_filename: HuggingFace 파일명
        hf_revision: HuggingFace 리비전
        use_mlflow_dataset: MLflow 데이터셋 사용 여부
        mlflow_run_id: MLflow run ID (use_mlflow_dataset=True인 경우)
        mlflow_artifact_path: MLflow 아티팩트 경로
        target_column: 타겟 컬럼명 (clustering은 선택사항)
        feature_columns: 사용할 피처 컬럼 (None이면 자동 선택)
        model_names: 학습할 모델 목록 (None이면 태스크별 기본 모델 사용)
        overrides: 모델별 파라미터 오버라이드
        test_size: 테스트 세트 비율
        val_size: 검증 세트 비율
        random_state: 난수 시드
        use_cv: 교차 검증 사용 여부
        cv_folds: 교차 검증 폴드 수
        mlflow_experiment: MLflow 실험명
        artifact_base_uri: 외부 스토리지 URI (선택사항)
        storage_ctor_kwargs: 스토리지 생성자 kwargs
        hpo_config: 하이퍼파라미터 최적화 설정
        task_config: 태스크별 설정
            - timeseries: {"lookback_window": 10, "forecast_horizon": 1, "time_column": "date"}
            - anomaly_detection: {"contamination": 0.1}
            - clustering: {"n_clusters": 3}
    
    Returns:
        학습 결과 딕셔너리
    """
    training_start_time = time.time()
    execution_id = f"{model_id}_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    # 환경 설정
    USE_UNIQUE_PATHS = os.getenv("MLFLOW_USE_UNIQUE_PATHS", "true").lower() == "true"
    CLEANUP_TEMP_FILES = os.getenv("MLFLOW_CLEANUP_TEMP", "true").lower() == "true"
    temp_manager = TempDirectoryManager(cleanup_enabled=CLEANUP_TEMP_FILES)
    
    logger.info("=" * 80)
    logger.info("🚀 모델 학습 시작")
    logger.info("=" * 80)
    logger.info(f"🆔 고유 실행 ID: {execution_id}")
    
    try:
        # ========================================
        # 1. 태스크 검증
        # ========================================
        available_tasks = get_available_tasks()
        if task not in available_tasks:
            raise ValueError(
                f"Unknown task: {task}\n"
                f"Available tasks: {available_tasks}"
            )
        
        logger.info(f"\n📋 학습 설정:")
        logger.info(f"  - Model ID: {model_id}")
        logger.info(f"  - Execution ID: {execution_id}")
        logger.info(f"  - Task: {task}")
        logger.info(f"  - Data Source: {'MLflow' if use_mlflow_dataset else 'HuggingFace'}")
        
        if use_mlflow_dataset:
            logger.info(f"  - MLflow Run ID: {mlflow_run_id}")
            logger.info(f"  - Artifact Path: {mlflow_artifact_path}")
        else:
            logger.info(f"  - HF Repo: {hf_repo}")
            logger.info(f"  - HF Filename: {hf_filename}")
            logger.info(f"  - HF Revision: {hf_revision or 'default'}")
        
        logger.info(f"  - Target Column: {target_column or 'N/A (unsupervised)'}")
        logger.info(f"  - Feature Columns: {feature_columns or 'auto (all except target)'}")
        logger.info(f"  - Test Size: {test_size}")
        logger.info(f"  - Validation Size: {val_size}")
        logger.info(f"  - Use CV: {use_cv} ({cv_folds} folds)")
        logger.info(f"  - Random State: {random_state}")
        logger.info(f"  - Use Unique Paths: {USE_UNIQUE_PATHS}")
        
        if task_config:
            logger.info(f"  - Task Config: {task_config}")
        if overrides:
            logger.info(f"  - Overrides: {overrides}")
        if hpo_config:
            logger.info(f"  - HPO Config: {hpo_config}")
        
        # ========================================
        # 2. 모델 목록 검증
        # ========================================
        if not model_names:
            # 태스크별 기본 모델 사용
            available_models = [m["name"] for m in get_models_by_task(task)]
            model_names = available_models[:3]  # 상위 3개
            logger.info(f"\n모델 미지정, 태스크 '{task}' 기본 모델 사용: {model_names}")
        
        # 모델 유효성 검증 및 필수 패키지 확인
        validated_models = []
        for name in model_names:
            if not validate_model_name(task, name):
                logger.warning(f"⚠️  '{name}'은 태스크 '{task}'에 사용할 수 없습니다. 건너뜁니다.")
                continue
            
            # 패키지 요구사항 확인
            req_check = check_model_requirements(task, name)
            if not req_check["available"]:
                missing = req_check['missing_packages']
                logger.warning(
                    f"⚠️  '{name}' 필요 패키지 누락: {missing}. 건너뜁니다.\n"
                    f"    설치: pip install {' '.join(missing)}"
                )
                continue
            
            validated_models.append(name)
        
        if not validated_models:
            raise ValueError(
                f"사용 가능한 모델이 없습니다.\n"
                f"태스크: {task}\n"
                f"요청 모델: {model_names}\n"
                f"필요한 패키지를 설치했는지 확인하세요."
            )
        
        model_names = validated_models
        logger.info(f"✅ 검증된 모델: {model_names}")
        
        # Primary metric 자동 설정
        best_key = get_primary_metric(task)
        logger.info(f"평가 지표: {best_key}")
        
        # ========================================
        # 3. MLflow 설정
        # ========================================
        _setup_mlflow()
        
        # 실험 이름 및 MLflow 매니저 생성
        experiment_name = _get_experiment_name(
            mlflow_experiment, model_id, execution_id, USE_UNIQUE_PATHS
        )
        mlflow_manager = MLflowManager(experiment_name)
        
        # ========================================
        # 4. 데이터 로드
        # ========================================
        data_loader = DataLoader(task=task, task_config=task_config)
        df, data_source_info = data_loader.load_data(
            use_mlflow_dataset=use_mlflow_dataset,
            mlflow_run_id=mlflow_run_id,
            mlflow_artifact_path=mlflow_artifact_path,
            hf_repo=hf_repo,
            hf_filename=hf_filename,
            hf_revision=hf_revision
        )
        
        # ========================================
        # 5. 피처 준비
        # ========================================
        X, y, feature_names, task_metadata = data_loader.prepare_features(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns
        )
        
        # ========================================
        # 6. 데이터 분할
        # ========================================
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(
            X=X, y=y,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )
        
        # 라벨 인코딩 정보
        label_encoding_info = data_loader.get_label_encoding_info()
        
        # ========================================
        # 7. HPO 설정
        # ========================================
        optimizer = None
        use_hpo = hpo_config and hpo_config.get('enable_hpo', False)
        if use_hpo:
            logger.info("\n🎯 하이퍼파라미터 최적화 활성화")
            optimizer = HyperparameterOptimizer(
                n_trials=hpo_config.get('n_trials', 50),
                timeout=hpo_config.get('timeout_minutes', None) * 60 
                    if hpo_config.get('timeout_minutes') else None
            )
            logger.info(f"  - Trials: {hpo_config.get('n_trials', 50)}")
            if hpo_config.get('timeout_minutes'):
                logger.info(f"  - Timeout: {hpo_config.get('timeout_minutes')} minutes")
        
        # ========================================
        # 8. 모델 학습
        # ========================================
        trainer = ModelTrainer(task, mlflow_manager, USE_UNIQUE_PATHS)
        if optimizer:
            trainer.set_optimizer(optimizer)
        
        results = []
        best = None
        best_score = -1e18 if task == "regression" else -1.0
        
        logger.info(f"\n🤖 모델 학습 시작 ({len(model_names)}개 모델)")
        logger.info(f"평가 지표: {best_key}")
        
        for idx, name in enumerate(model_names, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"[{idx}/{len(model_names)}] {name} 학습 중...")
            logger.info(f"{'=' * 60}")
            
            try:
                summary = trainer.train_model(
                    model_name=name,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test,
                    execution_id=execution_id,
                    data_source_info=data_source_info,
                    label_encoding_info=label_encoding_info,
                    use_cv=use_cv,
                    cv_folds=cv_folds,
                    overrides=overrides,
                    hpo_config=hpo_config
                )
                
                results.append(summary)
                
                # 베스트 모델 업데이트
                score = summary["metrics"]["test"][best_key]
                if score > best_score:
                    best_score = score
                    best = summary
                    hpo_info = f" (HPO)" if summary.get("hpo_used") else ""
                    logger.info(f"🏆 새로운 베스트 모델: {name}{hpo_info} ({best_key}={score:.4f})")
                
            except Exception as e:
                logger.error(f"❌ {name} 모델 학습 실패: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # ========================================
        # 9. 결과 검증
        # ========================================
        if not best:
            raise RuntimeError("모든 모델 학습이 실패했습니다")
        
        hpo_info = f" (HPO)" if best.get("hpo_used") else ""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"🏆 최고 성능 모델: {best['algorithm']}{hpo_info}")
        logger.info(f"Run ID: {best['run_id']}")
        logger.info(f"최고 {best_key}: {best_score:.4f}")
        logger.info(f"{'=' * 80}")
        
        if best.get("hpo_used") and best.get("hpo_results"):
            logger.info(f"HPO 최적 파라미터: {best['final_params']}")
        
        # ========================================
        # 10. Manifest 생성 및 저장
        # ========================================
        manifest = _create_manifest(
            results=results,
            best=best,
            feature_names=feature_names,
            model_id=model_id,
            execution_id=execution_id,
            task=task,
            training_start_time=training_start_time,
            data_source_info=data_source_info,
            use_hpo=use_hpo,
            hpo_config=hpo_config,
            label_encoding_info=label_encoding_info,
            task_metadata=task_metadata,
            USE_UNIQUE_PATHS=USE_UNIQUE_PATHS,
            CLEANUP_TEMP_FILES=CLEANUP_TEMP_FILES
        )
        
        mlflow_manager.save_manifest(manifest, best["run_id"])
        
        # ========================================
        # 11. Model Registry 등록
        # ========================================
        production_version = None
        version_tags = {}
        
        if os.getenv("ENABLE_MODEL_REGISTRY", "true").lower() == "true" and best.get("model_saved", False):
            version_tags = _create_version_tags(
                execution_id=execution_id,
                data_source_info=data_source_info,
                use_mlflow_dataset=use_mlflow_dataset,
                mlflow_run_id=mlflow_run_id,
                hf_repo=hf_repo,
                best=best,
                label_encoding_info=label_encoding_info,
                task_metadata=task_metadata,
                USE_UNIQUE_PATHS=USE_UNIQUE_PATHS
            )
            
            production_version = mlflow_manager.register_best_model(
                model_name=model_id,
                best=best,
                feature_names=feature_names,
                version_tags=version_tags
            )
        elif not best.get("model_saved", False):
            logger.warning("⚠️  베스트 모델의 아티팩트가 저장되지 않아 Model Registry 등록을 건너뜁니다")
        else:
            logger.info("ℹ️  Model Registry 비활성화됨 (ENABLE_MODEL_REGISTRY=false)")
        
        # ========================================
        # 12. 학습 완료 요약
        # ========================================
        _log_training_summary(
            execution_id=execution_id,
            num_results=len(results),
            best=best,
            data_source_info=data_source_info,
            use_hpo=use_hpo,
            results=results,
            label_encoding_info=label_encoding_info,
            production_version=production_version,
            model_id=model_id,
            training_start_time=training_start_time,
            task=task,
            task_metadata=task_metadata
        )
        
        # ========================================
        # 13. 반환 데이터 생성
        # ========================================
        return _create_return_data(
            results=results,
            best=best,
            execution_id=execution_id,
            model_id=model_id,
            production_version=production_version,
            version_tags=version_tags,
            training_start_time=training_start_time,
            feature_names=feature_names,
            data_source_info=data_source_info,
            use_hpo=use_hpo,
            hpo_config=hpo_config,
            label_encoding_info=label_encoding_info,
            task_metadata=task_metadata,
            USE_UNIQUE_PATHS=USE_UNIQUE_PATHS,
            CLEANUP_TEMP_FILES=CLEANUP_TEMP_FILES,
            task=task
        )
        
    except Exception as e:
        total_duration = time.time() - training_start_time
        logger.error(f"\n{'=' * 80}")
        logger.error(f"💥 학습 실패! ({total_duration:.2f}초)")
        logger.error(f"{'=' * 80}")
        logger.error(f"실행 ID: {execution_id}")
        logger.error(f"에러: {str(e)}")
        logger.error(f"에러 타입: {type(e).__name__}")
        import traceback
        logger.error(f"전체 스택트레이스:\n{traceback.format_exc()}")
        logger.error(f"{'=' * 80}")
        raise
    
    finally:
        # 임시 디렉토리 정리
        temp_manager.cleanup()


# ============================================================================
# Helper Functions
# ============================================================================

def _setup_mlflow():
    """MLflow 기본 설정"""
    logger.info("\n🔧 MLflow 설정 중...")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        logger.warning("⚠️  MLFLOW_TRACKING_URI 환경변수가 설정되지 않음")
        raise ValueError("MLFLOW_TRACKING_URI 환경변수가 필요합니다")
    
    logger.info(f"MLflow Tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "s3://mlflow-artifacts")
    s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    
    logger.info(f"Artifact Root: {artifact_root}")
    if s3_endpoint:
        logger.info(f"S3 Endpoint (MinIO): {s3_endpoint}")
    
    logger.info("✅ MLflow 설정 완료")


def _get_experiment_name(
    mlflow_experiment: Optional[str],
    model_id: str,
    execution_id: str,
    use_unique_paths: bool
) -> str:
    """실험 이름 생성"""
    if use_unique_paths:
        return mlflow_experiment or f"model_{execution_id}"
    else:
        return mlflow_experiment or f"model_{model_id}"


def _create_manifest(
    results, best, feature_names, model_id, execution_id, task,
    training_start_time, data_source_info, use_hpo, hpo_config,
    label_encoding_info, task_metadata, USE_UNIQUE_PATHS, CLEANUP_TEMP_FILES
) -> Dict[str, Any]:
    """Manifest 생성"""
    logger.info("\n📄 Manifest 생성 중...")
    
    # JSON 직렬화 가능하도록 처리
    serializable_results = []
    for result in results:
        result_copy = result.copy()
        if result_copy.get('hpo_results'):
            hpo_copy = result_copy['hpo_results'].copy()
            hpo_copy.pop('study', None)  # Optuna Study 객체 제거
            result_copy['hpo_results'] = hpo_copy
        serializable_results.append(result_copy)
    
    best_copy = best.copy()
    if best_copy.get('hpo_results'):
        hpo_copy = best_copy['hpo_results'].copy()
        hpo_copy.pop('study', None)
        best_copy['hpo_results'] = hpo_copy
    
    manifest = {
        "results": serializable_results,
        "best": best_copy,
        "feature_names": feature_names,
        "model_id": model_id,
        "execution_id": execution_id,
        "task": task,
        "training_timestamp": datetime.now().isoformat(),
        "training_duration": time.time() - training_start_time,
        "data_source": data_source_info,
        "hpo_summary": {
            "enabled": use_hpo,
            "models_optimized": sum(1 for r in results if r.get("hpo_used", False)) if use_hpo else 0,
            "config": hpo_config if use_hpo else None
        },
        "label_encoding": label_encoding_info,
        "task_metadata": task_metadata,
        "version_info": {
            "unique_execution": USE_UNIQUE_PATHS,
            "cleanup_enabled": CLEANUP_TEMP_FILES,
        }
    }
    
    return manifest


def _create_version_tags(
    execution_id, data_source_info, use_mlflow_dataset,
    mlflow_run_id, hf_repo, best, label_encoding_info,
    task_metadata, USE_UNIQUE_PATHS
) -> Dict[str, str]:
    """모델 버전 태그 생성"""
    version_tags = {
        "execution_id": execution_id,
        "training_timestamp": datetime.now().isoformat(),
        "data_source": data_source_info["source_type"],
        "unique_run": str(USE_UNIQUE_PATHS),
    }
    
    if use_mlflow_dataset:
        version_tags["source_mlflow_run_id"] = mlflow_run_id
    else:
        version_tags["source_hf_repo"] = hf_repo
    
    if best.get("hpo_used"):
        version_tags["hpo_optimized"] = "true"
        version_tags["hpo_trials"] = str(best.get("hpo_results", {}).get("n_trials", 0))
    
    if label_encoding_info and label_encoding_info.get("used"):
        version_tags["label_encoded"] = "true"
        version_tags["num_classes"] = str(len(label_encoding_info.get("original_classes", [])))
    
    # 태스크 메타데이터 추가
    if task_metadata:
        for key, value in task_metadata.items():
            if key not in ["encoder", "label_mapping"]:  # 직렬화 불가 객체 제외
                version_tags[f"task_{key}"] = str(value)
    
    return version_tags


def _log_training_summary(
    execution_id, num_results, best, data_source_info,
    use_hpo, results, label_encoding_info, production_version,
    model_id, training_start_time, task, task_metadata
):
    """학습 완료 요약 로깅"""
    total_duration = time.time() - training_start_time
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"🎉 전체 학습 완료! ({total_duration:.2f}초)")
    logger.info(f"{'=' * 80}")
    logger.info(f"실행 ID: {execution_id}")
    logger.info(f"태스크: {task}")
    logger.info(f"학습된 모델 수: {num_results}")
    logger.info(f"베스트 모델: {best['algorithm']}")
    logger.info(f"데이터 소스: {data_source_info['source_type']}")
    
    if use_hpo:
        hpo_count = sum(1 for r in results if r.get("hpo_used", False))
        logger.info(f"HPO 적용된 모델 수: {hpo_count}/{num_results}")
    
    if label_encoding_info and label_encoding_info.get("used"):
        original_classes = label_encoding_info.get("original_classes", [])
        logger.info(f"라벨 인코딩 적용: {original_classes} -> {list(range(len(original_classes)))}")
    
    # 태스크별 메타데이터 로깅
    if task == "timeseries":
        logger.info(f"시계열 설정: lookback={task_metadata.get('lookback_window')}, "
                   f"horizon={task_metadata.get('forecast_horizon')}")
    elif task == "anomaly_detection":
        logger.info(f"이상 탐지 모드: {'지도' if task_metadata.get('is_supervised') else '비지도'}")
        logger.info(f"Contamination: {task_metadata.get('contamination')}")
    elif task == "clustering":
        logger.info(f"목표 클러스터 수: {task_metadata.get('n_clusters')}")
    
    if production_version:
        logger.info(f"Model Registry: {model_id} v{production_version} (Production)")
    
    logger.info(f"{'=' * 80}")


def _create_return_data(
    results, best, execution_id, model_id, production_version,
    version_tags, training_start_time, feature_names,
    data_source_info, use_hpo, hpo_config, label_encoding_info,
    task_metadata, USE_UNIQUE_PATHS, CLEANUP_TEMP_FILES, task
) -> Dict[str, Any]:
    """반환 데이터 생성"""
    return {
        "results": results,
        "best": best,
        "execution_id": execution_id,
        "runs_manifest_uri": "",
        "registry": {
            "model_name": model_id,
            "production_version": production_version or "",
            "version_tags": version_tags if production_version else {}
        },
        "training_duration": time.time() - training_start_time,
        "feature_names": feature_names,
        "training_timestamp": datetime.now().isoformat(),
        "data_source": data_source_info,
        "task": task,
        "task_metadata": task_metadata,
        "hpo_summary": {
            "enabled": use_hpo,
            "models_optimized": sum(1 for r in results if r.get("hpo_used", False)) if use_hpo else 0,
            "config": hpo_config if use_hpo else None
        },
        "label_encoding": label_encoding_info,
        "version_info": {
            "unique_execution": USE_UNIQUE_PATHS,
            "cleanup_enabled": CLEANUP_TEMP_FILES,
        }
    }