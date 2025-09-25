# /src/xgenml/core/train_many.py
import os
import uuid
import json
import tempfile
import joblib
import mlflow
import mlflow.sklearn
import logging
import traceback
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from mlflow.tracking import MlflowClient

from .data_service import HFDataService
from .model_provider import create_estimator
from .metrics import classification_metrics, regression_metrics
from .storage import StorageClient
from ..services.hyperparameter_optimization import HyperparameterOptimizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/training.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

PRIMARY_METRIC = {"classification": "accuracy", "regression": "r2"}


def create_safe_directory(base_path: str, identifier: str) -> str:
    """
    안전하게 고유한 디렉토리를 생성합니다.
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex[:8]
    safe_dir = f"{base_path}/{identifier}_{timestamp}_{unique_id}"
    
    try:
        os.makedirs(safe_dir, exist_ok=True, mode=0o755)
        # 권한 확인
        test_file = os.path.join(safe_dir, "test_write")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        logger.info(f"✅ 안전한 디렉토리 생성: {safe_dir}")
        return safe_dir
    except Exception as e:
        logger.error(f"❌ 디렉토리 생성 실패: {e}")
        # 폴백: 완전히 임시 디렉토리 사용
        fallback_dir = tempfile.mkdtemp(prefix=f"mlflow_{identifier}_")
        logger.info(f"🔄 폴백 디렉토리 사용: {fallback_dir}")
        return fallback_dir


def _register_best_to_mlflow(model_name: str, best: dict, feature_names: list[str], version_tags: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    best['run_id']가 가리키는 run의 'model' 아티팩트를
    MLflow Model Registry에 새 버전으로 등록하고 Production으로 승격.
    feature_names는 모델 버전 태그로 저장.
    버전별 태그 추가 지원.
    반환: 등록된 모델 버전 문자열(예: '3'), 실패 시 None
    """
    logger.info(f"🏷️  Model Registry 등록 시작: {model_name}")
    
    try:
        client = MlflowClient()
        
        # MLflow 서버 연결 테스트
        logger.info("MLflow 서버 연결 테스트...")
        try:
            client.search_experiments(max_results=1)
            logger.info("✅ MLflow 서버 연결 성공")
        except Exception as conn_error:
            logger.error(f"❌ MLflow 서버 연결 실패: {conn_error}")
            return None
        
        run_id = best["run_id"]
        logger.info(f"Run ID: {run_id}")

        # run.artifact_uri + '/model' 이 모델 소스 경로
        logger.info("Run 정보 조회 중...")
        run = client.get_run(run_id)
        source = f"{run.info.artifact_uri}/model"
        logger.info(f"모델 소스 경로: {source}")

        # 등록 모델 보장
        logger.info(f"등록 모델 확인/생성: {model_name}")
        try:
            registered_model = client.get_registered_model(model_name)
            logger.info(f"✅ 기존 등록 모델 발견: {model_name}")
            logger.info(f"모델 설명: {registered_model.description or 'None'}")
        except Exception as e:
            logger.info(f"새 등록 모델 생성: {model_name}")
            client.create_registered_model(
                model_name,
                description=f"Auto-generated model registry for {model_name}"
            )
            logger.info("✅ 새 등록 모델 생성 완료")

        # 버전 생성
        logger.info("모델 버전 생성 중...")
        mv = client.create_model_version(name=model_name, source=source, run_id=run_id)
        logger.info(f"✅ 모델 버전 생성 완료: v{mv.version}")

        # 태그로 feature_names 저장
        logger.info("피처 이름 태그 저장 중...")
        feature_names_json = json.dumps(feature_names, ensure_ascii=False)
        client.set_model_version_tag(
            name=model_name,
            version=mv.version,
            key="feature_names",
            value=feature_names_json,
        )
        
        # 버전 관리 태그들 추가
        version_info_tags = {
            "created_at": datetime.now().isoformat(),
            "training_run_id": run_id,
            "algorithm": best.get("algorithm", "unknown"),
            "best_metric": json.dumps({
                "name": PRIMARY_METRIC.get(best.get("task", "classification"), "accuracy"),
                "value": best.get("metrics", {}).get("test", {}).get(PRIMARY_METRIC.get(best.get("task", "classification"), "accuracy"), 0.0)
            }),
        }
        
        # 사용자 정의 태그 추가
        if version_tags:
            version_info_tags.update(version_tags)
        
        # 모든 태그 설정
        for tag_key, tag_value in version_info_tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=mv.version,
                key=tag_key,
                value=str(tag_value),
            )
        
        logger.info(f"✅ 태그 저장 완료: {len(feature_names)}개 피처 + {len(version_info_tags)}개 버전 태그")

        # 승격(Production), 기존 Production은 archive
        logger.info(f"모델 v{mv.version}을 Production으로 승격 중...")
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"🎉 모델 v{mv.version} Production 승격 완료!")
        
        return mv.version
        
    except Exception as e:
        logger.error(f"❌ Model Registry 등록 실패: {str(e)}")
        logger.error(f"에러 타입: {type(e).__name__}")
        logger.error(f"상세 스택트레이스:\n{traceback.format_exc()}")
        return None


def train_from_hf(
    model_id: str,
    task: str,
    hf_repo: str,
    hf_filename: str,
    hf_revision: Optional[str],
    target_column: str,
    feature_columns: Optional[List[str]],
    model_names: List[str],
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    use_cv: bool = False,
    cv_folds: int = 5,
    mlflow_experiment: Optional[str] = None,
    artifact_base_uri: Optional[str] = None,
    storage_ctor_kwargs: Optional[Dict[str, Any]] = None,
    hpo_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    전처리 완료 데이터 가정 / 여러 모델 일괄 학습·평가·로깅.
    - 고유한 실행 ID로 충돌 방지
    - 베스트 run을 MLflow Model Registry에 등록하고 Production으로 승격
    - feature_names를 모델 버전 태그로 저장
    - run 아티팩트로 manifest.json도 남김(백업/참조용)
    - (선택) 외부 스토리지에도 manifest 업로드
    - 하이퍼파라미터 최적화 지원
    - 자동 라벨 인코딩 지원
    - 개선된 버전 관리 및 태깅
    """
    training_start_time = time.time()
    
    # 고유한 실행 ID 생성 (충돌 방지)
    execution_id = f"{model_id}_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    logger.info("🚀 모델 학습 시작")
    logger.info("=" * 80)
    logger.info(f"🆔 고유 실행 ID: {execution_id}")
    
    # 환경 변수로 동작 제어
    USE_UNIQUE_PATHS = os.getenv("MLFLOW_USE_UNIQUE_PATHS", "true").lower() == "true"
    CLEANUP_TEMP_FILES = os.getenv("MLFLOW_CLEANUP_TEMP", "true").lower() == "true"
    
    # 임시 디렉토리 관리
    temp_directories = []
    
    try:
        # 입력 파라미터 로깅
        logger.info(f"📋 학습 설정:")
        logger.info(f"  - Model ID: {model_id}")
        logger.info(f"  - Execution ID: {execution_id}")
        logger.info(f"  - Task: {task}")
        logger.info(f"  - HF Repo: {hf_repo}")
        logger.info(f"  - HF Filename: {hf_filename}")
        logger.info(f"  - HF Revision: {hf_revision or 'default'}")
        logger.info(f"  - Target Column: {target_column}")
        logger.info(f"  - Feature Columns: {feature_columns or 'auto (all except target)'}")
        logger.info(f"  - Model Names: {model_names}")
        logger.info(f"  - Test Size: {test_size}")
        logger.info(f"  - Validation Size: {val_size}")
        logger.info(f"  - Use CV: {use_cv} ({cv_folds} folds)")
        logger.info(f"  - Random State: {random_state}")
        logger.info(f"  - Use Unique Paths: {USE_UNIQUE_PATHS}")
        if overrides:
            logger.info(f"  - Overrides: {overrides}")
        if hpo_config:
            logger.info(f"  - HPO Config: {hpo_config}")
        
        # MLflow 설정
        logger.info("\n🔧 MLflow 설정 중...")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        if not tracking_uri:
            logger.warning("⚠️  MLFLOW_TRACKING_URI 환경변수가 설정되지 않음")
            raise ValueError("MLFLOW_TRACKING_URI 환경변수가 필요합니다")
        
        logger.info(f"MLflow Tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        # 아티팩트 경로 설정 - 충돌 방지
        logger.info("MLflow 아티팩트 설정 중...")
        
        if tracking_uri.startswith("http"):
            # 원격 서버 사용
            artifact_uri = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow-artifacts/artifacts"
            os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = artifact_uri
            logger.info(f"원격 아티팩트 URI: {artifact_uri}")
        else:
            # 로컬 사용시 고유 경로 (충돌 방지)
            if USE_UNIQUE_PATHS:
                safe_local_path = create_safe_directory("/app/mlruns", execution_id)
                temp_directories.append(safe_local_path)
                os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = safe_local_path
                logger.info(f"로컬 고유 아티팩트 경로: {safe_local_path}")
            else:
                artifact_uri = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow-artifacts/artifacts"
                os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = artifact_uri
                logger.info(f"기본 아티팩트 URI: {artifact_uri}")
        
        # 실험 이름도 고유하게 (선택적)
        if USE_UNIQUE_PATHS:
            experiment_name = mlflow_experiment or f"model_{execution_id}"
        else:
            experiment_name = mlflow_experiment or f"model_{model_id}"
        
        logger.info(f"실험 이름: {experiment_name}")
        
        # 실험 생성/설정
        try:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f"mlflow-artifacts:/{experiment_name}"
            )
            logger.info(f"✅ 새 실험 생성: {experiment_name}")
        except Exception:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            logger.info(f"✅ 기존 실험 사용: {experiment_name}")
        
        mlflow.set_experiment(experiment_id=experiment_id)
        logger.info("✅ MLflow 설정 완료")

        # 데이터 로드
        logger.info(f"\n📊 데이터 로드 중: {hf_repo}/{hf_filename}")
        load_start_time = time.time()
        
        try:
            df = HFDataService().load_dataset_data(hf_repo, hf_filename, hf_revision)
            load_duration = time.time() - load_start_time
            logger.info(f"✅ 데이터 로드 완료 ({load_duration:.2f}초)")
            logger.info(f"데이터셋 형태: {df.shape}")
            logger.info(f"컬럼: {df.columns.tolist()}")
            logger.info(f"데이터 타입:\n{df.dtypes}")
            
            # 결측치 확인
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                logger.warning(f"⚠️  결측치 발견:\n{null_counts[null_counts > 0]}")
            else:
                logger.info("✅ 결측치 없음")
                
        except Exception as e:
            logger.error(f"❌ 데이터 로드 실패: {str(e)}")
            logger.error(f"스택트레이스:\n{traceback.format_exc()}")
            raise

        # 타겟 컬럼 확인
        if target_column not in df.columns:
            error_msg = f"타겟 컬럼 '{target_column}'이 데이터셋에 없습니다. 사용 가능한 컬럼: {df.columns.tolist()}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        logger.info(f"타겟 컬럼 '{target_column}' 정보:")
        logger.info(f"  - 데이터 타입: {df[target_column].dtype}")
        logger.info(f"  - 고유값 개수: {df[target_column].nunique()}")
        logger.info(f"  - 고유값: {df[target_column].unique()[:10]}")
        
        # 피처 설정
        if feature_columns:
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                error_msg = f"피처 컬럼이 데이터셋에 없습니다: {missing_features}"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            X = df[feature_columns]
            logger.info(f"✅ 지정된 피처 컬럼 사용: {len(feature_columns)}개")
        else:
            X = df.drop(columns=[target_column])
            logger.info(f"✅ 자동 피처 선택: 타겟 제외 {X.shape[1]}개 컬럼")
        
        y = df[target_column]
        logger.info(f"최종 피처 형태: {X.shape}")
        logger.info(f"타겟 형태: {y.shape}")

        # 라벨 인코딩 처리 (분류 태스크이고 타겟이 문자열인 경우)
        label_encoder = None
        original_classes = None
        label_mapping = None
        
        if task == "classification" and y.dtype == 'object':
            logger.info(f"\n🏷️  타겟 라벨이 문자열입니다. 자동으로 숫자로 변환합니다.")
            original_classes = y.unique().tolist()
            logger.info(f"원본 라벨: {original_classes}")
            
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # 라벨 매핑 정보
            label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
            logger.info(f"변환된 라벨: {np.unique(y_encoded)}")
            logger.info(f"라벨 매핑: {label_mapping}")
            
            # y 값들을 인코딩된 값으로 교체
            y = y_encoded
            logger.info("✅ 라벨 인코딩 완료")

        # 데이터 분할
        logger.info(f"\n🔀 데이터 분할 (test_size={test_size}, val_size={val_size})")
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if task == "classification" else None,
            )
            logger.info(f"Train+Val: {X_temp.shape}, Test: {X_test.shape}")
            
            if val_size > 0:
                val_ratio = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=val_ratio,
                    random_state=random_state,
                    stratify=y_temp if task == "classification" else None,
                )
                logger.info(f"Train: {X_train.shape}, Validation: {X_val.shape}")
            else:
                X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
                logger.info(f"Train: {X_train.shape}, Validation: 없음")
            
            logger.info("✅ 데이터 분할 완료")
            
        except Exception as e:
            logger.error(f"❌ 데이터 분할 실패: {str(e)}")
            raise

        # 학습에 사용된 피처 이름
        used_feature_names = X.columns.tolist()
        logger.info(f"사용된 피처 이름: {used_feature_names}")

        # HPO 설정
        use_hpo = hpo_config and hpo_config.get('enable_hpo', False)
        optimizer = None
        if use_hpo:
            logger.info("\n🎯 하이퍼파라미터 최적화 활성화")
            optimizer = HyperparameterOptimizer(
                n_trials=hpo_config.get('n_trials', 50),
                timeout=hpo_config.get('timeout_minutes', None) * 60 if hpo_config.get('timeout_minutes') else None
            )
            logger.info(f"  - Trials: {hpo_config.get('n_trials', 50)}")
            logger.info(f"  - Timeout: {hpo_config.get('timeout_minutes', 'None')} minutes")

        results: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        best_key = PRIMARY_METRIC[task]
        best_score = -1e18 if task == "regression" else -1.0
        
        logger.info(f"\n🤖 모델 학습 시작 ({len(model_names)}개 모델)")
        logger.info(f"평가 지표: {best_key}")

        # 여러 알고리즘 반복
        for idx, name in enumerate(model_names, 1):
            model_start_time = time.time()
            logger.info(f"\n[{idx}/{len(model_names)}] {name} 학습 중...")
            
            try:
                # 고유한 run 이름 생성 (충돌 방지)
                if USE_UNIQUE_PATHS:
                    run_name = f"{task}:{name}:{execution_id[-8:]}"
                else:
                    run_name = f"{task}:{name}"
                
                with mlflow.start_run(run_name=run_name):
                    run_id = mlflow.active_run().info.run_id
                    logger.info(f"MLflow Run ID: {run_id}")
                    
                    # 실행 ID를 MLflow에도 저장
                    mlflow.log_param("execution_id", execution_id)
                    mlflow.log_param("unique_run", USE_UNIQUE_PATHS)
                    
                    # 라벨 인코딩 정보를 MLflow에 저장
                    if label_encoder is not None:
                        mlflow.log_param("label_encoded", True)
                        mlflow.log_param("original_classes", json.dumps(original_classes, ensure_ascii=False))
                        mlflow.log_param("label_mapping", json.dumps(label_mapping, ensure_ascii=False))
                        
                        # 라벨 인코더 자체도 아티팩트로 저장
                        label_encoder_path = f"/tmp/{run_id}_label_encoder.pkl"
                        joblib.dump(label_encoder, label_encoder_path)
                        mlflow.log_artifact(label_encoder_path, artifact_path="preprocessing")
                        os.unlink(label_encoder_path)  # 임시 파일 정리
                        
                        logger.info("📋 라벨 인코딩 정보를 MLflow에 저장했습니다.")
                    else:
                        mlflow.log_param("label_encoded", False)
                    
                    # 모델 생성 - HPO 사용 여부에 따라 분기
                    model_params = {}
                    hpo_results = None
                    
                    if use_hpo and optimizer._get_default_param_space(name, task):
                        logger.info(f"🎯 {name} 하이퍼파라미터 최적화 시작...")
                        
                        # 사용자 정의 파라미터 공간 또는 기본값 사용
                        custom_param_space = None
                        if hpo_config.get('param_spaces') and name in hpo_config['param_spaces']:
                            custom_param_space = hpo_config['param_spaces'][name]
                            logger.info(f"사용자 정의 파라미터 공간 사용: {custom_param_space}")
                        
                        # HPO 실행
                        hpo_results = optimizer.optimize_model(
                            model_name=name,
                            task=task,
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_val,
                            y_val=y_val,
                            cv_folds=cv_folds,
                            param_space=custom_param_space
                        )
                        
                        # 최적 파라미터 사용
                        model_params = hpo_results['best_params']
                        logger.info(f"✅ HPO 완료 - 최적 파라미터: {model_params}")
                        logger.info(f"HPO 최고 점수: {hpo_results['best_score']:.4f}")
                        
                        # HPO 결과를 MLflow에 로깅
                        mlflow.log_params({f"hpo_{k}": v for k, v in model_params.items()})
                        mlflow.log_metric("hpo_best_score", hpo_results['best_score'])
                        mlflow.log_metric("hpo_n_trials", hpo_results['n_trials'])
                        mlflow.log_param("hpo_enabled", True)
                        
                    else:
                        # 기존 방식: 기본 파라미터 + 오버라이드
                        if use_hpo and not optimizer._get_default_param_space(name, task):
                            logger.info(f"⚠️  {name}에 대한 HPO 파라미터 공간이 정의되지 않음. 기본 파라미터 사용.")
                        
                        model_overrides = (overrides or {}).get(name, {})
                        model_params = model_overrides
                        mlflow.log_param("hpo_enabled", False)
                        if model_overrides:
                            logger.info(f"사용자 오버라이드 파라미터: {model_overrides}")
                    
                    # 모델 생성
                    logger.info("모델 생성 중...")
                    est = create_estimator(task, name, model_params)
                    logger.info(f"모델 생성 완료: {type(est).__name__}")
                    
                    # 기본 파라미터 로깅
                    params_to_log = {
                        "algorithm": name,
                        "test_size": test_size,
                        "val_size": val_size,
                        "use_cv": use_cv,
                        "cv_folds": cv_folds,
                        "hf_repo": hf_repo,
                        "hf_filename": hf_filename,
                        "hf_revision": hf_revision or "default",
                    }
                    params_to_log.update(model_params)
                    
                    mlflow.log_params(params_to_log)
                    logger.info("파라미터 로깅 완료")

                    # 모델 학습
                    logger.info("모델 학습 시작...")
                    fit_start_time = time.time()
                    est.fit(X_train, y_train)
                    fit_duration = time.time() - fit_start_time
                    logger.info(f"✅ 모델 학습 완료 ({fit_duration:.2f}초)")

                    # 평가
                    logger.info("모델 평가 중...")
                    metrics: Dict[str, Any] = {}
                    
                    # 테스트 세트 평가
                    y_pred_test = est.predict(X_test)
                    if task == "classification":
                        metrics["test"] = classification_metrics(y_test, y_pred_test)
                    else:
                        metrics["test"] = regression_metrics(y_test, y_pred_test)
                    
                    test_score = metrics["test"][best_key]
                    logger.info(f"테스트 {best_key}: {test_score:.4f}")

                    # 검증 세트 평가
                    if X_val is not None:
                        y_pred_val = est.predict(X_val)
                        if task == "classification":
                            metrics["validation"] = classification_metrics(y_val, y_pred_val)
                        else:
                            metrics["validation"] = regression_metrics(y_val, y_pred_val)
                        
                        val_score = metrics["validation"][best_key]
                        logger.info(f"검증 {best_key}: {val_score:.4f}")

                    # 교차 검증
                    if use_cv:
                        logger.info(f"교차 검증 수행 중 ({cv_folds} folds)...")
                        cv_start_time = time.time()
                        scoring = "accuracy" if task == "classification" else "r2"
                        scores = cross_val_score(est, X_train, y_train, cv=cv_folds, scoring=scoring)
                        cv_duration = time.time() - cv_start_time
                        
                        metrics["cross_validation"] = {
                            "scores": scores.tolist(),
                            "mean": float(scores.mean()),
                            "std": float(scores.std()),
                        }
                        logger.info(f"교차 검증 완료 ({cv_duration:.2f}초)")
                        logger.info(f"CV {scoring}: {scores.mean():.4f} (±{scores.std():.4f})")

                    # MLflow 메트릭 로깅
                    logger.info("메트릭 로깅 중...")
                    metric_count = 0
                    for mtype, d in metrics.items():
                        for k, v in d.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"{mtype}_{k}", float(v))
                                metric_count += 1
                    logger.info(f"메트릭 로깅 완료 ({metric_count}개)")

                    # 모델 저장 (개선된 버전)
                    logger.info("모델 아티팩트 저장 중...")
                    local_pkl = f"/tmp/{run_id}.pkl"
                    joblib.dump(est, local_pkl)
                    
                    model_saved = False
                    
                    # 단계별 모델 저장 시도
                    try:
                        from mlflow.models.signature import infer_signature
                        signature = infer_signature(X_train, y_pred_test)
                        input_example = X_train.iloc[:3] if hasattr(X_train, 'iloc') else X_train[:3]
                        
                        mlflow.sklearn.log_model(
                            est, 
                            artifact_path="model",
                            signature=signature,
                            input_example=input_example,
                            registered_model_name=None,
                            await_registration_for=0
                        )
                        logger.info("✅ 모델 아티팩트 저장 완료 (signature + input_example)")
                        model_saved = True
                        
                    except Exception as signature_error:
                        logger.warning(f"⚠️ Signature 포함 저장 실패: {signature_error}")
                        
                        try:
                            mlflow.sklearn.log_model(
                                est, 
                                artifact_path="model",
                                registered_model_name=None,
                                await_registration_for=0
                            )
                            logger.info("✅ 모델 아티팩트 저장 완료 (기본)")
                            model_saved = True
                            
                        except Exception as basic_error:
                            logger.warning(f"⚠️ 기본 모델 저장도 실패: {basic_error}")
                            
                            try:
                                mlflow.log_artifact(local_pkl, artifact_path="model_files")
                                logger.info("✅ 모델 파일 수동 업로드 완료")
                                model_saved = True
                                
                            except Exception as manual_error:
                                logger.error(f"❌ 모든 모델 저장 방법 실패: {manual_error}")
                    
                    finally:
                        # 임시 파일 정리
                        if os.path.exists(local_pkl):
                            os.unlink(local_pkl)

                    if not model_saved:
                        logger.warning("⚠️ 모델 아티팩트 저장 실패")

                    summary = {
                        "run_id": run_id,
                        "algorithm": name,
                        "metrics": metrics,
                        "training_duration": time.time() - model_start_time,
                        "model_saved": model_saved,
                        "hpo_used": use_hpo and bool(hpo_results),
                        "hpo_results": hpo_results,
                        "final_params": model_params,
                        "task": task,
                        "execution_id": execution_id
                    }
                    results.append(summary)

                    # 베스트 모델 업데이트
                    score = metrics["test"][best_key]
                    if score > best_score:
                        best_score = score
                        best = summary
                        hpo_info = f" (HPO)" if summary.get("hpo_used") else ""
                        logger.info(f"🏆 새로운 베스트 모델: {name}{hpo_info} ({best_key}={score:.4f})")
                    
                    model_duration = time.time() - model_start_time
                    logger.info(f"✅ {name} 모델 완료 ({model_duration:.2f}초)")
                    
            except Exception as e:
                logger.error(f"❌ {name} 모델 학습 실패: {str(e)}")
                logger.error(f"스택트레이스:\n{traceback.format_exc()}")
                continue

        # 안전망: best가 반드시 있어야 함
        if not best:
            error_msg = "모든 모델 학습이 실패했습니다"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        hpo_info = f" (HPO)" if best.get("hpo_used") else ""
        logger.info(f"\n🏆 최고 성능 모델: {best['algorithm']}{hpo_info} (Run ID: {best['run_id']})")
        logger.info(f"최고 {best_key}: {best_score:.4f}")
        
        if best.get("hpo_used") and best.get("hpo_results"):
            logger.info(f"HPO 최적 파라미터: {best['final_params']}")

        # manifest 생성 (HPO 및 라벨 인코딩 정보 포함)
        logger.info("\n📄 Manifest 생성 중...")
        manifest = {
            "results": results,
            "best": best,
            "feature_names": used_feature_names,
            "model_id": model_id,
            "execution_id": execution_id,
            "task": task,
            "training_timestamp": datetime.now().isoformat(),
            "training_duration": time.time() - training_start_time,
            "data_info": {
                "hf_repo": hf_repo,
                "hf_filename": hf_filename,
                "hf_revision": hf_revision or "default",
                "target_column": target_column,
                "feature_count": len(used_feature_names),
                "data_shape": df.shape
            },
            "hpo_summary": {
                "enabled": use_hpo,
                "models_optimized": sum(1 for r in results if r.get("hpo_used", False)) if use_hpo else 0,
                "config": hpo_config if use_hpo else None
            },
            # 라벨 인코딩 정보 추가
            "label_encoding": {
                "used": label_encoder is not None,
                "original_classes": original_classes,
                "label_mapping": label_mapping
            } if task == "classification" else None,
            # 버전 관리 정보 추가
            "version_info": {
                "unique_execution": USE_UNIQUE_PATHS,
                "cleanup_enabled": CLEANUP_TEMP_FILES,
                "temp_directories": temp_directories if temp_directories else []
            }
        }

        # 베스트 run 아티팩트로 manifest.json 저장
        logger.info("베스트 run에 manifest 저장 중...")
        tmp_manifest = None
        try:
            client = MlflowClient()
            tmp_manifest = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
            with open(tmp_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            client.log_artifact(best["run_id"], tmp_manifest, artifact_path="manifest")
            logger.info("✅ Manifest 저장 완료")
        except Exception as e:
            logger.error(f"⚠️  Manifest 저장 실패: {e}")
        finally:
            if tmp_manifest and os.path.exists(tmp_manifest):
                os.unlink(tmp_manifest)

        # 외부 스토리지에도 저장 (선택사항)
        runs_manifest_uri: Optional[str] = None
        if artifact_base_uri:
            logger.info("외부 스토리지에 manifest 업로드 중...")
            try:
                storage = StorageClient(
                    artifact_base_uri,
                    **(storage_ctor_kwargs or {}),
                )
                
                # 고유한 파일명으로 저장 (충돌 방지)
                if USE_UNIQUE_PATHS:
                    storage_key = f"models/{model_id}/{execution_id}_runs.json"
                else:
                    storage_key = f"models/{model_id}/_last_runs.json"
                
                tmp_manifest_storage = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
                with open(tmp_manifest_storage, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
                
                runs_manifest_uri = storage.upload_file(tmp_manifest_storage, storage_key)
                os.unlink(tmp_manifest_storage)
                logger.info(f"✅ 외부 스토리지 업로드 완료: {runs_manifest_uri}")
            except Exception as e:
                logger.error(f"⚠️  외부 스토리지 업로드 실패: {e}")

        # MLflow Model Registry 등록 & Production 승격
        production_version = None
        if os.getenv("ENABLE_MODEL_REGISTRY", "true").lower() == "true" and best.get("model_saved", False):
            try:
                # 버전 태그 생성
                version_tags = {
                    "execution_id": execution_id,
                    "training_timestamp": datetime.now().isoformat(),
                    "data_source": f"{hf_repo}/{hf_filename}",
                    "unique_run": str(USE_UNIQUE_PATHS),
                }
                
                if best.get("hpo_used"):
                    version_tags["hpo_optimized"] = "true"
                    version_tags["hpo_trials"] = str(best.get("hpo_results", {}).get("n_trials", 0))
                
                if label_encoder is not None:
                    version_tags["label_encoded"] = "true"
                    version_tags["num_classes"] = str(len(original_classes))
                
                production_version = _register_best_to_mlflow(
                    model_name=model_id,
                    best=best,
                    feature_names=used_feature_names,
                    version_tags=version_tags
                )
            except Exception as e:
                logger.error(f"⚠️  Model Registry 등록 실패 (무시하고 계속): {e}")
                production_version = None
        elif not best.get("model_saved", False):
            logger.warning("⚠️  베스트 모델의 아티팩트가 저장되지 않아 Model Registry 등록을 건너뜁니다")
        else:
            logger.info("ℹ️  Model Registry 비활성화됨 (ENABLE_MODEL_REGISTRY=false)")

        total_duration = time.time() - training_start_time
        logger.info(f"\n🎉 전체 학습 완료! ({total_duration:.2f}초)")
        logger.info(f"실행 ID: {execution_id}")
        logger.info(f"학습된 모델 수: {len(results)}")
        logger.info(f"베스트 모델: {best['algorithm']}")
        if use_hpo:
            hpo_count = sum(1 for r in results if r.get("hpo_used", False))
            logger.info(f"HPO 적용된 모델 수: {hpo_count}/{len(results)}")
        if label_encoder is not None:
            logger.info(f"라벨 인코딩 적용: {original_classes} -> {list(range(len(original_classes)))}")
        if production_version:
            logger.info(f"Model Registry: {model_id} v{production_version} (Production)")
        logger.info("=" * 80)

        # 반환 데이터에 모든 정보 포함
        return {
            "results": results,
            "best": best,
            "execution_id": execution_id,
            "runs_manifest_uri": runs_manifest_uri or "",
            "registry": {
                "model_name": model_id, 
                "production_version": production_version or "",
                "version_tags": version_tags if 'version_tags' in locals() else {}
            },
            "training_duration": total_duration,
            "feature_names": used_feature_names,
            "training_timestamp": datetime.now().isoformat(),
            "data_info": {
                "hf_repo": hf_repo,
                "hf_filename": hf_filename,
                "hf_revision": hf_revision or "default",
                "target_column": target_column,
                "feature_count": len(used_feature_names),
                "data_shape": df.shape
            },
            "hpo_summary": {
                "enabled": use_hpo,
                "models_optimized": sum(1 for r in results if r.get("hpo_used", False)) if use_hpo else 0,
                "config": hpo_config if use_hpo else None
            },
            # 라벨 인코딩 정보 반환
            "label_encoding": {
                "used": label_encoder is not None,
                "original_classes": original_classes,
                "label_mapping": label_mapping
            } if task == "classification" else None,
            # 버전 관리 정보
            "version_info": {
                "unique_execution": USE_UNIQUE_PATHS,
                "cleanup_enabled": CLEANUP_TEMP_FILES,
            }
        }
        
    except Exception as e:
        total_duration = time.time() - training_start_time
        logger.error(f"\n💥 학습 실패! ({total_duration:.2f}초)")
        logger.error(f"실행 ID: {execution_id}")
        logger.error(f"에러: {str(e)}")
        logger.error(f"에러 타입: {type(e).__name__}")
        logger.error(f"전체 스택트레이스:\n{traceback.format_exc()}")
        logger.error("=" * 80)
        raise
    
    finally:
        # 임시 디렉토리 정리
        if CLEANUP_TEMP_FILES and temp_directories:
            logger.info("🧹 임시 디렉토리 정리 중...")
            import shutil
            for temp_dir in temp_directories:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        logger.info(f"✅ 정리 완료: {temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️  디렉토리 정리 실패: {temp_dir} - {cleanup_error}")