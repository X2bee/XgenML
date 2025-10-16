# /src/xgenml/core/training/model_trainer.py
import time
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.xgenml.core.model_provider import create_estimator, UserScriptModel
from src.xgenml.services.hyperparameter_optimization import HyperparameterOptimizer
from src.xgenml.core.training.evaluator import ModelEvaluator
from src.xgenml.core.training.mlflow_manager import MLflowManager
from src.xgenml.utils.logger_config import setup_logger
from src.xgenml.services.script_executor import ScriptExecutor

logger = setup_logger(__name__)


class ModelTrainer:
    """개별 모델 학습 담당 클래스"""
    
    def __init__(
        self,
        task: str,
        mlflow_manager: MLflowManager,
        use_unique_paths: bool = True
    ):
        self.task = task
        self.mlflow_manager = mlflow_manager
        self.use_unique_paths = use_unique_paths
        self.evaluator = ModelEvaluator(task)
        self.optimizer: Optional[HyperparameterOptimizer] = None
    
    def set_optimizer(self, optimizer: HyperparameterOptimizer):
        """하이퍼파라미터 옵티마이저 설정"""
        self.optimizer = optimizer
    
    def train_model(
        self,
        model_name: str,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        execution_id: str,
        data_source_info: Dict[str, Any],
        label_encoding_info: Optional[Dict[str, Any]] = None,
        use_cv: bool = False,
        cv_folds: int = 5,
        overrides: Optional[Dict[str, Any]] = None,
        hpo_config: Optional[Dict[str, Any]] = None,
        input_schema: Optional[Dict[str, Any]] = None,  # NEW
        output_schema: Optional[Dict[str, Any]] = None  # NEW
    ) -> Dict[str, Any]:
        """단일 모델 학습"""
        model_start_time = time.time()
        logger.info(f"\n{model_name} 학습 중...")
        
        try:
            # 모델 파라미터 준비
            model_params, hpo_results = self._prepare_model_params(
                model_name, X_train, y_train, X_val, y_val,
                cv_folds, overrides, hpo_config
            )
            
            # 모델 생성
            logger.info("모델 생성 중...")
            estimator = create_estimator(self.task, model_name, model_params)
            logger.info(f"모델 생성 완료: {type(estimator).__name__}")

            if isinstance(estimator, UserScriptModel):
                # User Script Execution
                logger.info("사용자 스크립트 실행...")
                script_executor = ScriptExecutor()
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    artifact_dir = temp_path / "artifacts"
                    artifact_dir.mkdir(parents=True, exist_ok=True)

                    X_train_path = temp_path / "X_train.parquet"
                    y_train_path = temp_path / "y_train.parquet"
                    X_val_path = temp_path / "X_val.parquet"
                    y_val_path = temp_path / "y_val.parquet"
                    X_test_path = temp_path / "X_test.parquet"
                    y_test_path = temp_path / "y_test.parquet"

                    X_train.to_parquet(X_train_path)
                    pd.Series(y_train).to_frame().to_parquet(y_train_path)
                    X_val.to_parquet(X_val_path)
                    pd.Series(y_val).to_frame().to_parquet(y_val_path)
                    X_test.to_parquet(X_test_path)
                    pd.Series(y_test).to_frame().to_parquet(y_test_path)

                    run_config = {
                        "X_train_path": str(X_train_path),
                        "y_train_path": str(y_train_path),
                        "X_val_path": str(X_val_path),
                        "y_val_path": str(y_val_path),
                        "X_test_path": str(X_test_path),
                        "y_test_path": str(y_test_path),
                        "artifact_dir": str(artifact_dir),
                        "params": model_params
                    }
                    execution_result = script_executor.execute(estimator.model_info['content'], run_config)

                # 실행 결과 확인
                logger.info(f"UserScript 실행 결과 키: {execution_result.keys()}")
                logger.info(f"UserScript 전체 실행 결과: {execution_result}")

                result_data = execution_result.get("result") or {}
                raw_metrics = result_data.get("metrics", {})
                artifacts = result_data.get("artifacts", [])

                # stdout/stderr 로깅
                stdout = execution_result.get("stdout", [])
                stderr = execution_result.get("stderr", [])
                exit_code = execution_result.get("exit_code", -1)

                # 항상 stdout/stderr 출력 (디버깅용)
                logger.info("=" * 80)
                logger.info("UserScript stdout:")
                if stdout:
                    for line in stdout:
                        logger.info(f"  {line}")
                else:
                    logger.info("  (비어있음)")

                logger.warning("UserScript stderr:")
                if stderr:
                    for line in stderr:
                        logger.warning(f"  {line}")
                else:
                    logger.warning("  (비어있음)")

                logger.info(f"UserScript exit code: {exit_code}")
                logger.info("=" * 80)

                # 에러가 있으면 예외 발생
                if exit_code != 0:
                    # stderr와 stdout 모두 확인
                    error_lines = []
                    if stderr:
                        error_lines.extend(stderr)
                    if stdout and not stderr:
                        # stderr가 비어있으면 stdout도 확인
                        error_lines.extend(stdout)

                    error_msg = "\n".join(error_lines) if error_lines else f"실행 실패 (상세 정보 없음). execution_result: {execution_result}"
                    raise RuntimeError(f"UserScript 실행 실패 (exit_code={exit_code}):\n{error_msg}")

                if not raw_metrics:
                    logger.error("UserScript가 메트릭을 반환하지 않았습니다!")
                    logger.error(f"result_data: {result_data}")
                    raise RuntimeError("UserScript가 비어있는 메트릭을 반환했습니다")

                # UserScript 메트릭을 표준 구조로 변환 (train/val/test)
                # UserScript가 직접 반환한 메트릭은 test 메트릭으로 간주
                metrics = {
                    "train": {},
                    "val": {},
                    "test": raw_metrics  # UserScript 메트릭을 test로 사용
                }

                logger.info(f"✅ UserScript 실행 완료 - 메트릭: {raw_metrics}")
                logger.info(f"✅ UserScript 아티팩트: {len(artifacts)}개")

                estimator_for_log = None  # No estimator object for user scripts
                user_script_artifacts = artifacts  # 아티팩트 저장
            else:
                # 모델 학습
                logger.info("모델 학습 시작...")
                fit_start_time = time.time()
                estimator.fit(X_train, y_train)
                fit_duration = time.time() - fit_start_time
                logger.info(f"✅ 모델 학습 완료 ({fit_duration:.2f}초)")

                # 모델 평가
                metrics = self.evaluator.evaluate(
                    estimator, X_train, y_train, X_val, y_val, X_test, y_test,
                    use_cv, cv_folds
                )
                estimator_for_log = estimator
                user_script_artifacts = []  # 일반 모델은 아티팩트 없음

            # MLflow 로깅
            run_name = self._generate_run_name(model_name, execution_id)

            params_to_log = {
                "algorithm": model_name,
                "use_cv": use_cv,
                "cv_folds": cv_folds,
            }
            params_to_log.update(model_params)

            run_id, model_saved = self.mlflow_manager.log_model_training(
                    run_name=run_name,
                    estimator=estimator_for_log,
                    params=params_to_log,
                    metrics=metrics,
                    X_train=X_train,
                    y_pred_test=estimator.predict(X_test) if not isinstance(estimator, UserScriptModel) else None,
                    execution_id=execution_id,
                    data_source_info=data_source_info,
                    label_encoding_info=label_encoding_info,
                    hpo_results=hpo_results,
                    input_schema=input_schema,  # NEW
                    output_schema=output_schema,  # NEW
                    user_script_artifacts=user_script_artifacts if isinstance(estimator, UserScriptModel) else None
                )
            
            # 결과 요약
            summary = {
                "run_id": run_id,
                "algorithm": model_name,
                "metrics": metrics,
                "training_duration": time.time() - model_start_time,
                "model_saved": model_saved,
                "hpo_used": bool(hpo_results),
                "hpo_results": hpo_results,
                "final_params": model_params,
                "task": self.task,
                "execution_id": execution_id
            }
            
            model_duration = time.time() - model_start_time
            logger.info(f"✅ {model_name} 모델 완료 ({model_duration:.2f}초)")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ {model_name} 모델 학습 실패: {str(e)}")
            raise
    
    def _prepare_model_params(
        self,
        model_name: str,
        X_train, y_train,
        X_val, y_val,
        cv_folds: int,
        overrides: Optional[Dict[str, Any]],
        hpo_config: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """모델 파라미터 준비 (HPO 또는 기본값)"""
        use_hpo = hpo_config and hpo_config.get('enable_hpo', False)
        hpo_results = None
        
        if use_hpo and self.optimizer and self.optimizer._get_default_param_space(model_name, self.task):
            logger.info(f"🎯 {model_name} 하이퍼파라미터 최적화 시작...")
            
            # 사용자 정의 파라미터 공간
            custom_param_space = None
            if hpo_config.get('param_spaces') and model_name in hpo_config['param_spaces']:
                custom_param_space = hpo_config['param_spaces'][model_name]
                logger.info(f"사용자 정의 파라미터 공간 사용: {custom_param_space}")
            
            # HPO 실행
            hpo_results = self.optimizer.optimize_model(
                model_name=model_name,
                task=self.task,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                cv_folds=cv_folds,
                param_space=custom_param_space
            )
            
            model_params = hpo_results['best_params']
            logger.info(f"✅ HPO 완료 - 최적 파라미터: {model_params}")
            logger.info(f"HPO 최고 점수: {hpo_results['best_score']:.4f}")
            
        else:
            # 기본 파라미터 + 오버라이드
            if use_hpo and not self.optimizer._get_default_param_space(model_name, self.task):
                logger.info(f"⚠️  {model_name}에 대한 HPO 파라미터 공간이 정의되지 않음. 기본 파라미터 사용.")
            
            model_overrides = (overrides or {}).get(model_name, {})
            model_params = model_overrides
            if model_overrides:
                logger.info(f"사용자 오버라이드 파라미터: {model_overrides}")
        
        return model_params, hpo_results
    
    def _generate_run_name(self, model_name: str, execution_id: str) -> str:
        """MLflow run 이름 생성"""
        if self.use_unique_paths:
            return f"{self.task}:{model_name}:{execution_id[-8:]}"
        else:
            return f"{self.task}:{model_name}"