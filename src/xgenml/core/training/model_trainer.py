# /src/xgenml/core/training/model_trainer.py
import time
from typing import Dict, Any, Optional, Tuple

from src.xgenml.core.model_provider import create_estimator
from src.xgenml.services.hyperparameter_optimization import HyperparameterOptimizer
from src.xgenml.core.training.evaluator import ModelEvaluator
from src.xgenml.core.training.mlflow_manager import MLflowManager
from src.xgenml.utils.logger_config import setup_logger

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
        hpo_config: Optional[Dict[str, Any]] = None
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
                estimator=estimator,
                params=params_to_log,
                metrics=metrics,
                X_train=X_train,
                y_pred_test=estimator.predict(X_test),
                execution_id=execution_id,
                data_source_info=data_source_info,
                label_encoding_info=label_encoding_info,
                hpo_results=hpo_results
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