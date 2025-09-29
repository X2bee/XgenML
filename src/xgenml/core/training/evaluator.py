# /src/xgenml/core/training/evaluator.py
import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.model_selection import cross_val_score

from src.xgenml.core.metrics import classification_metrics, regression_metrics
from src.xgenml.utils.logger_config import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """모델 평가 담당 클래스"""
    
    def __init__(self, task: str):
        self.task = task
        self.primary_metric = "accuracy" if task == "classification" else "r2"
    
    def evaluate(
        self,
        estimator,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        use_cv: bool = False,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """전체 평가 수행"""
        logger.info("모델 평가 중...")
        metrics: Dict[str, Any] = {}
        
        # 테스트 세트 평가
        metrics["test"] = self._evaluate_split(estimator, X_test, y_test, "테스트")
        
        # 검증 세트 평가
        if X_val is not None:
            metrics["validation"] = self._evaluate_split(estimator, X_val, y_val, "검증")
        
        # 교차 검증
        if use_cv:
            metrics["cross_validation"] = self._cross_validate(
                estimator, X_train, y_train, cv_folds
            )
        
        return metrics
    
    def _evaluate_split(self, estimator, X, y, split_name: str) -> Dict[str, float]:
        """단일 데이터 분할 평가"""
        y_pred = estimator.predict(X)
        
        if self.task == "classification":
            metrics = classification_metrics(y, y_pred)
        else:
            metrics = regression_metrics(y, y_pred)
        
        score = metrics[self.primary_metric]
        logger.info(f"{split_name} {self.primary_metric}: {score:.4f}")
        
        return metrics
    
    def _cross_validate(
        self, estimator, X_train, y_train, cv_folds: int
    ) -> Dict[str, Any]:
        """교차 검증 수행"""
        logger.info(f"교차 검증 수행 중 ({cv_folds} folds)...")
        cv_start_time = time.time()
        
        scoring = "accuracy" if self.task == "classification" else "r2"
        scores = cross_val_score(estimator, X_train, y_train, cv=cv_folds, scoring=scoring)
        
        cv_duration = time.time() - cv_start_time
        cv_metrics = {
            "scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }
        
        logger.info(f"교차 검증 완료 ({cv_duration:.2f}초)")
        logger.info(f"CV {scoring}: {scores.mean():.4f} (±{scores.std():.4f})")
        
        return cv_metrics