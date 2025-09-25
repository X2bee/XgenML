# /src/xgenml/services/hyperparameter_optimization.py
import optuna
import logging
from typing import Dict, Any, List, Optional, Callable
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

from ..core.model_provider import create_estimator

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Optuna를 사용한 하이퍼파라미터 최적화"""
    
    def __init__(self, n_trials: int = 50, timeout: Optional[int] = None):
        self.n_trials = n_trials
        self.timeout = timeout  # seconds
    
    def optimize_model(
        self,
        model_name: str,
        task: str,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        cv_folds: int = 5,
        param_space: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """단일 모델에 대한 하이퍼파라미터 최적화"""
        
        logger.info(f"Starting hyperparameter optimization for {model_name}")
        
        # 기본 파라미터 공간 정의
        if param_space is None:
            param_space = self._get_default_param_space(model_name, task)
        
        def objective(trial):
            # 파라미터 샘플링
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        step=param_config.get('step', 1)
                    )
                elif param_config['type'] == 'float':
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config['choices']
                    )
            
            try:
                # 모델 생성
                model = create_estimator(task, model_name, params)
                
                # 평가 방법 선택
                if X_val is not None and y_val is not None:
                    # 검증 세트가 있으면 직접 평가
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    if task == "classification":
                        score = accuracy_score(y_val, y_pred)
                    else:
                        score = r2_score(y_val, y_pred)
                else:
                    # 교차 검증 사용
                    scoring = "accuracy" if task == "classification" else "r2"
                    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
                    score = scores.mean()
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed with params {params}: {str(e)}")
                return float('-inf') if task == "regression" else 0.0
        
        # Optuna 스터디 실행
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(
            objective, 
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        logger.info(f"Optimization completed. Best score: {study.best_value:.4f}")
        
        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
            "study": study  # 전체 스터디 객체 (분석용)
        }
    
    def _get_default_param_space(self, model_name: str, task: str) -> Dict[str, Any]:
        """모델별 기본 하이퍼파라미터 탐색 공간 정의"""
        
        spaces = {
            "classification": {
                "random_forest": {
                    "n_estimators": {"type": "int", "low": 100, "high": 1000, "step": 50},
                    "max_depth": {"type": "int", "low": 3, "high": 20},
                    "min_samples_split": {"type": "int", "low": 2, "high": 20},
                    "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
                    "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]}
                },
                "xgboost": {
                    "n_estimators": {"type": "int", "low": 100, "high": 1000, "step": 50},
                    "max_depth": {"type": "int", "low": 3, "high": 12},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                    "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                    "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True}
                },
                "svm": {
                    "C": {"type": "float", "low": 0.1, "high": 100.0, "log": True},
                    "gamma": {"type": "categorical", "choices": ["scale", "auto"]} if task == "classification" else {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
                    "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly"]}
                },
                "gradient_boosting": {
                    "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                    "max_depth": {"type": "int", "low": 3, "high": 10},
                    "min_samples_split": {"type": "int", "low": 2, "high": 20},
                    "min_samples_leaf": {"type": "int", "low": 1, "high": 10}
                }
            },
            "regression": {
                "random_forest": {
                    "n_estimators": {"type": "int", "low": 100, "high": 1000, "step": 50},
                    "max_depth": {"type": "int", "low": 3, "high": 20},
                    "min_samples_split": {"type": "int", "low": 2, "high": 20},
                    "min_samples_leaf": {"type": "int", "low": 1, "high": 10}
                },
                "xgboost": {
                    "n_estimators": {"type": "int", "low": 100, "high": 1000, "step": 50},
                    "max_depth": {"type": "int", "low": 3, "high": 12},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0}
                },
                "ridge": {
                    "alpha": {"type": "float", "low": 0.001, "high": 100.0, "log": True}
                },
                "lasso": {
                    "alpha": {"type": "float", "low": 0.001, "high": 10.0, "log": True}
                }
            }
        }
        
        return spaces.get(task, {}).get(model_name, {})