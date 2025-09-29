# /src/xgenml/core/tasks/regression.py
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .base import BaseTask
from . import TaskRegistry
from ..metrics import regression_metrics


@TaskRegistry.register("regression")
class RegressionTask(BaseTask):
    """회귀 태스크"""
    
    def _get_task_type(self) -> str:
        return "regression"
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str], Dict[str, Any]]:
        """회귀 데이터 준비"""
        self.validate_data(df, target_column)
        
        if feature_columns:
            X = df[feature_columns]
        else:
            X = df.drop(columns=[target_column])
        
        y = df[target_column].values
        feature_names = X.columns.tolist()
        
        metadata = {"label_encoded": False}
        return X, y, feature_names, metadata
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        **kwargs
    ) -> Tuple:
        """회귀 데이터 분할"""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if val_size > 0:
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_default_models(self) -> List[str]:
        """회귀 기본 모델"""
        return [
            "linear_regression",
            "ridge",
            "lasso",
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "lightgbm"
        ]
    
    def get_primary_metric(self) -> str:
        return "r2"
    
    def evaluate_model(
        self,
        estimator,
        X_test,
        y_test,
        **kwargs
    ) -> Dict[str, Any]:
        """회귀 모델 평가"""
        y_pred = estimator.predict(X_test)
        return regression_metrics(y_test, y_pred)