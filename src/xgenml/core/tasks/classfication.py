# /src/xgenml/core/tasks/classification.py
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .base import BaseTask
from . import TaskRegistry
from ..metrics import classification_metrics
from ...utils.logger_config import setup_logger

logger = setup_logger(__name__)


@TaskRegistry.register("classification")
class ClassificationTask(BaseTask):
    """분류 태스크"""
    
    def _get_task_type(self) -> str:
        return "classification"
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str], Dict[str, Any]]:
        """분류 데이터 준비"""
        self.validate_data(df, target_column)
        
        # 피처 선택
        if feature_columns:
            X = df[feature_columns]
        else:
            X = df.drop(columns=[target_column])
        
        y = df[target_column]
        
        # 라벨 인코딩
        metadata = {}
        if y.dtype == 'object':
            logger.info("🏷️  라벨 인코딩 수행")
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            metadata = {
                "label_encoded": True,
                "original_classes": y.unique().tolist(),
                "label_mapping": dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))),
                "encoder": label_encoder
            }
            y = y_encoded
        else:
            metadata = {"label_encoded": False}
        
        feature_names = X.columns.tolist()
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
        """분류 데이터 분할 (stratified)"""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        if val_size > 0:
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio,
                random_state=random_state, stratify=y_temp
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_default_models(self) -> List[str]:
        """분류 기본 모델"""
        return [
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "svm"
        ]
    
    def get_primary_metric(self) -> str:
        return "accuracy"
    
    def evaluate_model(
        self,
        estimator,
        X_test,
        y_test,
        **kwargs
    ) -> Dict[str, Any]:
        """분류 모델 평가"""
        y_pred = estimator.predict(X_test)
        return classification_metrics(y_test, y_pred)