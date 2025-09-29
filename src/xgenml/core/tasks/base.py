# /src/xgenml/core/tasks/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

class BaseTask(ABC):
    """모든 태스크의 베이스 클래스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.task_type = self._get_task_type()
    
    @abstractmethod
    def _get_task_type(self) -> str:
        """태스크 타입 반환 (classification, regression, timeseries, anomaly_detection 등)"""
        pass
    
    @abstractmethod
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[Any, Any, List[str], Dict[str, Any]]:
        """
        데이터 준비 (태스크별로 다름)
        
        Returns:
            X: 피처
            y: 타겟
            feature_names: 피처 이름 목록
            metadata: 추가 메타데이터
        """
        pass
    
    @abstractmethod
    def split_data(
        self,
        X: Any,
        y: Any,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        **kwargs
    ) -> Tuple:
        """데이터 분할 (태스크별로 다름)"""
        pass
    
    @abstractmethod
    def get_default_models(self) -> List[str]:
        """태스크에 적합한 기본 모델 목록"""
        pass
    
    @abstractmethod
    def get_primary_metric(self) -> str:
        """주요 평가 지표"""
        pass
    
    @abstractmethod
    def evaluate_model(
        self,
        estimator,
        X_test,
        y_test,
        **kwargs
    ) -> Dict[str, Any]:
        """모델 평가"""
        pass
    
    def validate_data(self, df: pd.DataFrame, target_column: str) -> bool:
        """데이터 유효성 검사"""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        return True
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """태스크별 모델 기본 파라미터"""
        return {}