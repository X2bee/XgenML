# /src/xgenml/core/tasks/timeseries.py
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .base import BaseTask
from . import TaskRegistry
from ..metrics import timeseries_metrics
from ...utils.logger_config import setup_logger

logger = setup_logger(__name__)


@TaskRegistry.register("timeseries")
class TimeSeriesTask(BaseTask):
    """시계열 예측 태스크"""
    
    def _get_task_type(self) -> str:
        return "timeseries"
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        lookback_window: int = 10,
        forecast_horizon: int = 1,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """
        시계열 데이터 준비
        
        Args:
            time_column: 시간 컬럼명
            lookback_window: 과거 몇 개 시점을 볼지
            forecast_horizon: 미래 몇 개 시점을 예측할지
        """
        self.validate_data(df, target_column)
        
        # 시간 컬럼 처리
        if time_column:
            df = df.sort_values(time_column)
            logger.info(f"시간 컬럼 '{time_column}'으로 정렬")
        
        # 피처 선택
        if feature_columns:
            feature_cols = feature_columns
        else:
            feature_cols = [col for col in df.columns 
                          if col not in [target_column, time_column]]
        
        # 시계열 윈도우 생성
        X, y = self._create_sequences(
            df[feature_cols].values,
            df[target_column].values,
            lookback_window,
            forecast_horizon
        )
        
        # 피처 이름 생성 (lag 정보 포함)
        feature_names = []
        for lag in range(lookback_window, 0, -1):
            for col in feature_cols:
                feature_names.append(f"{col}_lag_{lag}")
        
        metadata = {
            "label_encoded": False,
            "time_column": time_column,
            "lookback_window": lookback_window,
            "forecast_horizon": forecast_horizon,
            "original_feature_names": feature_cols,
            "time_series_type": "univariate" if len(feature_cols) == 1 else "multivariate"
        }
        
        logger.info(f"시계열 시퀀스 생성 완료: {X.shape}")
        logger.info(f"Lookback: {lookback_window}, Forecast: {forecast_horizon}")
        
        return X, y, feature_names, metadata
    
    def _create_sequences(
        self,
        features: np.ndarray,
        target: np.ndarray,
        lookback: int,
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 생성"""
        X, y = [], []
        
        for i in range(len(features) - lookback - horizon + 1):
            # 과거 lookback 시점의 데이터
            X.append(features[i:i+lookback].flatten())
            # 미래 horizon 시점의 타겟
            if horizon == 1:
                y.append(target[i+lookback])
            else:
                y.append(target[i+lookback:i+lookback+horizon])
        
        return np.array(X), np.array(y)
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        **kwargs
    ) -> Tuple:
        """
        시계열 데이터 분할 (시간 순서 유지)
        ⚠️ 시계열은 랜덤 셔플하지 않음!
        """
        n_samples = len(X)
        
        # 테스트 분할점
        test_split = int(n_samples * (1 - test_size))
        X_temp, X_test = X[:test_split], X[test_split:]
        y_temp, y_test = y[:test_split], y[test_split:]
        
        logger.info(f"시계열 분할 (시간 순서 유지): Train+Val={len(X_temp)}, Test={len(X_test)}")
        
        # 검증 분할
        if val_size > 0:
            val_split = int(len(X_temp) * (1 - val_size / (1 - test_size)))
            X_train, X_val = X_temp[:val_split], X_temp[val_split:]
            y_train, y_val = y_temp[:val_split], y_temp[val_split:]
            logger.info(f"Train={len(X_train)}, Val={len(X_val)}")
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_default_models(self) -> List[str]:
        """시계열 예측 모델"""
        return [
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "linear_regression",
            # 나중에 추가 가능: ARIMA, Prophet, LSTM 등
        ]
    
    def get_primary_metric(self) -> str:
        return "rmse"
    
    def evaluate_model(
        self,
        estimator,
        X_test,
        y_test,
        **kwargs
    ) -> Dict[str, Any]:
        """시계열 모델 평가"""
        y_pred = estimator.predict(X_test)
        return timeseries_metrics(y_test, y_pred)