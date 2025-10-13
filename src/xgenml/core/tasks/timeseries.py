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
    lookback_window: int = 30,
    forecast_horizon: int = 1,
    **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """시계열 데이터 준비"""
        self.validate_data(df, target_column)
        
        # 🔥 수정 1: 시간 컬럼 처리 강화
        if time_column:
            if time_column not in df.columns:
                raise ValueError(f"time_column '{time_column}'이 데이터에 없습니다")
            
            # 날짜 타입 변환
            if df[time_column].dtype == 'object':
                try:
                    df[time_column] = pd.to_datetime(df[time_column])
                    logger.info(f"'{time_column}' 컬럼을 datetime으로 변환")
                except Exception as e:
                    logger.warning(f"날짜 변환 실패: {e}. 원본 데이터 사용")
            
            # 정렬 및 인덱스 리셋
            df = df.sort_values(time_column).reset_index(drop=True)
            logger.info(f"시간 컬럼 '{time_column}'으로 정렬 완료")
        
        # 🔥 수정 2: 피처 선택 로직 명확화
        if feature_columns:
            # 명시적으로 지정된 경우
            feature_cols = feature_columns
            logger.info(f"명시적 피처 사용: {feature_cols}")
        else:
            # 자동 선택: target과 time_column 제외
            exclude_cols = {target_column}
            if time_column:
                exclude_cols.add(time_column)
            
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            logger.info(f"자동 피처 선택: {feature_cols}")
            logger.info(f"제외된 컬럼: {exclude_cols}")
        
        # 🔥 추가: 피처 검증
        missing_features = set(feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f"데이터에 없는 피처: {missing_features}")
        
        # 시계열 시퀀스 생성
        X, y = self._create_sequences(
            df[feature_cols].values,
            df[target_column].values,
            lookback_window,
            forecast_horizon
        )
        
        # 피처 이름 생성
        feature_names = []
        for lag in range(lookback_window, 0, -1):
            for col in feature_cols:
                feature_names.append(f"{col}_lag_{lag}")
        
        metadata = {
            "label_encoded": False,
            "time_column": time_column,  # 🔥 None이 아닌 실제 값 저장
            "lookback_window": lookback_window,
            "forecast_horizon": forecast_horizon,
            "original_feature_names": feature_cols,
            "time_series_type": "univariate" if len(feature_cols) == 1 else "multivariate"
        }
        
        logger.info(f"✅ 시계열 시퀀스 생성: X={X.shape}, y={y.shape}")
        logger.info(f"   Lookback: {lookback_window}, Forecast: {forecast_horizon}")
        logger.info(f"   피처 수: {len(feature_cols)}, 총 lag 피처: {len(feature_names)}")
        
        return X, y, feature_names, metadata

    
    def _create_sequences(
    self,
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 생성 - 3D 형태 유지"""
        X, y = [], []
        
        for i in range(len(features) - lookback - horizon + 1):
            # 🔥 수정: flatten 제거 → 3D 유지
            X.append(features[i:i+lookback])  # (lookback, n_features)
            
            if horizon == 1:
                y.append(target[i+lookback])
            else:
                y.append(target[i+lookback:i+lookback+horizon])
        
        X = np.array(X)  # (n_samples, lookback, n_features)
        y = np.array(y)
        
        # 🔥 추가: sklearn 모델용으로는 2D 변환 필요
        # 나중에 모델 타입에 따라 reshape 선택
        X_2d = X.reshape(X.shape[0], -1)  # (n_samples, lookback*n_features)
        
    return X_2d, y  # 일단 2D 반환 (기존 코드 호환)
    
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