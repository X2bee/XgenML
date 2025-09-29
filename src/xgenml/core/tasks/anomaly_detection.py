# /src/xgenml/core/tasks/anomaly_detection.py
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .base import BaseTask
from . import TaskRegistry
from ..metrics import anomaly_detection_metrics
from ...utils.logger_config import setup_logger

logger = setup_logger(__name__)


@TaskRegistry.register("anomaly_detection")
class AnomalyDetectionTask(BaseTask):
    """이상 탐지 태스크"""
    
    def _get_task_type(self) -> str:
        return "anomaly_detection"
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,  # 이상 탐지는 타겟이 없을 수도 있음
        feature_columns: Optional[List[str]] = None,
        contamination: float = 0.1,  # 이상치 비율
        **kwargs
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[str], Dict[str, Any]]:
        """
        이상 탐지 데이터 준비
        
        Args:
            contamination: 예상 이상치 비율 (unsupervised의 경우)
        """
        # 피처 선택
        if feature_columns:
            X = df[feature_columns]
        elif target_column:
            X = df.drop(columns=[target_column])
        else:
            X = df.copy()
        
        # 타겟 (있는 경우)
        y = None
        is_supervised = False
        
        if target_column and target_column in df.columns:
            y = df[target_column].values
            is_supervised = True
            
            # 이상치 비율 계산
            actual_contamination = np.sum(y == 1) / len(y) if len(y) > 0 else contamination
            logger.info(f"✅ 지도 학습 이상 탐지 (라벨 있음)")
            logger.info(f"이상치 비율: {actual_contamination:.2%}")
        else:
            logger.info(f"✅ 비지도 학습 이상 탐지 (라벨 없음)")
            logger.info(f"예상 이상치 비율: {contamination:.2%}")
        
        feature_names = X.columns.tolist()
        
        metadata = {
            "label_encoded": False,
            "is_supervised": is_supervised,
            "contamination": contamination,
            "n_features": len(feature_names)
        }
        
        return X, y, feature_names, metadata
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        **kwargs
    ) -> Tuple:
        """이상 탐지 데이터 분할"""
        if y is not None:
            # 지도 학습: stratified split
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
        else:
            # 비지도 학습: 일반 split
            X_temp, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            y_temp, y_test = None, None
            
            if val_size > 0:
                val_ratio = val_size / (1 - test_size)
                X_train, X_val = train_test_split(
                    X_temp, test_size=val_ratio, random_state=random_state
                )
                y_train, y_val = None, None
            else:
                X_train, X_val, y_train, y_val = X_temp, None, None, None
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_default_models(self) -> List[str]:
        """이상 탐지 모델"""
        return [
            "isolation_forest",
            "one_class_svm",
            "local_outlier_factor",
            "elliptic_envelope",
            # 지도 학습의 경우 일반 분류 모델도 사용 가능
        ]
    
    def get_primary_metric(self) -> str:
        return "roc_auc"
    
    def evaluate_model(
        self,
        estimator,
        X_test,
        y_test,
        **kwargs
    ) -> Dict[str, Any]:
        """이상 탐지 모델 평가"""
        # predict: -1 (이상), 1 (정상)을 반환
        y_pred = estimator.predict(X_test)
        
        # -1을 1로, 1을 0으로 변환 (이상=1, 정상=0)
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        
        if y_test is not None:
            # 지도 학습: 실제 라벨과 비교
            return anomaly_detection_metrics(y_test, y_pred_binary)
        else:
            # 비지도 학습: 탐지된 이상치 비율만 반환
            anomaly_ratio = np.sum(y_pred_binary) / len(y_pred_binary)
            logger.info(f"탐지된 이상치 비율: {anomaly_ratio:.2%}")
            return {
                "detected_anomaly_ratio": anomaly_ratio,
                "n_anomalies": int(np.sum(y_pred_binary)),
                "n_normal": int(len(y_pred_binary) - np.sum(y_pred_binary))
            }