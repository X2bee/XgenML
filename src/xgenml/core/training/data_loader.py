# /src/xgenml/core/training/data_loader.py
import time
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..data_service import HFDataService, MLflowDataService
from ...utils.logger_config import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """
    데이터 로드 및 전처리 담당 클래스
    태스크별 데이터 준비 로직 분리
    """
    
    def __init__(self, task: str, task_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            task: 태스크 타입 (classification, regression, timeseries, anomaly_detection, clustering)
            task_config: 태스크별 설정
                - timeseries: {"lookback_window": 10, "forecast_horizon": 1, "time_column": "date"}
                - anomaly_detection: {"contamination": 0.1}
                - clustering: {"n_clusters": 3}
        """
        self.task = task
        self.task_config = task_config or {}
        self.hf_service = HFDataService()
        self.mlflow_service = MLflowDataService()
        
        # 라벨 인코딩 관련
        self.label_encoder: Optional[LabelEncoder] = None
        self.label_mapping: Optional[Dict] = None
        self.original_classes: Optional[List] = None
    
    def load_data(
        self,
        use_mlflow_dataset: bool = False,
        mlflow_run_id: Optional[str] = None,
        mlflow_artifact_path: str = "dataset",
        hf_repo: Optional[str] = None,
        hf_filename: Optional[str] = None,
        hf_revision: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """데이터 로드"""
        logger.info(f"\n📊 데이터 로드 중...")
        load_start_time = time.time()
        
        if use_mlflow_dataset:
            df, data_source_info = self._load_from_mlflow(
                mlflow_run_id, mlflow_artifact_path
            )
        else:
            df, data_source_info = self._load_from_huggingface(
                hf_repo, hf_filename, hf_revision
            )
        
        load_duration = time.time() - load_start_time
        logger.info(f"데이터 로드 완료 ({load_duration:.2f}초)")
        self._log_data_info(df)
        
        return df, data_source_info
    
    def _load_from_mlflow(
        self, run_id: str, artifact_path: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """MLflow에서 데이터 로드"""
        if not run_id:
            raise ValueError("MLflow 데이터셋 사용 시 mlflow_run_id가 필요합니다")
        
        logger.info(f"MLflow 데이터셋 사용: run_id={run_id}")
        dataset_info = self.mlflow_service.get_dataset_info(run_id)
        logger.info(f"데이터셋 정보: {dataset_info.get('tags', {}).get('dataset_name', 'Unknown')}")
        
        df = self.mlflow_service.load_dataset_from_run(run_id, artifact_path)
        
        data_source_info = {
            "source_type": "mlflow",
            "mlflow_run_id": run_id,
            "mlflow_artifact_path": artifact_path,
            "dataset_metrics": dataset_info.get('metrics', {}),
        }
        
        return df, data_source_info
    
    def _load_from_huggingface(
        self, repo: str, filename: str, revision: Optional[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """HuggingFace에서 데이터 로드"""
        if not repo or not filename:
            raise ValueError("HuggingFace 데이터셋 사용 시 hf_repo와 hf_filename이 필요합니다")
        
        logger.info(f"HuggingFace 데이터셋 사용: {repo}/{filename}")
        df = self.hf_service.load_dataset_data(repo, filename, revision)
        
        data_source_info = {
            "source_type": "huggingface",
            "hf_repo": repo,
            "hf_filename": filename,
            "hf_revision": revision or "default",
        }
        
        return df, data_source_info
    
    def _log_data_info(self, df: pd.DataFrame):
        """데이터 정보 로깅"""
        logger.info(f"데이터셋 형태: {df.shape}")
        logger.info(f"컬럼: {df.columns.tolist()}")
        logger.info(f"데이터 타입:\n{df.dtypes}")
        
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"결측치 발견:\n{null_counts[null_counts > 0]}")
        else:
            logger.info("결측치 없음")
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[Any, Any, List[str], Dict[str, Any]]:
        """
        태스크별 피처와 타겟 준비
        
        Returns:
            X: 피처
            y: 타겟
            feature_names: 피처 이름 목록
            metadata: 추가 메타데이터
        """
        logger.info(f"\n피처 준비 (task: {self.task})")
        
        if self.task == "classification":
            return self._prepare_classification(df, target_column, feature_columns)
        elif self.task == "regression":
            return self._prepare_regression(df, target_column, feature_columns)
        elif self.task == "timeseries":
            return self._prepare_timeseries(df, target_column, feature_columns)
        elif self.task == "anomaly_detection":
            return self._prepare_anomaly_detection(df, target_column, feature_columns)
        elif self.task == "clustering":
            return self._prepare_clustering(df, feature_columns)
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _prepare_classification(
        self, df: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str], Dict[str, Any]]:
        """분류 데이터 준비"""
        if not target_column or target_column not in df.columns:
            raise ValueError(f"타겟 컬럼 '{target_column}'이 필요합니다")
        
        self._log_target_info(df, target_column)
        
        # 피처 선택
        if feature_columns:
            X = df[feature_columns]
        else:
            X = df.drop(columns=[target_column])
        
        y = df[target_column]
        
        # 라벨 인코딩
        metadata = {}
        if y.dtype == 'object':
            logger.info("라벨 인코딩 수행")
            y = self._encode_labels(y)
            metadata = {
                "label_encoded": True,
                "original_classes": self.original_classes,
                "label_mapping": self.label_mapping,
                "encoder": self.label_encoder
            }
        else:
            metadata = {"label_encoded": False}
        
        feature_names = X.columns.tolist()
        logger.info(f"피처 형태: {X.shape}, 타겟 형태: {y.shape}")
        
        return X, y, feature_names, metadata
    
    def _prepare_regression(
        self, df: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str], Dict[str, Any]]:
        """회귀 데이터 준비"""
        if not target_column or target_column not in df.columns:
            raise ValueError(f"타겟 컬럼 '{target_column}'이 필요합니다")
        
        self._log_target_info(df, target_column)
        
        if feature_columns:
            X = df[feature_columns]
        else:
            X = df.drop(columns=[target_column])
        
        y = df[target_column].values
        feature_names = X.columns.tolist()
        
        metadata = {"label_encoded": False}
        logger.info(f"피처 형태: {X.shape}, 타겟 형태: {y.shape}")
        
        return X, y, feature_names, metadata
    
    def _prepare_timeseries(
        self, df: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """시계열 데이터 준비"""
        if not target_column or target_column not in df.columns:
            raise ValueError(f"타겟 컬럼 '{target_column}'이 필요합니다")
        
        lookback_window = self.task_config.get("lookback_window", 10)
        forecast_horizon = self.task_config.get("forecast_horizon", 1)
        time_column = self.task_config.get("time_column")
        
        logger.info(f"시계열 설정: lookback={lookback_window}, horizon={forecast_horizon}")
        
        # 시간 컬럼으로 정렬
        if time_column and time_column in df.columns:
            df = df.sort_values(time_column)
            logger.info(f"시간 컬럼 '{time_column}'으로 정렬")
        
        # 피처 선택
        if feature_columns:
            feature_cols = feature_columns
        else:
            feature_cols = [col for col in df.columns 
                          if col not in [target_column, time_column]]
        
        # 시계열 시퀀스 생성
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
        
        logger.info(f"시계열 시퀀스 생성 완료: X={X.shape}, y={y.shape}")
        
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
    
    def _prepare_anomaly_detection(
        self, df: pd.DataFrame, target_column: Optional[str], feature_columns: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[str], Dict[str, Any]]:
        """이상 탐지 데이터 준비"""
        contamination = self.task_config.get("contamination", 0.1)
        
        # 피처 선택
        if feature_columns:
            X = df[feature_columns]
        elif target_column and target_column in df.columns:
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
            logger.info(f"지도 학습 이상 탐지 (라벨 있음)")
            logger.info(f"이상치 비율: {actual_contamination:.2%}")
        else:
            logger.info(f"비지도 학습 이상 탐지 (라벨 없음)")
            logger.info(f"예상 이상치 비율: {contamination:.2%}")
        
        feature_names = X.columns.tolist()
        
        metadata = {
            "label_encoded": False,
            "is_supervised": is_supervised,
            "contamination": contamination,
            "n_features": len(feature_names)
        }
        
        logger.info(f"피처 형태: {X.shape}, 타겟: {'있음' if y is not None else '없음'}")
        
        return X, y, feature_names, metadata
    
    def _prepare_clustering(
        self, df: pd.DataFrame, feature_columns: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, None, List[str], Dict[str, Any]]:
        """클러스터링 데이터 준비"""
        n_clusters = self.task_config.get("n_clusters", 3)
        
        # 피처 선택
        if feature_columns:
            X = df[feature_columns]
        else:
            X = df.copy()
        
        feature_names = X.columns.tolist()
        
        metadata = {
            "label_encoded": False,
            "n_clusters": n_clusters,
            "n_features": len(feature_names)
        }
        
        logger.info(f"클러스터링 데이터: {X.shape}, 목표 클러스터 수: {n_clusters}")
        
        return X, None, feature_names, metadata
    
    def _log_target_info(self, df: pd.DataFrame, target_column: str):
        """타겟 컬럼 정보 로깅"""
        logger.info(f"타겟 컬럼 '{target_column}' 정보:")
        logger.info(f"  - 데이터 타입: {df[target_column].dtype}")
        logger.info(f"  - 고유값 개수: {df[target_column].nunique()}")
        logger.info(f"  - 고유값: {df[target_column].unique()[:10]}")
    
    def _encode_labels(self, y: pd.Series) -> np.ndarray:
        """라벨 인코딩"""
        self.original_classes = y.unique().tolist()
        logger.info(f"원본 라벨: {self.original_classes}")
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.label_mapping = dict(
            zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_)))
        )
        logger.info(f"변환된 라벨: {np.unique(y_encoded)}")
        logger.info(f"라벨 매핑: {self.label_mapping}")
        
        return y_encoded
    
    def split_data(
        self,
        X: Any,
        y: Optional[Any],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple:
        """
        태스크별 데이터 분할
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info(f"\n데이터 분할 (task: {self.task})")
        logger.info(f"test_size={test_size}, val_size={val_size}")
        
        if self.task == "timeseries":
            return self._split_timeseries(X, y, test_size, val_size)
        elif self.task == "anomaly_detection":
            return self._split_anomaly_detection(X, y, test_size, val_size, random_state)
        elif self.task == "clustering":
            return self._split_clustering(X, test_size, val_size, random_state)
        else:
            # classification, regression
            stratify = y if self.task == "classification" and y is not None else None
            return self._split_standard(X, y, test_size, val_size, random_state, stratify)
    
    def _split_standard(
        self, X, y, test_size, val_size, random_state, stratify
    ) -> Tuple:
        """표준 데이터 분할 (classification, regression)"""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        logger.info(f"Train+Val: {X_temp.shape}, Test: {X_test.shape}")
        
        if val_size > 0:
            val_ratio = val_size / (1 - test_size)
            stratify_temp = y_temp if stratify is not None else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio,
                random_state=random_state, stratify=stratify_temp
            )
            logger.info(f"Train: {X_train.shape}, Validation: {X_val.shape}")
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
            logger.info(f"Train: {X_train.shape}, Validation: 없음")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _split_timeseries(self, X, y, test_size, val_size) -> Tuple:
        """시계열 데이터 분할 (시간 순서 유지, 셔플 X)"""
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
    
    def _split_anomaly_detection(
        self, X, y, test_size, val_size, random_state
    ) -> Tuple:
        """이상 탐지 데이터 분할"""
        if y is not None:
            # 지도 학습: stratified split
            stratify = y
            return self._split_standard(X, y, test_size, val_size, random_state, stratify)
        else:
            # 비지도 학습: 일반 split (타겟 없음)
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
            
            logger.info(f"비지도 분할: Train={len(X_train)}, Test={len(X_test)}")
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _split_clustering(self, X, test_size, val_size, random_state) -> Tuple:
        """클러스터링 데이터 분할 (타겟 없음)"""
        X_temp, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        
        if val_size > 0:
            val_ratio = val_size / (1 - test_size)
            X_train, X_val = train_test_split(
                X_temp, test_size=val_ratio, random_state=random_state
            )
        else:
            X_train, X_val = X_temp, None
        
        logger.info(f"클러스터링 분할: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_val, X_test, None, None, None
    
    def get_label_encoding_info(self) -> Optional[Dict[str, Any]]:
        """라벨 인코딩 정보 반환"""
        if self.label_encoder is None:
            return None
        
        return {
            "used": True,
            "original_classes": self.original_classes,
            "label_mapping": self.label_mapping,
            "encoder": self.label_encoder
        }

    # Add to DataLoader class

    def get_input_schema(self, X, feature_names: List[str]) -> Dict[str, Any]:
        """입력 데이터 스키마 생성"""
        import pandas as pd
        import numpy as np
        
        schema = {
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "features": []
        }
        
        # 각 피처의 데이터 타입 및 통계 정보
        df = pd.DataFrame(X, columns=feature_names) if not isinstance(X, pd.DataFrame) else X
        
        for col in feature_names:
            feature_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "nullable": bool(df[col].isnull().any()),
                "null_count": int(df[col].isnull().sum())
            }
            
            # 수치형 데이터
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_info.update({
                    "type": "numeric",
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "median": float(df[col].median())
                })
            # 범주형 데이터
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
                unique_vals = df[col].unique()
                feature_info.update({
                    "type": "categorical",
                    "n_unique": len(unique_vals),
                    "categories": unique_vals.tolist()[:100]  # 처음 100개만
                })
            
            schema["features"].append(feature_info)
        
        return schema

    def get_output_schema(self, y, task: str, label_encoding_info: Optional[Dict] = None) -> Dict[str, Any]:
        """출력 데이터 스키마 생성"""
        import pandas as pd
        import numpy as np
        
        schema = {
            "type": task,
            "shape": y.shape if hasattr(y, 'shape') else (len(y),)
        }
        
        if task == "classification":
            if label_encoding_info and label_encoding_info.get("used"):
                # 라벨 인코딩이 적용된 경우
                schema.update({
                    "n_classes": len(label_encoding_info["original_classes"]),
                    "class_names": label_encoding_info["original_classes"],
                    "encoded": True,
                    "label_mapping": label_encoding_info["label_mapping"]
                })
            else:
                # 원본 라벨
                unique_classes = pd.Series(y).unique().tolist()
                schema.update({
                    "n_classes": len(unique_classes),
                    "class_names": unique_classes,
                    "encoded": False
                })
        
        elif task == "regression":
            y_series = pd.Series(y)
            schema.update({
                "min": float(y_series.min()),
                "max": float(y_series.max()),
                "mean": float(y_series.mean()),
                "std": float(y_series.std())
            })
        
        elif task == "clustering":
            schema.update({
                "n_samples": len(y) if y is not None else 0,
                "unsupervised": y is None
            })
        
        return schema