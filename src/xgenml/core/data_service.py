# src/xgenml/core/data_service.py
import os
from typing import Optional
import pandas as pd
from huggingface_hub import hf_hub_download
import logging
from mlflow.tracking import MlflowClient
import tempfile
import mlflow

logger = logging.getLogger("uvicorn.error")

class HFDataService:
    """
    전처리 완료 데이터(숫자/원핫, NaN 없음) 가정.
    parquet/csv 지원.
    """
    def load_dataset_data(
        self,
        repo_id: str,
        filename: str,
        revision: Optional[str] = None,
        repo_type: str = "dataset",                   # ✅ 기본값을 dataset으로
    ) -> pd.DataFrame:
        local_fp = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type=repo_type,                      # ✅ 중요!
            token=os.getenv("HUGGINGFACE_HUB_TOKEN")  # (옵션) private/gated용
        )
        if filename.endswith(".parquet"):
            return pd.read_parquet(local_fp)
        if filename.endswith(".csv"):
            return pd.read_csv(local_fp)
        raise ValueError("지원 포맷은 .parquet, .csv 입니다.")


class MLflowDataService:
    """MLflow에서 데이터셋 로딩"""
    
    def __init__(self):
        # MLflow S3 설정
        os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin123'
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://minio.x2bee.com'
        
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://polar-mlflow-git.x2bee.com")
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        logger.info(f"MLflow Data Service 초기화: {tracking_uri}")
    
    def load_dataset_from_run(
        self, 
        run_id: str, 
        artifact_path: str = "dataset"
    ) -> pd.DataFrame:
        """
        MLflow run에서 데이터셋 다운로드 및 로드
        
        Args:
            run_id: MLflow run ID
            artifact_path: 아티팩트 경로 (기본값: "dataset")
        
        Returns:
            pandas DataFrame
        """
        logger.info(f"MLflow 데이터셋 로딩 시작: run_id={run_id}, path={artifact_path}")
        
        try:
            # 아티팩트 목록 조회
            artifacts = self.client.list_artifacts(run_id, artifact_path)
            
            # 데이터셋 파일 찾기 (.csv 또는 .parquet)
            dataset_artifact = None
            for artifact in artifacts:
                if artifact.path.endswith('.csv') or artifact.path.endswith('.parquet'):
                    dataset_artifact = artifact
                    break
            
            if not dataset_artifact:
                raise ValueError(f"Run {run_id}의 {artifact_path}에서 데이터셋 파일을 찾을 수 없습니다")
            
            logger.info(f"데이터셋 파일 발견: {dataset_artifact.path}")
            
            # 임시 디렉토리에 다운로드
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = self.client.download_artifacts(
                    run_id,
                    dataset_artifact.path,
                    dst_path=temp_dir
                )
                
                logger.info(f"다운로드 완료: {local_path}")
                
                # 파일 형식에 따라 로드
                if dataset_artifact.path.endswith('.parquet'):
                    df = pd.read_parquet(local_path)
                    logger.info(f"Parquet 파일 로드: {df.shape}")
                else:  # CSV
                    df = pd.read_csv(local_path)
                    logger.info(f"CSV 파일 로드: {df.shape}")
                
                return df
                
        except Exception as e:
            logger.error(f"MLflow 데이터셋 로딩 실패: {e}")
            raise
    
    def get_dataset_info(self, run_id: str) -> dict:
        """데이터셋 메타정보 조회"""
        try:
            run = self.client.get_run(run_id)
            
            return {
                "run_id": run_id,
                "experiment_id": run.info.experiment_id,
                "artifact_uri": run.info.artifact_uri,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
        except Exception as e:
            logger.error(f"데이터셋 정보 조회 실패: {e}")
            return {}