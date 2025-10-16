import os
from typing import Any, Dict, List, Optional
from src.xgenml.config import settings
from src.xgenml.core.train_many import train_from_hf

# /src/xgenml/services/training_service.py
class TrainingService:
    """고수준 서비스: 입력 파라미터 → 학습 실행 → 결과 반환"""
    
    def run(
        self,
        model_id: str,
        task: str,
        # HuggingFace 관련
        hf_repo: Optional[str] = None,
        hf_filename: Optional[str] = None,
        hf_revision: Optional[str] = None,
        # MLflow 관련 (새로 추가)
        use_mlflow_dataset: bool = False,
        mlflow_run_id: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = "dataset",
        # 나머지
        target_column: str = None,
        feature_columns: Optional[List[str]] = None,
        model_names: List[str] = None,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        use_cv: bool = False,
        cv_folds: int = 5,
        hpo_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        
        final_artifact_path = mlflow_artifact_path if mlflow_artifact_path is not None else "dataset"

        return train_from_hf(
            model_id=model_id,
            task=task,
            # HuggingFace
            hf_repo=hf_repo,
            hf_filename=hf_filename,
            hf_revision=hf_revision,
            # MLflow
            use_mlflow_dataset=use_mlflow_dataset,
            mlflow_run_id=mlflow_run_id,
            mlflow_artifact_path=final_artifact_path,
            # 나머지
            target_column=target_column,
            feature_columns=feature_columns,
            model_names=model_names,
            overrides=overrides,
            test_size=test_size,
            val_size=validation_size,
            use_cv=use_cv,
            cv_folds=cv_folds,
            hpo_config=hpo_config,
            mlflow_experiment=f"model_{model_id}",
            artifact_base_uri=settings.ARTIFACT_BASE_URI,
            storage_ctor_kwargs=dict(
                s3_endpoint_url=settings.MLFLOW_S3_ENDPOINT_URL,
                access_key=settings.AWS_ACCESS_KEY_ID,
                secret_key=settings.AWS_SECRET_ACCESS_KEY,
                region=settings.AWS_DEFAULT_REGION,
            )
        )