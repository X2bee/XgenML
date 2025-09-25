import os
from typing import Any, Dict, List, Optional
from ..config import settings
from ..core.train_many import train_from_hf

class TrainingService:
    """고수준 서비스: 입력 파라미터 → 학습 실행 → 결과 반환"""
    def run(
        self,
        model_id: str,
        task: str,
        hf_repo: str, hf_filename: str, hf_revision: Optional[str],
        target_column: str, feature_columns: Optional[List[str]],
        model_names: List[str], overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        test_size: float = 0.2, validation_size: float = 0.1,
        use_cv: bool = False, cv_folds: int = 5
    ) -> Dict[str, Any]:
        return train_from_hf(
            model_id=model_id, task=task,
            hf_repo=hf_repo, hf_filename=hf_filename, hf_revision=hf_revision,
            target_column=target_column, feature_columns=feature_columns,
            model_names=model_names, overrides=overrides,
            test_size=test_size, val_size=validation_size, use_cv=use_cv, cv_folds=cv_folds,
            mlflow_experiment=f"model_{model_id}",
            artifact_base_uri=settings.ARTIFACT_BASE_URI,
            storage_ctor_kwargs=dict(
                s3_endpoint_url=settings.S3_ENDPOINT_URL,
                access_key=settings.AWS_ACCESS_KEY_ID,
                secret_key=settings.AWS_SECRET_ACCESS_KEY,
                region=settings.AWS_DEFAULT_REGION,
            )
        )
