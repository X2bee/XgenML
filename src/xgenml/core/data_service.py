# src/xgenml/core/data_service.py
import os
from typing import Optional
import pandas as pd
from huggingface_hub import hf_hub_download

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