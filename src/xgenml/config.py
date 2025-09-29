from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # MLflow Tracking
    MLFLOW_TRACKING_URI: str = ""
    
    # MLflow Artifact Storage (MinIO S3)
    MLFLOW_DEFAULT_ARTIFACT_ROOT: str = "s3://mlflow-artifacts"
    MLFLOW_S3_ENDPOINT_URL: str | None = None
    
    # S3/MinIO Credentials
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_DEFAULT_REGION: str = "ap-northeast-2"
    
    # Legacy (backward compatibility, 이제 필요 없음)
    ARTIFACT_BASE_URI: str = "file:///tmp/artifacts"
    S3_ENDPOINT_URL: str | None = None  # MLFLOW_S3_ENDPOINT_URL 사용 권장

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()