from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Storage & MLflow
    ARTIFACT_BASE_URI: str = "file:///tmp/artifacts"
    MLFLOW_TRACKING_URI: str = ""

    # S3/MinIO (optional)
    S3_ENDPOINT_URL: str | None = None
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_DEFAULT_REGION: str = "ap-northeast-2"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
