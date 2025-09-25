# my_project

전처리 완료 데이터(HF)로 여러 모델을 학습/평가/로깅하고, 로컬/S3/MinIO에 모델을 저장하는 최소 파이프라인.

## Quickstart

```bash
pip install -e '.[dev]'
cp .env.example .env
# .env에서 ARTIFACT_BASE_URI, MLFLOW_TRACKING_URI 확인
python -m src.xgenml.main  # FastAPI 서버: http://localhost:8000
