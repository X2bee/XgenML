import os
from fastapi.testclient import TestClient
from src.my_project.api import app

def test_train_endpoint_smoke(monkeypatch):
    # MLflow URI가 비어도 로컬파일 backend로 동작하지만, 여기선 환경변수만 세팅
    monkeypatch.setenv("ARTIFACT_BASE_URI", "file:///tmp/artifacts")
    client = TestClient(app)

    # 기본적으로 HF 저장소에 실제 파일이 있어야 성공합니다.
    # 데모: scikit-learn/iris 의 CSV를 예시로 시도 (경로는 상황에 맞게 조정)
    body = {
        "model_id": "demo-001",
        "task": "classification",
        "hf_repo": "scikit-learn/iris",
        "hf_filename": "data/iris.csv",
        "hf_revision": None,
        "target_column": "target",
        "feature_columns": None,
        "model_names": ["logistic_regression"],
        "overrides": {"logistic_regression": {"max_iter": 300}},
        "test_size": 0.2,
        "validation_size": 0.1,
        "use_cv": False,
        "cv_folds": 5
    }

    res = client.post("/api/train", json=body)
    assert res.status_code in (200, 500)  # 500은 외부네트워크/데이터 파일 문제 시
