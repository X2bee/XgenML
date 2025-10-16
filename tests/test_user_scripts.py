"""
User Script Management Tests
사용자 스크립트 관리 시스템 테스트
"""
import pytest
from fastapi.testclient import TestClient
from src.xgenml.api import app
from src.xgenml.services.script_validator import validate_script

client = TestClient(app)


# ========================
# 검증 테스트
# ========================
def test_validate_valid_script():
    """유효한 스크립트 검증 테스트"""
    valid_script = """
from types import SimpleNamespace

USER_SCRIPT_METADATA = {
    "name": "test_classifier",
    "display_name": "Test Classifier",
    "version": "1.0.0",
    "task": "classification",
    "description": "Test classification model",
    "tags": ["test"]
}

def train(config):
    print("Training started")

    result = SimpleNamespace(
        metrics={"accuracy": 0.95},
        warnings=[],
        errors=[],
        artifacts=[]
    )

    return result
"""

    is_valid, messages, metadata = validate_script(valid_script)

    assert is_valid == True
    assert metadata is not None
    assert metadata["name"] == "test_classifier"
    assert metadata["version"] == "1.0.0"
    assert metadata["task"] == "classification"


def test_validate_missing_metadata():
    """메타데이터 누락 스크립트 검증 테스트"""
    invalid_script = """
def train(config):
    return {}
"""

    is_valid, messages, metadata = validate_script(invalid_script)

    assert is_valid == False
    assert any("USER_SCRIPT_METADATA" in msg["message"] for msg in messages)


def test_validate_missing_train_function():
    """train 함수 누락 스크립트 검증 테스트"""
    invalid_script = """
USER_SCRIPT_METADATA = {
    "name": "test",
    "display_name": "Test",
    "version": "1.0.0",
    "task": "classification"
}

def some_other_function():
    pass
"""

    is_valid, messages, metadata = validate_script(invalid_script)

    assert is_valid == False
    assert any("train 함수" in msg["message"] for msg in messages)


def test_validate_blocked_module():
    """금지된 모듈 import 검증 테스트"""
    malicious_script = """
import os

USER_SCRIPT_METADATA = {
    "name": "test",
    "display_name": "Test",
    "version": "1.0.0",
    "task": "classification"
}

def train(config):
    os.system("rm -rf /")
    return {}
"""

    is_valid, messages, metadata = validate_script(malicious_script)

    assert is_valid == False
    assert any("금지된 모듈" in msg["message"] for msg in messages)


def test_validate_blocked_function():
    """금지된 함수 호출 검증 테스트"""
    malicious_script = """
USER_SCRIPT_METADATA = {
    "name": "test",
    "display_name": "Test",
    "version": "1.0.0",
    "task": "classification"
}

def train(config):
    eval("print('malicious code')")
    return {}
"""

    is_valid, messages, metadata = validate_script(malicious_script)

    assert is_valid == False
    assert any("금지된 함수" in msg["message"] for msg in messages)


# ========================
# API 엔드포인트 테스트
# ========================
def test_api_validate_script():
    """스크립트 검증 API 테스트"""
    valid_script = """
USER_SCRIPT_METADATA = {
    "name": "test_model",
    "display_name": "Test Model",
    "version": "1.0.0",
    "task": "classification",
    "description": "Test"
}

def train(config):
    from types import SimpleNamespace
    return SimpleNamespace(metrics={}, warnings=[], errors=[], artifacts=[])
"""

    response = client.post(
        "/api/scripts/validate",
        json={
            "script_path": "/tmp/test.py",
            "content": valid_script
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] == True
    assert data["metadata"] is not None


def test_api_create_draft():
    """드래프트 생성 API 테스트"""
    response = client.post(
        "/api/scripts/drafts",
        json={
            "name": "My Test Draft",
            "script_path": "/tmp/test.py",
            "content": "# test script",
            "task": "classification",
            "run_config": {
                "dataset_uri": "test://dataset",
                "target_column": "label",
                "feature_columns": ["f1", "f2"],
                "artifact_dir": "/tmp/artifacts",
                "random_seed": 42
            }
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["name"] == "My Test Draft"
    assert data["task"] == "classification"

    # 생성된 드래프트 ID 저장
    draft_id = data["id"]

    # 드래프트 조회
    response = client.get(f"/api/scripts/drafts/{draft_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == draft_id
    assert data["content"] == "# test script"

    # 드래프트 삭제
    response = client.delete(f"/api/scripts/drafts/{draft_id}")
    assert response.status_code == 200


def test_api_list_drafts():
    """드래프트 목록 조회 API 테스트"""
    response = client.get("/api/scripts/drafts")

    assert response.status_code == 200
    data = response.json()
    assert "drafts" in data
    assert "total" in data
    assert "has_more" in data


def test_api_update_draft():
    """드래프트 수정 API 테스트"""
    # 드래프트 생성
    create_response = client.post(
        "/api/scripts/drafts",
        json={
            "name": "Original Name",
            "script_path": "/tmp/test.py",
            "content": "# original",
            "task": "classification",
            "run_config": {
                "dataset_uri": "test://dataset",
                "target_column": "label",
                "feature_columns": [],
                "artifact_dir": "/tmp/artifacts",
                "random_seed": 42
            }
        }
    )
    draft_id = create_response.json()["id"]

    # 드래프트 수정
    update_response = client.put(
        f"/api/scripts/drafts/{draft_id}",
        json={"name": "Updated Name"}
    )

    assert update_response.status_code == 200
    data = update_response.json()
    assert data["name"] == "Updated Name"
    assert data["content"] == "# original"  # 변경 안 됨

    # 정리
    client.delete(f"/api/scripts/drafts/{draft_id}")


def test_api_clone_draft():
    """드래프트 복제 API 테스트"""
    # 드래프트 생성
    create_response = client.post(
        "/api/scripts/drafts",
        json={
            "name": "Original Draft",
            "script_path": "/tmp/test.py",
            "content": "# original content",
            "task": "classification",
            "run_config": {
                "dataset_uri": "test://dataset",
                "target_column": "label",
                "feature_columns": [],
                "artifact_dir": "/tmp/artifacts",
                "random_seed": 42
            }
        }
    )
    original_id = create_response.json()["id"]

    # 드래프트 복제
    clone_response = client.post(
        f"/api/scripts/drafts/{original_id}/clone",
        json={}
    )

    assert clone_response.status_code == 200
    cloned_data = clone_response.json()
    assert cloned_data["id"] != original_id
    assert cloned_data["name"] == "Original Draft (복사본)"
    assert cloned_data["content"] == "# original content"

    # 정리
    client.delete(f"/api/scripts/drafts/{original_id}")
    client.delete(f"/api/scripts/drafts/{cloned_data['id']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
