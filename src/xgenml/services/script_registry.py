"""
Script Registry Service
등록된 사용자 스크립트 관리 서비스
"""
import json
import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4


def _get_default_registry_path() -> str:
    """기본 레지스트리 경로 결정"""
    if "XGENML_SCRIPT_REGISTRY" in os.environ:
        return os.environ["XGENML_SCRIPT_REGISTRY"]

    # 프로젝트 루트 찾기
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent

    # 프로젝트 루트에 user_scripts 디렉토리 생성 시도
    scripts_dir = project_root / "data" / "user_scripts"

    try:
        scripts_dir.mkdir(parents=True, exist_ok=True)
        # 쓰기 권한 테스트
        test_file = scripts_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        return str(scripts_dir)
    except (OSError, PermissionError):
        # 권한 없으면 임시 디렉토리 사용
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "xgenml-user-scripts"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return str(temp_dir)


class ScriptRegistry:
    """스크립트 레지스트리 관리"""

    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            registry_path = _get_default_registry_path()

        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # 카탈로그 파일
        self.catalog_file = self.registry_path / "catalog.json"
        self._ensure_catalog()

    def _ensure_catalog(self):
        """카탈로그 파일 초기화"""
        if not self.catalog_file.exists():
            initial_catalog = {
                "version": "1.0.0",
                "updated_at": datetime.utcnow().isoformat() + 'Z',
                "scripts": []
            }
            self.catalog_file.write_text(
                json.dumps(initial_catalog, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )

    def _load_catalog(self) -> Dict[str, Any]:
        """카탈로그 로드"""
        try:
            content = self.catalog_file.read_text(encoding='utf-8')
            return json.loads(content)
        except:
            return {
                "version": "1.0.0",
                "updated_at": datetime.utcnow().isoformat() + 'Z',
                "scripts": []
            }

    def _save_catalog(self, catalog: Dict[str, Any]):
        """카탈로그 저장"""
        catalog["updated_at"] = datetime.utcnow().isoformat() + 'Z'
        self.catalog_file.write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    def _compute_checksum(self, content: str) -> str:
        """스크립트 체크섬 계산 (SHA256)"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def register_script(
        self,
        content: str,
        metadata: Dict[str, Any],
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        스크립트 등록

        Args:
            content: 스크립트 내용
            metadata: 메타데이터 (name, version, task, etc.)
            overwrite: 덮어쓰기 허용 여부

        Returns:
            등록된 스크립트 정보

        Raises:
            ValueError: 중복된 name+version (overwrite=False일 경우)
        """
        name = metadata["name"]
        version = metadata["version"]

        # 중복 검사
        if self.script_exists(name, version):
            if not overwrite:
                raise ValueError(f"스크립트 '{name}' 버전 '{version}'이(가) 이미 존재합니다.")
            else:
                # 기존 스크립트 삭제
                self.delete_script(name, version)

        # 체크섬 계산
        checksum = self._compute_checksum(content)

        # 파일명 생성: {name}_v{version}_{short_checksum}.py
        short_checksum = checksum[:8]
        filename = f"{name}_v{version}_{short_checksum}.py"
        script_path = self.registry_path / filename

        # 스크립트 저장
        script_path.write_text(content, encoding='utf-8')

        # 카탈로그 엔트리 생성
        catalog_entry = {
            "id": str(uuid4()),
            "name": name,
            "display_name": metadata.get("display_name", name),
            "version": version,
            "task": metadata["task"],
            "description": metadata.get("description", ""),
            "tags": metadata.get("tags", []) + ["user_script"],
            "script_path": str(script_path.relative_to(self.registry_path)),
            "absolute_path": str(script_path),
            "checksum": checksum,
            "size_bytes": len(content.encode('utf-8')),
            "registered_at": datetime.utcnow().isoformat() + 'Z',
            "metadata": metadata
        }

        # 카탈로그 업데이트
        catalog = self._load_catalog()
        catalog["scripts"].append(catalog_entry)
        self._save_catalog(catalog)

        return catalog_entry

    def script_exists(self, name: str, version: str) -> bool:
        """스크립트 존재 여부 확인"""
        catalog = self._load_catalog()
        for script in catalog["scripts"]:
            if script["name"] == name and script["version"] == version:
                return True
        return False

    def get_script(self, name: str, version: str) -> Optional[Dict[str, Any]]:
        """스크립트 정보 조회"""
        catalog = self._load_catalog()
        for script in catalog["scripts"]:
            if script["name"] == name and script["version"] == version:
                return script
        return None

    def get_script_content(self, name: str, version: str) -> Optional[str]:
        """스크립트 내용 읽기"""
        script_info = self.get_script(name, version)
        if not script_info:
            return None

        script_path = Path(script_info["absolute_path"])
        if not script_path.exists():
            return None

        return script_path.read_text(encoding='utf-8')

    def list_scripts(
        self,
        task: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """스크립트 목록 조회"""
        catalog = self._load_catalog()
        scripts = catalog["scripts"]

        # 필터링
        if task:
            scripts = [s for s in scripts if s["task"] == task]

        if tags:
            scripts = [
                s for s in scripts
                if any(tag in s.get("tags", []) for tag in tags)
            ]

        return scripts

    def delete_script(self, name: str, version: str) -> bool:
        """스크립트 삭제"""
        script_info = self.get_script(name, version)
        if not script_info:
            return False

        # 파일 삭제
        script_path = Path(script_info["absolute_path"])
        if script_path.exists():
            script_path.unlink()

        # 카탈로그에서 제거
        catalog = self._load_catalog()
        catalog["scripts"] = [
            s for s in catalog["scripts"]
            if not (s["name"] == name and s["version"] == version)
        ]
        self._save_catalog(catalog)

        return True

    def update_script_metadata(
        self,
        name: str,
        version: str,
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """스크립트 메타데이터 업데이트 (description, tags 등)"""
        catalog = self._load_catalog()

        for script in catalog["scripts"]:
            if script["name"] == name and script["version"] == version:
                # 업데이트 가능한 필드만 수정
                allowed_fields = ["description", "tags", "display_name"]
                for field in allowed_fields:
                    if field in updates:
                        script[field] = updates[field]

                script["updated_at"] = datetime.utcnow().isoformat() + 'Z'
                self._save_catalog(catalog)
                return script

        return None

    def get_catalog_summary(self) -> Dict[str, Any]:
        """카탈로그 요약 정보"""
        catalog = self._load_catalog()
        scripts = catalog["scripts"]

        # 태스크별 집계
        tasks = {}
        for script in scripts:
            task = script["task"]
            tasks[task] = tasks.get(task, 0) + 1

        return {
            "version": catalog["version"],
            "updated_at": catalog["updated_at"],
            "total_scripts": len(scripts),
            "tasks": tasks
        }


# 싱글톤 인스턴스
_script_registry = None


def get_script_registry() -> ScriptRegistry:
    """스크립트 레지스트리 싱글톤 인스턴스 가져오기"""
    global _script_registry
    if _script_registry is None:
        _script_registry = ScriptRegistry()
    return _script_registry
