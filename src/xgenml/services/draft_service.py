"""
Draft Service
드래프트 관리를 위한 데이터베이스 서비스 (JSON 파일 기반)
"""
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from ..models.user_script_schemas import Draft, UserScriptRunConfig


class DraftService:
    """드래프트 관리 서비스"""

    def __init__(self, storage_path: str = "./data/drafts"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self._ensure_index()

    def _ensure_index(self):
        """인덱스 파일 초기화"""
        if not self.index_file.exists():
            self.index_file.write_text(json.dumps([], ensure_ascii=False), encoding='utf-8')

    def _load_index(self) -> List[Dict[str, Any]]:
        """인덱스 로드"""
        try:
            content = self.index_file.read_text(encoding='utf-8')
            return json.loads(content)
        except:
            return []

    def _save_index(self, index: List[Dict[str, Any]]):
        """인덱스 저장"""
        self.index_file.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    def _load_draft_file(self, draft_id: str) -> Optional[Dict[str, Any]]:
        """드래프트 파일 로드"""
        draft_file = self.storage_path / f"{draft_id}.json"
        if not draft_file.exists():
            return None
        try:
            content = draft_file.read_text(encoding='utf-8')
            return json.loads(content)
        except:
            return None

    def _save_draft_file(self, draft_id: str, draft_data: Dict[str, Any]):
        """드래프트 파일 저장"""
        draft_file = self.storage_path / f"{draft_id}.json"
        draft_file.write_text(
            json.dumps(draft_data, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8'
        )

    def _delete_draft_file(self, draft_id: str):
        """드래프트 파일 삭제"""
        draft_file = self.storage_path / f"{draft_id}.json"
        if draft_file.exists():
            draft_file.unlink()

    def create_draft(
        self,
        name: str,
        script_path: str,
        content: str,
        task: str,
        run_config: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """드래프트 생성"""
        draft_id = str(uuid4())
        now = datetime.utcnow()

        draft_data = {
            "id": draft_id,
            "name": name,
            "script_path": script_path,
            "content": content,
            "task": task,
            "run_config": run_config,
            "created_at": now.isoformat() + 'Z',
            "updated_at": now.isoformat() + 'Z',
            "last_validation_status": None,
            "last_validation_at": None,
            "user_id": user_id
        }

        # 파일 저장
        self._save_draft_file(draft_id, draft_data)

        # 인덱스 업데이트
        index = self._load_index()
        index.append({
            "id": draft_id,
            "name": name,
            "task": task,
            "created_at": draft_data["created_at"],
            "updated_at": draft_data["updated_at"],
            "last_validation_status": None
        })
        self._save_index(index)

        return draft_data

    def get_draft(self, draft_id: str) -> Optional[Dict[str, Any]]:
        """드래프트 조회 (content 포함)"""
        return self._load_draft_file(draft_id)

    def list_drafts(
        self,
        task: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        sort: str = "updated_at",
        order: str = "desc"
    ) -> Dict[str, Any]:
        """드래프트 목록 조회 (content 제외)"""
        index = self._load_index()

        # 필터링
        if task:
            index = [d for d in index if d.get("task") == task]

        # 정렬
        reverse = (order == "desc")
        if sort in ["created_at", "updated_at", "name"]:
            index.sort(key=lambda x: x.get(sort, ""), reverse=reverse)

        total = len(index)

        # 페이지네이션
        paginated = index[offset:offset + limit]

        has_more = (offset + limit) < total

        return {
            "drafts": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": has_more
        }

    def update_draft(
        self,
        draft_id: str,
        update_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """드래프트 수정 (부분 업데이트)"""
        draft = self._load_draft_file(draft_id)
        if not draft:
            return None

        # 업데이트 가능한 필드
        allowed_fields = ["name", "script_path", "content", "run_config"]

        for field in allowed_fields:
            if field in update_data and update_data[field] is not None:
                if field == "run_config":
                    # run_config는 부분 업데이트
                    current_config = draft.get("run_config", {})
                    current_config.update(update_data[field])
                    draft["run_config"] = current_config
                else:
                    draft[field] = update_data[field]

        # updated_at 갱신
        draft["updated_at"] = datetime.utcnow().isoformat() + 'Z'

        # 파일 저장
        self._save_draft_file(draft_id, draft)

        # 인덱스 업데이트
        index = self._load_index()
        for item in index:
            if item["id"] == draft_id:
                item["name"] = draft["name"]
                item["updated_at"] = draft["updated_at"]
                break
        self._save_index(index)

        return draft

    def delete_draft(self, draft_id: str) -> bool:
        """드래프트 삭제"""
        draft = self._load_draft_file(draft_id)
        if not draft:
            return False

        # 파일 삭제
        self._delete_draft_file(draft_id)

        # 인덱스에서 제거
        index = self._load_index()
        index = [d for d in index if d["id"] != draft_id]
        self._save_index(index)

        return True

    def clone_draft(
        self,
        draft_id: str,
        new_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """드래프트 복제"""
        original = self._load_draft_file(draft_id)
        if not original:
            return None

        # 이름 생성
        if new_name is None:
            # 자동 이름 생성
            existing_names = [d["name"] for d in self._load_index()]
            new_name = self._generate_clone_name(original["name"], existing_names)

        # 새 드래프트 생성
        new_draft = self.create_draft(
            name=new_name,
            script_path=original["script_path"],
            content=original["content"],
            task=original["task"],
            run_config=original["run_config"],
            user_id=original.get("user_id")
        )

        return new_draft

    def _generate_clone_name(self, original_name: str, existing_names: List[str]) -> str:
        """
        복제 시 자동 이름 생성

        예시:
        - "My Script" -> "My Script (복사본)"
        - "My Script (복사본)" -> "My Script (복사본 2)"
        - "My Script (복사본 2)" -> "My Script (복사본 3)"
        """
        pattern = r'^(.+?)(?: \(복사본(?: (\d+))?\))?$'
        match = re.match(pattern, original_name)

        if not match:
            base_name = original_name
            copy_number = None
        else:
            base_name = match.group(1)
            copy_number_str = match.group(2)
            copy_number = int(copy_number_str) if copy_number_str else None

        # 다음 복사본 번호 계산
        if copy_number is None:
            candidate = f"{base_name} (복사본)"
        else:
            next_number = copy_number + 1
            candidate = f"{base_name} (복사본 {next_number})"

        # 중복 확인 및 재귀적으로 번호 증가
        while candidate in existing_names:
            match = re.match(pattern, candidate)
            base = match.group(1)
            num_str = match.group(2)

            if num_str is None:
                candidate = f"{base} (복사본 2)"
            else:
                candidate = f"{base} (복사본 {int(num_str) + 1})"

        return candidate

    def update_validation_status(
        self,
        draft_id: str,
        status: str,
        validated_at: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """검증 상태 업데이트"""
        draft = self._load_draft_file(draft_id)
        if not draft:
            return None

        draft["last_validation_status"] = status
        draft["last_validation_at"] = (validated_at or datetime.utcnow()).isoformat() + 'Z'

        self._save_draft_file(draft_id, draft)

        # 인덱스 업데이트
        index = self._load_index()
        for item in index:
            if item["id"] == draft_id:
                item["last_validation_status"] = status
                break
        self._save_index(index)

        return draft


# 싱글톤 인스턴스
_draft_service = None


def get_draft_service() -> DraftService:
    """드래프트 서비스 싱글톤 인스턴스 가져오기"""
    global _draft_service
    if _draft_service is None:
        _draft_service = DraftService()
    return _draft_service
