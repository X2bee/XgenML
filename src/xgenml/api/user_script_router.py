"""
User Script Router
사용자 스크립트 관리 API 엔드포인트
"""
from fastapi import APIRouter, HTTPException, status, Query
from typing import Optional
from datetime import datetime

from ..models.user_script_schemas import (
    ScriptValidateRequest,
    ScriptValidateResponse,
    ScriptExecuteRequest,
    ScriptExecuteResponse,
    ScriptRegisterRequest,
    ScriptRegisterResponse,
    DraftCreateRequest,
    DraftUpdateRequest,
    DraftCloneRequest,
    DraftListResponse,
    DraftDeleteResponse,
    ValidationMessage,
    CatalogEntry,
    Draft,
)
from ..services.script_validator import validate_script
from ..services.script_executor import execute_script
from ..services.draft_service import get_draft_service
from ..services.script_registry import get_script_registry

router = APIRouter()


# ========================
# 스크립트 검증
# ========================
@router.post("/scripts/validate", response_model=ScriptValidateResponse)
async def validate_user_script(request: ScriptValidateRequest):
    """
    사용자 스크립트 검증

    - 구문 검증
    - 메타데이터 검증
    - 보안 검증
    """
    try:
        # 크기 제한 검증 (1MB)
        if len(request.content) > 1048576:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": {
                        "message": "스크립트 크기가 제한을 초과했습니다.",
                        "detail": "최대 1MB까지 허용됩니다.",
                        "code": "SCRIPT_SIZE_EXCEEDED"
                    }
                }
            )

        # 검증 실행
        is_valid, messages, metadata = validate_script(request.content)

        return ScriptValidateResponse(
            is_valid=is_valid,
            messages=[ValidationMessage(**msg) for msg in messages],
            metadata=metadata,
            validated_at=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "검증 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


# ========================
# 스크립트 실행
# ========================
@router.post("/scripts/execute", response_model=ScriptExecuteResponse)
async def execute_user_script(request: ScriptExecuteRequest):
    """
    샌드박스 환경에서 사용자 스크립트 실행

    - 리소스 제한 (메모리 2GB, CPU 10분)
    - 네트워크 차단
    - 타임아웃 설정
    """
    try:
        # 크기 제한 검증
        if len(request.content) > 1048576:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": {
                        "message": "스크립트 크기가 제한을 초과했습니다.",
                        "detail": "최대 1MB까지 허용됩니다.",
                        "code": "SCRIPT_SIZE_EXCEEDED"
                    }
                }
            )

        # 실행 설정 준비
        run_config = request.run_config.model_dump()

        # 스크립트 실행
        result = execute_script(
            script_content=request.content,
            run_config=run_config,
            timeout_seconds=300  # 5분
        )

        # 타임아웃 체크
        if result.get("exit_code") == -1 and "시간 제한" in str(result.get("stderr", [])):
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail={
                    "error": {
                        "message": "스크립트 실행 시간이 제한을 초과했습니다.",
                        "detail": "최대 실행 시간은 5분입니다.",
                        "code": "EXECUTION_TIMEOUT"
                    }
                }
            )

        return ScriptExecuteResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "실행 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


# ========================
# 스크립트 등록
# ========================
@router.post("/scripts/register", response_model=ScriptRegisterResponse)
async def register_user_script(
    request: ScriptRegisterRequest,
    overwrite: bool = Query(False, description="같은 이름과 버전의 스크립트가 존재할 경우 덮어쓸지 여부")
):
    """
    검증된 스크립트를 모델 카탈로그에 등록

    - 중복 검사 (name + version)
    - 파일 저장
    - 카탈로그 업데이트
    """
    try:
        # 크기 제한 검증
        if len(request.content) > 1048576:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": {
                        "message": "스크립트 크기가 제한을 초과했습니다.",
                        "detail": "최대 1MB까지 허용됩니다.",
                        "code": "SCRIPT_SIZE_EXCEEDED"
                    }
                }
            )

        # 검증 실행
        is_valid, messages, metadata = validate_script(request.content)

        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "message": "스크립트 검증에 실패했습니다.",
                        "detail": f"{len(messages)}개의 검증 오류가 있습니다.",
                        "code": "REGISTRATION_NOT_VALIDATED"
                    }
                }
            )

        # 메타데이터 추출
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "message": "메타데이터를 추출할 수 없습니다.",
                        "detail": "USER_SCRIPT_METADATA를 확인해주세요.",
                        "code": "REGISTRATION_INVALID_METADATA"
                    }
                }
            )

        # 스크립트 레지스트리에 등록
        registry = get_script_registry()

        # 덮어쓰기 전 존재 여부 확인
        was_existing = registry.script_exists(metadata["name"], metadata["version"])

        try:
            registered_script = registry.register_script(
                content=request.content,
                metadata=metadata,
                overwrite=overwrite
            )
        except ValueError as e:
            # 중복 스크립트
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": {
                        "message": "스크립트 등록 실패",
                        "detail": str(e),
                        "code": "REGISTRATION_DUPLICATE"
                    }
                }
            )

        # 응답 생성
        catalog_entry = CatalogEntry(
            name=registered_script["name"],
            display_name=registered_script["display_name"],
            tags=registered_script["tags"],
            script_path=registered_script["script_path"],
            version=registered_script["version"],
            task=registered_script["task"],
            description=registered_script["description"]
        )

        if overwrite and was_existing:
            message = f"스크립트 '{catalog_entry.name}' 버전 '{catalog_entry.version}'을(를) 덮어썼습니다."
        else:
            message = "스크립트가 성공적으로 등록되었습니다."

        return ScriptRegisterResponse(
            catalog_entry=catalog_entry,
            message=message,
            registered_at=datetime.fromisoformat(registered_script["registered_at"].replace('Z', '+00:00'))
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "등록 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


# ========================
# 드래프트 생성
# ========================
@router.post("/scripts/drafts", response_model=Draft)
async def create_draft(request: DraftCreateRequest):
    """
    새 드래프트 생성

    - UUID 자동 생성
    - 타임스탬프 자동 설정
    """
    try:
        draft_service = get_draft_service()

        draft_data = draft_service.create_draft(
            name=request.name,
            script_path=request.script_path,
            content=request.content,
            task=request.task,
            run_config=request.run_config.model_dump()
        )

        return Draft(**draft_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "드래프트 생성 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


# ========================
# 드래프트 목록 조회
# ========================
@router.get("/scripts/drafts", response_model=DraftListResponse)
async def list_drafts(
    task: Optional[str] = Query(None, description="태스크 필터"),
    limit: int = Query(50, ge=1, le=100, description="결과 개수 제한"),
    offset: int = Query(0, ge=0, description="페이지네이션 오프셋"),
    sort: str = Query("updated_at", description="정렬 기준"),
    order: str = Query("desc", description="정렬 순서")
):
    """
    드래프트 목록 조회 (content 제외)

    - 필터링 (task)
    - 정렬 (created_at, updated_at, name)
    - 페이지네이션
    """
    try:
        draft_service = get_draft_service()

        result = draft_service.list_drafts(
            task=task,
            limit=limit,
            offset=offset,
            sort=sort,
            order=order
        )

        return DraftListResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "드래프트 목록 조회 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


# ========================
# 드래프트 상세 조회
# ========================
@router.get("/scripts/drafts/{draft_id}", response_model=Draft)
async def get_draft(draft_id: str):
    """
    드래프트 상세 조회 (content 포함)
    """
    try:
        draft_service = get_draft_service()

        draft_data = draft_service.get_draft(draft_id)

        if not draft_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": "드래프트를 찾을 수 없습니다.",
                        "detail": f"ID: {draft_id}",
                        "code": "DRAFT_NOT_FOUND"
                    }
                }
            )

        return Draft(**draft_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "드래프트 조회 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


# ========================
# 드래프트 수정
# ========================
@router.put("/scripts/drafts/{draft_id}", response_model=Draft)
async def update_draft(draft_id: str, request: DraftUpdateRequest):
    """
    드래프트 수정 (부분 업데이트)

    - 보내지 않은 필드는 기존 값 유지
    - updated_at 자동 갱신
    """
    try:
        draft_service = get_draft_service()

        # 업데이트할 데이터 준비
        update_data = request.model_dump(exclude_none=True)

        draft_data = draft_service.update_draft(draft_id, update_data)

        if not draft_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": "드래프트를 찾을 수 없습니다.",
                        "detail": f"ID: {draft_id}",
                        "code": "DRAFT_NOT_FOUND"
                    }
                }
            )

        return Draft(**draft_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "드래프트 수정 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


# ========================
# 드래프트 삭제
# ========================
@router.delete("/scripts/drafts/{draft_id}", response_model=DraftDeleteResponse)
async def delete_draft(draft_id: str):
    """
    드래프트 삭제
    """
    try:
        draft_service = get_draft_service()

        success = draft_service.delete_draft(draft_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": "드래프트를 찾을 수 없습니다.",
                        "detail": f"ID: {draft_id}",
                        "code": "DRAFT_NOT_FOUND"
                    }
                }
            )

        return DraftDeleteResponse(
            message="드래프트가 삭제되었습니다.",
            deleted_id=draft_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "드래프트 삭제 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


# ========================
# 드래프트 복제
# ========================
@router.post("/scripts/drafts/{draft_id}/clone", response_model=Draft)
async def clone_draft(draft_id: str, request: DraftCloneRequest = DraftCloneRequest()):
    """
    드래프트 복제

    - 자동 이름 생성 (선택적)
    - 타임스탬프 초기화
    - validation 상태 초기화
    """
    try:
        draft_service = get_draft_service()

        draft_data = draft_service.clone_draft(
            draft_id=draft_id,
            new_name=request.name
        )

        if not draft_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": "원본 드래프트를 찾을 수 없습니다.",
                        "detail": f"ID: {draft_id}",
                        "code": "DRAFT_NOT_FOUND"
                    }
                }
            )

        return Draft(**draft_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "드래프트 복제 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


# ========================
# 스크립트 카탈로그 조회
# ========================
@router.get("/scripts/catalog")
async def get_script_catalog(
    task: Optional[str] = Query(None, description="태스크 필터"),
    tags: Optional[str] = Query(None, description="태그 필터 (콤마 구분)")
):
    """
    등록된 스크립트 카탈로그 조회

    - 태스크별 필터링
    - 태그별 필터링
    """
    try:
        registry = get_script_registry()

        # 태그 파싱
        tag_list = None
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        scripts = registry.list_scripts(task=task, tags=tag_list)

        return {
            "scripts": scripts,
            "total": len(scripts)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "카탈로그 조회 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


@router.get("/scripts/catalog/summary")
async def get_catalog_summary():
    """
    카탈로그 요약 정보

    - 총 스크립트 수
    - 태스크별 통계
    """
    try:
        registry = get_script_registry()
        summary = registry.get_catalog_summary()
        return summary

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "카탈로그 요약 조회 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


@router.get("/scripts/catalog/{name}/{version}")
async def get_script_info(name: str, version: str):
    """
    특정 스크립트 정보 조회
    """
    try:
        registry = get_script_registry()
        script_info = registry.get_script(name, version)

        if not script_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": "스크립트를 찾을 수 없습니다.",
                        "detail": f"name: {name}, version: {version}",
                        "code": "SCRIPT_NOT_FOUND"
                    }
                }
            )

        return script_info

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "스크립트 정보 조회 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


@router.get("/scripts/catalog/{name}/{version}/content")
async def get_script_content(name: str, version: str):
    """
    스크립트 내용 조회
    """
    try:
        registry = get_script_registry()
        content = registry.get_script_content(name, version)

        if content is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": "스크립트를 찾을 수 없습니다.",
                        "detail": f"name: {name}, version: {version}",
                        "code": "SCRIPT_NOT_FOUND"
                    }
                }
            )

        return {
            "name": name,
            "version": version,
            "content": content
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "스크립트 내용 조회 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )


@router.delete("/scripts/catalog/{name}/{version}")
async def delete_script(name: str, version: str):
    """
    등록된 스크립트 삭제
    """
    try:
        registry = get_script_registry()
        success = registry.delete_script(name, version)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": "스크립트를 찾을 수 없습니다.",
                        "detail": f"name: {name}, version: {version}",
                        "code": "SCRIPT_NOT_FOUND"
                    }
                }
            )

        return {
            "message": "스크립트가 삭제되었습니다.",
            "name": name,
            "version": version
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "스크립트 삭제 중 오류가 발생했습니다.",
                    "detail": str(e),
                    "code": "INTERNAL_ERROR"
                }
            }
        )
