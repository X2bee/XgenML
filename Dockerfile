# =========
# Base deps
# =========
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 러닝타임 유틸 & 보안 업데이트
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

# 가상환경(선택) 대신 시스템 Python 사용
WORKDIR /app

# ============
# Builder step
# ============
FROM base AS builder

# 빌드 휠 캐시 최적화: 먼저 메타만 복사
COPY pyproject.toml README.md ./
# 선택: requirements.txt가 있다면 함께 복사 (배포 환경에서 핀 고정 시)
# COPY requirements.txt ./

# 빌드에 필요한 헤더가 필요하면 아래 주석 해제
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc \
#  && rm -rf /var/lib/apt/lists/*

# 의존성 설치 (프로덕션만)
RUN pip install --upgrade pip \
 && pip install --no-cache-dir .

# 소스 코드 복사
COPY src ./src

# (테스트/개발 의존성을 컨테이너 안에서 쓰고 싶다면 아래를 사용)
# RUN pip install --no-cache-dir ".[dev]"

# ============
# Runtime step
# ============
FROM base AS runtime

# 비-root 실행 유저
RUN useradd -m -u 10001 appuser
USER appuser

WORKDIR /app

# 빌더 레이어에서 설치된 site-packages만 복사
# (슬림 이미지를 위해 소스 전체가 아닌 실행 산출물만 가져옵니다)
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# 앱 소스도 필요 (리로딩 없이 실행하는 프로덕션 가정)
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser pyproject.toml README.md ./

# 기본 환경 변수 (필요에 맞게 docker run 시 덮어쓰기)
ENV ARTIFACT_BASE_URI="file:///tmp/artifacts" \
    MLFLOW_TRACKING_URI="" \
    S3_ENDPOINT_URL="" \
    AWS_ACCESS_KEY_ID="" \
    AWS_SECRET_ACCESS_KEY="" \
    AWS_DEFAULT_REGION="ap-northeast-2" \
    PORT=8001

# 헬스체크 (FastAPI 루트 또는 /docs 기준으로 체크)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/openapi.json" || exit 1

# 네트워크 포트
EXPOSE 8001

# tini 로 zombie reaping
ENTRYPOINT ["/usr/bin/tini", "--"]

# uvicorn 실행 (패키지 엔트리: src/xgenml/main.py의 FastAPI 앱은 src/xgenml/api/__init__.py 내 app)
CMD ["uvicorn", "src.xgenml.api:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
