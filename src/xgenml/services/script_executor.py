"""
Script Executor Service
사용자 스크립트를 샌드박스 환경에서 실행
"""
import subprocess
import json
import shutil
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any, Optional
import tempfile
import time


class ScriptExecutor:
    """스크립트 실행 클래스"""

    def __init__(
        self,
        max_memory_mb: int = 2048,
        max_cpu_time_seconds: int = 600,
        timeout_seconds: int = 300
    ):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time_seconds = max_cpu_time_seconds
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        script_content: str,
        run_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        샌드박스 환경에서 스크립트 실행

        Args:
            script_content: 스크립트 내용
            run_config: 실행 설정

        Returns:
            실행 결과 딕셔너리
        """
        # 임시 작업 디렉토리 생성
        work_dir = Path(tempfile.gettempdir()) / "xgenml-sandbox" / uuid4().hex
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 사용자 스크립트 저장
            user_script_path = work_dir / "user_script.py"
            user_script_path.write_text(script_content, encoding='utf-8')

            # 실행 래퍼 스크립트 생성
            runner_script = self._create_runner_script(work_dir)
            runner_script_path = work_dir / "runner.py"
            runner_script_path.write_text(runner_script, encoding='utf-8')

            # 디버깅: runner 스크립트의 UserScriptMetadata 부분 출력
            import sys
            lines = runner_script.split('\n')
            for i, line in enumerate(lines, 1):
                if 'class UserScriptMetadata' in line:
                    print(f"\n[DEBUG] Runner script UserScriptMetadata class (lines {i}-{i+10}):", file=sys.stderr)
                    for j in range(i-1, min(i+10, len(lines))):
                        print(f"  {j+1}: {lines[j]}", file=sys.stderr)
                    break

            # 설정 파일 저장
            config_path = work_dir / "config.json"
            config_path.write_text(json.dumps(run_config), encoding='utf-8')

            # artifact_dir 생성
            artifact_dir = Path(run_config.get('artifact_dir', work_dir / 'artifacts'))
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # 실행
            start_time = datetime.utcnow()
            result = self._run_subprocess(runner_script_path, work_dir)
            end_time = datetime.utcnow()

            duration = (end_time - start_time).total_seconds()

            # 출력 파싱
            execution_result = self._parse_output(
                result['stdout'],
                result['stderr'],
                result['returncode'],
                start_time,
                end_time,
                duration
            )

            return execution_result

        except subprocess.TimeoutExpired as e:
            # 타임아웃 시에도 부분적인 출력을 캡처
            stdout_output = e.stdout if hasattr(e, 'stdout') and e.stdout else ""
            stderr_output = e.stderr if hasattr(e, 'stderr') and e.stderr else ""
            return {
                "stdout": [line for line in stdout_output.split('\n') if line.strip()],
                "stderr": [f"실행 시간 제한({self.timeout_seconds}초)을 초과했습니다."] + [line for line in stderr_output.split('\n') if line.strip()],
                "result": None,
                "duration_seconds": self.timeout_seconds,
                "started_at": datetime.utcnow().isoformat() + 'Z',
                "finished_at": datetime.utcnow().isoformat() + 'Z',
                "exit_code": -1,
                "resource_usage": None
            }
        except Exception as e:
            # 상세한 traceback 포함
            import traceback
            error_traceback = traceback.format_exc()
            return {
                "stdout": [],
                "stderr": [
                    f"실행 중 오류 발생: {str(e)}",
                    "=" * 60,
                    "Traceback:",
                    error_traceback
                ],
                "result": None,
                "duration_seconds": 0,
                "started_at": datetime.utcnow().isoformat() + 'Z',
                "finished_at": datetime.utcnow().isoformat() + 'Z',
                "exit_code": -1,
                "resource_usage": None
            }
        finally:
            # 정리 - 디버깅을 위해 실패 시 디렉토리를 남길 수 있도록 개선
            try:
                # 성공적인 실행 또는 명시적으로 정리가 필요한 경우만 삭제
                # 실패 시에는 로그에 경로를 출력하고 일정 시간 후에 삭제
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception as cleanup_error:
                # 정리 실패는 무시하지만 로그에는 남김
                import sys
                print(f"Warning: Failed to clean up {work_dir}: {cleanup_error}", file=sys.stderr)

    def _create_runner_script(self, work_dir: Path) -> str:
        """실행 래퍼 스크립트 생성"""
        # f-string과 코드 내부의 중괄호 충돌을 피하기 위해 .format() 사용
        return '''
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# 초기 에러 캡처를 위한 플래그
_early_error = None

try:
    # pandas/numpy를 먼저 import (sys.path 수정 전에)
    import pandas as pd
    import numpy as np
except ImportError as e:
    _early_error = "Failed to import required libraries: " + str(e)
    print(_early_error, file=sys.stderr)
    sys.exit(1)

# UserScript 스키마 클래스 정의 (사용자 스크립트에서 사용할 수 있도록)
class UserScriptMetadata:
    def __init__(self, name: str, display_name: str, version: str, description: str = "", tags: List[str] = None, task: str = "classification"):
        self.name = name
        self.display_name = display_name
        self.version = version
        self.description = description
        self.tags = tags or []
        self.task = task

class Artifact:
    def __init__(self, name: str, path: str, size_bytes: int, type: str, created_at: datetime):
        self.name = name
        self.path = path
        self.size_bytes = size_bytes
        self.type = type
        self.created_at = created_at

class UserScriptResult:
    def __init__(self, metrics: Dict[str, float], warnings: List[str] = None, errors: List[str] = None, artifacts: List[Artifact] = None, model: Any = None):
        self.metrics = metrics
        self.warnings = warnings or []
        self.errors = errors or []
        self.artifacts = artifacts or []
        self.model = model

# 전역 네임스페이스에 추가 (사용자 스크립트가 import할 수 있도록)
import builtins
builtins.UserScriptMetadata = UserScriptMetadata
builtins.Artifact = Artifact
builtins.UserScriptResult = UserScriptResult

# 사용자 스크립트 디렉토리를 sys.path에 추가 (마지막에 추가)
sys.path.append(str(Path("{WORK_DIR}")))

try:
    # 사용자 스크립트 import
    print("[DEBUG] sys.path:", sys.path, file=sys.stderr)
    print("[DEBUG] Working directory:", Path.cwd(), file=sys.stderr)
    print("[DEBUG] user_script.py exists:", Path('user_script.py').exists(), file=sys.stderr)

    from user_script import train, USER_SCRIPT_METADATA
    print("[DEBUG] Successfully imported user_script", file=sys.stderr)

    # run_config 로딩
    with open(str(Path("{WORK_DIR}") / "config.json"), 'r', encoding='utf-8') as f:
        run_config_dict = json.load(f)

    # 데이터 로딩
    if 'X_train_path' in run_config_dict:
        run_config_dict['X_train'] = pd.read_parquet(run_config_dict['X_train_path'])
        run_config_dict['y_train'] = pd.read_parquet(run_config_dict['y_train_path']).iloc[:, 0]
        run_config_dict['X_val'] = pd.read_parquet(run_config_dict['X_val_path'])
        run_config_dict['y_val'] = pd.read_parquet(run_config_dict['y_val_path']).iloc[:, 0]
        run_config_dict['X_test'] = pd.read_parquet(run_config_dict['X_test_path'])
        run_config_dict['y_test'] = pd.read_parquet(run_config_dict['y_test_path']).iloc[:, 0]

    # UserScriptRunConfig 객체 생성 (간단한 namespace)
    from types import SimpleNamespace
    run_config = SimpleNamespace(**run_config_dict)

    # 실행
    started_at = datetime.utcnow().isoformat() + 'Z'
    print("[INFO] Training started at", started_at)

    result = train(run_config)

    finished_at = datetime.utcnow().isoformat() + 'Z'
    print("[INFO] Training finished at", finished_at)

    # 모델 저장 (있는 경우)
    model_artifact = None
    if hasattr(result, 'model') and result.model is not None:
        import joblib
        model_path = Path("{WORK_DIR}") / "artifacts" / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.model, model_path)

        model_artifact = {{
            "name": "trained_model",
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "type": "model",
            "created_at": datetime.utcnow().isoformat() + 'Z'
        }}
        print("[INFO] Model saved to", model_path)

    # 결과 출력 (JSON)
    artifacts_list = [
        {{
            "name": a.name,
            "path": a.path,
            "size_bytes": a.size_bytes,
            "type": a.type,
            "created_at": a.created_at.isoformat() + 'Z' if hasattr(a, 'created_at') else datetime.utcnow().isoformat() + 'Z'
        }}
        for a in (result.artifacts if hasattr(result, 'artifacts') else [])
    ]

    # 모델 아티팩트 추가
    if model_artifact:
        artifacts_list.append(model_artifact)

    output = {{
        "result": {{
            "metrics": result.metrics if hasattr(result, 'metrics') else {{}},
            "warnings": result.warnings if hasattr(result, 'warnings') else [],
            "errors": result.errors if hasattr(result, 'errors') else [],
            "artifacts": artifacts_list
        }},
        "started_at": started_at,
        "finished_at": finished_at
    }}

    print("__RESULT_START__")
    print(json.dumps(output, ensure_ascii=False))
    print("__RESULT_END__")
    sys.exit(0)

except Exception as e:
    # 에러 정보를 최대한 상세하게 출력
    print("__ERROR_START__", file=sys.stderr)
    error_data = {{
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc(),
        "sys_path": sys.path,
        "cwd": str(Path.cwd()),
        "work_dir_contents": [str(p) for p in Path("{WORK_DIR}").iterdir()] if Path("{WORK_DIR}").exists() else []
    }}
    print(json.dumps(error_data, ensure_ascii=False, indent=2), file=sys.stderr)
    print("__ERROR_END__", file=sys.stderr)

    # stderr에도 간단한 메시지 출력 (JSON 외부)
    print("\\n[ERROR] Script execution failed:", str(e), file=sys.stderr)
    print("[ERROR] Error type:", type(e).__name__, file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

    sys.exit(1)
'''.format(WORK_DIR=str(work_dir))

    def _run_subprocess(
        self,
        script_path: Path,
        work_dir: Path
    ) -> Dict[str, Any]:
        """서브프로세스에서 스크립트 실행"""
        import platform
        import os
        import sys

        # OpenBLAS 및 기타 수치 라이브러리 스레드 제한 환경 변수 설정
        env = os.environ.copy()
        env['OPENBLAS_NUM_THREADS'] = '4'
        env['MKL_NUM_THREADS'] = '4'
        env['OMP_NUM_THREADS'] = '4'
        env['NUMEXPR_NUM_THREADS'] = '4'

        # 디버깅: 실행 전 정보 출력
        print(f"[DEBUG] Executing script: {script_path}", file=sys.stderr)
        print(f"[DEBUG] Working directory: {work_dir}", file=sys.stderr)
        print(f"[DEBUG] Script exists: {script_path.exists()}", file=sys.stderr)
        print(f"[DEBUG] Work dir exists: {work_dir.exists()}", file=sys.stderr)
        if work_dir.exists():
            print(f"[DEBUG] Work dir contents: {list(work_dir.iterdir())}", file=sys.stderr)

        try:
            result = subprocess.run(
                ["python", str(script_path)],
                cwd=work_dir,
                capture_output=True,
                timeout=self.timeout_seconds,
                text=True,
                encoding='utf-8',
                env=env,
                # Linux에서만 리소스 제한 적용
                preexec_fn=self._setup_resource_limits if platform.system() == 'Linux' else None
            )

            # 디버깅: 실행 결과 출력
            print(f"[DEBUG] Process return code: {result.returncode}", file=sys.stderr)
            print(f"[DEBUG] stdout length: {len(result.stdout)}", file=sys.stderr)
            print(f"[DEBUG] stderr length: {len(result.stderr)}", file=sys.stderr)

            if result.returncode != 0:
                print(f"[DEBUG] Process failed with return code {result.returncode}", file=sys.stderr)
                print(f"[DEBUG] Full stderr output:\n{result.stderr}", file=sys.stderr)
                print(f"[DEBUG] Full stdout output:\n{result.stdout}", file=sys.stderr)

            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.CalledProcessError as e:
            print(f"[DEBUG] CalledProcessError: {e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"[DEBUG] Unexpected error in _run_subprocess: {e}", file=sys.stderr)
            import traceback
            print(traceback.format_exc(), file=sys.stderr)
            raise

    def _setup_resource_limits(self):
        """리소스 제한 설정 (Linux only)"""
        try:
            import resource

            # 메모리 제한
            memory_limit = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

            # CPU 시간 제한
            resource.setrlimit(resource.RLIMIT_CPU, (self.max_cpu_time_seconds, self.max_cpu_time_seconds))

            # 프로세스 수 제한 (OpenBLAS 등을 위해 충분히 높게 설정)
            # 기존 제한을 확인하고, 너무 낮으면 50으로 설정
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
                # hard limit보다 낮은 값으로 설정
                new_limit = min(50, hard) if hard > 0 else 50
                resource.setrlimit(resource.RLIMIT_NPROC, (new_limit, hard))
            except:
                # 제한 설정 실패 시 무시 (시스템이 허용하지 않을 수 있음)
                pass
        except Exception as e:
            import sys
            print(f"Warning: Failed to set resource limits: {e}", file=sys.stderr)

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        start_time: datetime,
        end_time: datetime,
        duration: float
    ) -> Dict[str, Any]:
        """실행 출력 파싱"""
        stdout_lines = stdout.split('\n') if stdout else []
        stderr_lines = stderr.split('\n') if stderr else []

        # 결과 추출
        result_json = None
        in_result = False
        result_lines = []

        for line in stdout_lines:
            if "__RESULT_START__" in line:
                in_result = True
            elif "__RESULT_END__" in line:
                in_result = False
                try:
                    result_json = json.loads('\n'.join(result_lines))
                except json.JSONDecodeError:
                    pass
            elif in_result:
                result_lines.append(line)

        # stdout에서 결과 부분 제거
        filtered_stdout = [
            line for line in stdout_lines
            if "__RESULT_START__" not in line
            and "__RESULT_END__" not in line
            and line not in result_lines
        ]

        # stderr에서 에러 정보 추출
        error_json = None
        in_error = False
        error_lines = []

        for line in stderr_lines:
            if "__ERROR_START__" in line:
                in_error = True
            elif "__ERROR_END__" in line:
                in_error = False
                try:
                    error_json = json.loads('\n'.join(error_lines))
                except json.JSONDecodeError:
                    pass
            elif in_error:
                error_lines.append(line)

        # stderr에서 에러 부분 제거
        filtered_stderr = [
            line for line in stderr_lines
            if "__ERROR_START__" not in line
            and "__ERROR_END__" not in line
            and line not in error_lines
        ]

        # 에러가 있으면 result에 추가
        if error_json and result_json:
            if 'result' not in result_json:
                result_json['result'] = {}
            if 'errors' not in result_json['result']:
                result_json['result']['errors'] = []
            result_json['result']['errors'].append(error_json.get('error', 'Unknown error'))

        return {
            "stdout": [line for line in filtered_stdout if line.strip()],
            "stderr": [line for line in filtered_stderr if line.strip()],
            "result": result_json.get("result") if result_json else None,
            "duration_seconds": duration,
            "started_at": start_time.isoformat() + 'Z',
            "finished_at": end_time.isoformat() + 'Z',
            "exit_code": returncode,
            "resource_usage": None  # 추후 구현 가능
        }


def execute_script(
    script_content: str,
    run_config: Dict[str, Any],
    timeout_seconds: int = 300
) -> Dict[str, Any]:
    """
    스크립트 실행 유틸리티 함수

    Args:
        script_content: 스크립트 내용
        run_config: 실행 설정
        timeout_seconds: 타임아웃 (초)

    Returns:
        실행 결과
    """
    executor = ScriptExecutor(timeout_seconds=timeout_seconds)
    return executor.execute(script_content, run_config)
