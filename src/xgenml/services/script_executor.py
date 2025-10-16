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

        except subprocess.TimeoutExpired:
            return {
                "stdout": [],
                "stderr": [f"실행 시간 제한({self.timeout_seconds}초)을 초과했습니다."],
                "result": None,
                "duration_seconds": self.timeout_seconds,
                "started_at": datetime.utcnow().isoformat() + 'Z',
                "finished_at": datetime.utcnow().isoformat() + 'Z',
                "exit_code": -1,
                "resource_usage": None
            }
        except Exception as e:
            return {
                "stdout": [],
                "stderr": [f"실행 중 오류 발생: {str(e)}"],
                "result": None,
                "duration_seconds": 0,
                "started_at": datetime.utcnow().isoformat() + 'Z',
                "finished_at": datetime.utcnow().isoformat() + 'Z',
                "exit_code": -1,
                "resource_usage": None
            }
        finally:
            # 정리
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except:
                pass

    def _create_runner_script(self, work_dir: Path) -> str:
        """실행 래퍼 스크립트 생성"""
        return f'''
import sys
import json
import traceback
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path("{work_dir}")))

try:
    # 사용자 스크립트 import
    from user_script import train, USER_SCRIPT_METADATA

    # run_config 로딩
    with open(str(Path("{work_dir}") / "config.json"), 'r', encoding='utf-8') as f:
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
    print(f"[INFO] Training started at {{started_at}}")

    result = train(run_config)

    finished_at = datetime.utcnow().isoformat() + 'Z'
    print(f"[INFO] Training finished at {{finished_at}}")

    # 모델 저장 (있는 경우)
    model_artifact = None
    if hasattr(result, 'model') and result.model is not None:
        import joblib
        model_path = Path("{work_dir}") / "artifacts" / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.model, model_path)

        model_artifact = {{
            "name": "trained_model",
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "type": "model",
            "created_at": datetime.utcnow().isoformat() + 'Z'
        }}
        print(f"[INFO] Model saved to {{model_path}}")

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
    print("__ERROR_START__", file=sys.stderr)
    error_data = {{
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
    print(json.dumps(error_data, ensure_ascii=False), file=sys.stderr)
    print("__ERROR_END__", file=sys.stderr)
    sys.exit(1)
'''

    def _run_subprocess(
        self,
        script_path: Path,
        work_dir: Path
    ) -> Dict[str, Any]:
        """서브프로세스에서 스크립트 실행"""
        import platform

        result = subprocess.run(
            ["python", str(script_path)],
            cwd=work_dir,
            capture_output=True,
            timeout=self.timeout_seconds,
            text=True,
            encoding='utf-8',
            # Linux에서만 리소스 제한 적용
            preexec_fn=self._setup_resource_limits if platform.system() == 'Linux' else None
        )

        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }

    def _setup_resource_limits(self):
        """리소스 제한 설정 (Linux only)"""
        try:
            import resource

            # 메모리 제한
            memory_limit = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

            # CPU 시간 제한
            resource.setrlimit(resource.RLIMIT_CPU, (self.max_cpu_time_seconds, self.max_cpu_time_seconds))

            # 프로세스 수 제한
            resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
        except Exception as e:
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
