"""
Script Validator Service
사용자 스크립트 검증 서비스 (AST 기반)
"""
import ast
from typing import List, Dict, Any, Tuple, Optional
import re


# 금지된 모듈 리스트
BLOCKED_MODULES = [
    'os',
    'subprocess',
    'shutil',
    'socket',
    'sys',
    'importlib',
    '__builtins__',
    'ctypes',
    'multiprocessing',
    'threading',
]

# 금지된 함수 리스트
BLOCKED_FUNCTIONS = [
    'eval',
    'exec',
    'compile',
    'open',  # 제한적 허용 가능하나 현재는 금지
    '__import__',
    'globals',
    'locals',
    'vars',
]

# 허용된 태스크 목록
ALLOWED_TASKS = [
    'classification',
    'regression',
    'forecasting',
    'clustering',
    'anomaly_detection',
    'timeseries'
]


class ScriptValidator:
    """스크립트 검증 클래스"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, content: str) -> Tuple[bool, List[Dict[str, str]], Optional[Dict[str, Any]]]:
        """
        스크립트 전체 검증

        Returns:
            (is_valid, messages, metadata)
        """
        self.errors = []
        self.warnings = []

        # 1. 구문 검증
        tree = self._validate_syntax(content)
        if tree is None:
            return False, self._format_messages(), None

        # 2. 구조 검증
        self._validate_structure(tree)

        # 3. 보안 검증
        self._validate_security(tree)

        # 4. 메타데이터 추출 및 검증
        metadata = None
        if len(self.errors) == 0:
            metadata = self._extract_metadata(tree, content)
            if metadata:
                self._validate_metadata(metadata)
            else:
                self.errors.append("USER_SCRIPT_METADATA를 찾을 수 없거나 파싱할 수 없습니다.")

        is_valid = len(self.errors) == 0
        messages = self._format_messages()

        return is_valid, messages, metadata

    def _validate_syntax(self, content: str) -> Optional[ast.AST]:
        """구문 검증"""
        try:
            tree = ast.parse(content)
            return tree
        except SyntaxError as e:
            self.errors.append(f"구문 오류 (라인 {e.lineno}): {e.msg}")
            return None

    def _validate_structure(self, tree: ast.AST) -> None:
        """구조 검증 (USER_SCRIPT_METADATA, train 함수)"""
        # USER_SCRIPT_METADATA 확인
        has_metadata = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'USER_SCRIPT_METADATA':
                        has_metadata = True
                        break

        if not has_metadata:
            self.errors.append("USER_SCRIPT_METADATA가 정의되지 않았습니다.")

        # train 함수 확인
        train_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'train':
                train_func = node
                break

        if not train_func:
            self.errors.append("train 함수가 정의되지 않았습니다.")
        elif len(train_func.args.args) != 1:
            self.errors.append("train 함수는 정확히 하나의 인자(config)만 받아야 합니다.")

    def _validate_security(self, tree: ast.AST) -> None:
        """보안 검증"""
        for node in ast.walk(tree):
            # 금지된 모듈 import
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in BLOCKED_MODULES:
                        self.errors.append(f"금지된 모듈 import: {alias.name}")

            # 금지된 from import
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in BLOCKED_MODULES:
                        self.errors.append(f"금지된 모듈 import: {node.module}")

            # 금지된 함수 호출
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_FUNCTIONS:
                        self.errors.append(f"금지된 함수 호출: {node.func.id}")

    def _extract_metadata(self, tree: ast.AST, content: str) -> Optional[Dict[str, Any]]:
        """
        메타데이터 추출

        USER_SCRIPT_METADATA = UserScriptMetadata(...) 형태에서 추출
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'USER_SCRIPT_METADATA':
                        # 메타데이터 값 추출
                        return self._parse_metadata_node(node.value)
        return None

    def _parse_metadata_node(self, node: ast.AST) -> Optional[Dict[str, Any]]:
        """
        AST 노드에서 메타데이터 딕셔너리 추출

        UserScriptMetadata(name="...", version="...", ...) 형태 파싱
        """
        metadata = {}

        if isinstance(node, ast.Call):
            # 키워드 인자 파싱
            for keyword in node.keywords:
                key = keyword.arg
                value = self._extract_value(keyword.value)
                if value is not None:
                    metadata[key] = value

        return metadata if metadata else None

    def _extract_value(self, node: ast.AST) -> Any:
        """AST 노드에서 값 추출"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):  # Python 3.7 호환
            return node.s
        elif isinstance(node, ast.Num):  # Python 3.7 호환
            return node.n
        elif isinstance(node, ast.List):
            return [self._extract_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            result = {}
            for k, v in zip(node.keys, node.values):
                key = self._extract_value(k)
                value = self._extract_value(v)
                if key is not None:
                    result[key] = value
            return result
        else:
            return None

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """메타데이터 필드 검증"""
        # name 검증
        name = metadata.get('name', '')
        if not name:
            self.errors.append("메타데이터: name 필드가 필수입니다.")
        elif not re.match(r'^[a-zA-Z0-9_]+$', name):
            self.errors.append("메타데이터: name은 영숫자와 언더스코어만 허용됩니다.")
        elif len(name) > 64:
            self.errors.append("메타데이터: name은 최대 64자입니다.")

        # version 검증
        version = metadata.get('version', '')
        if not version:
            self.errors.append("메타데이터: version 필드가 필수입니다.")
        elif not re.match(r'^\d+\.\d+\.\d+$', version):
            self.errors.append("메타데이터: version은 SemVer 형식(x.y.z)이어야 합니다.")

        # task 검증
        task = metadata.get('task', '')
        if not task:
            self.errors.append("메타데이터: task 필드가 필수입니다.")
        elif task not in ALLOWED_TASKS:
            self.errors.append(f"메타데이터: task는 {ALLOWED_TASKS} 중 하나여야 합니다.")

        # display_name 검증
        display_name = metadata.get('display_name', '')
        if not display_name:
            self.errors.append("메타데이터: display_name 필드가 필수입니다.")
        elif len(display_name) > 256:
            self.errors.append("메타데이터: display_name은 최대 256자입니다.")

        # description 검증 (선택적)
        description = metadata.get('description', '')
        if len(description) > 1000:
            self.warnings.append("메타데이터: description은 최대 1000자를 권장합니다.")

    def _format_messages(self) -> List[Dict[str, str]]:
        """검증 메시지 포맷팅"""
        messages = []

        for error in self.errors:
            messages.append({
                "level": "error",
                "message": error,
                "code": self._get_error_code(error)
            })

        for warning in self.warnings:
            messages.append({
                "level": "warning",
                "message": warning,
                "code": "VALIDATION_WARNING"
            })

        return messages

    def _get_error_code(self, error_msg: str) -> str:
        """에러 메시지에서 에러 코드 추출"""
        if "구문 오류" in error_msg:
            return "VALIDATION_SYNTAX_ERROR"
        elif "USER_SCRIPT_METADATA" in error_msg:
            return "VALIDATION_MISSING_METADATA"
        elif "train 함수" in error_msg:
            return "VALIDATION_MISSING_TRAIN_FUNCTION"
        elif "금지된 모듈" in error_msg:
            return "VALIDATION_SECURITY_VIOLATION"
        elif "금지된 함수" in error_msg:
            return "VALIDATION_SECURITY_VIOLATION"
        elif "version" in error_msg:
            return "VALIDATION_INVALID_VERSION"
        elif "name" in error_msg:
            return "VALIDATION_INVALID_NAME"
        elif "task" in error_msg:
            return "VALIDATION_INVALID_TASK"
        else:
            return "VALIDATION_ERROR"


def validate_script(content: str) -> Tuple[bool, List[Dict[str, str]], Optional[Dict[str, Any]]]:
    """
    스크립트 검증 유틸리티 함수

    Args:
        content: 스크립트 내용

    Returns:
        (is_valid, messages, metadata)
    """
    validator = ScriptValidator()
    return validator.validate(content)
