# /src/xgenml/utils/file_utils.py
import os
import uuid
import time
import tempfile
from typing import List
from src.xgenml.utils.logger_config import setup_logger

logger = setup_logger(__name__)


class TempDirectoryManager:
    """임시 디렉토리 관리 클래스"""
    
    def __init__(self, cleanup_enabled: bool = True):
        self.cleanup_enabled = cleanup_enabled
        self.temp_directories: List[str] = []
    
    def create_safe_directory(self, base_path: str, identifier: str) -> str:
        """안전하게 고유한 디렉토리를 생성"""
        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:8]
        safe_dir = f"{base_path}/{identifier}_{timestamp}_{unique_id}"
        
        try:
            os.makedirs(safe_dir, exist_ok=True, mode=0o755)
            # 권한 확인
            test_file = os.path.join(safe_dir, "test_write")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"✅ 안전한 디렉토리 생성: {safe_dir}")
            self.temp_directories.append(safe_dir)
            return safe_dir
        except Exception as e:
            logger.error(f"❌ 디렉토리 생성 실패: {e}")
            fallback_dir = tempfile.mkdtemp(prefix=f"mlflow_{identifier}_")
            logger.info(f"🔄 폴백 디렉토리 사용: {fallback_dir}")
            self.temp_directories.append(fallback_dir)
            return fallback_dir
    
    def cleanup(self):
        """임시 디렉토리 정리"""
        if not self.cleanup_enabled or not self.temp_directories:
            return
        
        logger.info("🧹 임시 디렉토리 정리 중...")
        import shutil
        for temp_dir in self.temp_directories:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f"✅ 정리 완료: {temp_dir}")
            except Exception as e:
                logger.warning(f"⚠️  디렉토리 정리 실패: {temp_dir} - {e}")