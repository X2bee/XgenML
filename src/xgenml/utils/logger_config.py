# /src/xgenml/utils/logger_config.py
import logging
import os

def setup_logger(name: str, log_file: str = '/tmp/training.log') -> logging.Logger:
    """통합 로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 핸들러가 이미 있으면 추가하지 않음
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger