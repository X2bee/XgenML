# /src/xgenml/exceptions.py
class XgenMLException(Exception):
    """Base exception for XgenML"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

class DataLoadError(XgenMLException):
    """데이터 로딩 관련 에러"""
    pass

class ModelNotFoundError(XgenMLException):
    """모델을 찾을 수 없는 경우"""
    pass

class ValidationError(XgenMLException):
    """입력 데이터 검증 실패"""
    pass

class TrainingError(XgenMLException):
    """모델 학습 실패"""
    pass

class PredictionError(XgenMLException):
    """예측 실패"""
    pass

class ConfigurationError(XgenMLException):
    """설정 오류"""
    pass
