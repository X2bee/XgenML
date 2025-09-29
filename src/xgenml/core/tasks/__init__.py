# /src/xgenml/core/tasks/__init__.py
from typing import Dict, Type
from src.xgenml.core.tasks.base import BaseTask

class TaskRegistry:
    """태스크 레지스트리 - 새로운 태스크 타입을 쉽게 추가"""
    _tasks: Dict[str, Type[BaseTask]] = {}
    
    @classmethod
    def register(cls, task_name: str):
        """데코레이터로 태스크 등록"""
        def wrapper(task_class):
            cls._tasks[task_name] = task_class
            return task_class
        return wrapper
    
    @classmethod
    def get_task(cls, task_name: str) -> Type[BaseTask]:
        """등록된 태스크 가져오기"""
        if task_name not in cls._tasks:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(cls._tasks.keys())}")
        return cls._tasks[task_name]
    
    @classmethod
    def list_tasks(cls):
        """사용 가능한 모든 태스크 목록"""
        return list(cls._tasks.keys())