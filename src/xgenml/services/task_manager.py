# /src/xgenml/services/task_manager.py
import uuid
import asyncio
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TrainingTask:
    task_id: str
    model_id: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, TrainingTask] = {}
        self._lock = threading.Lock()
        self.cleanup_delay = 300  # 5분 후 정리
    
    def cleanup_completed_tasks(self):
        """완료된 태스크를 일정 시간 후 정리"""
        with self._lock:
            current_time = datetime.now()
            to_remove = []
            
            for task_id, task in self.tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and 
                    task.completed_at and 
                    (current_time - task.completed_at).total_seconds() > self.cleanup_delay):
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]
                logger.info(f"Cleaned up completed task {task_id}")

    
    def create_task(self, model_id: str) -> str:
        task_id = str(uuid.uuid4())
        with self._lock:
            self.tasks[task_id] = TrainingTask(
                task_id=task_id,
                model_id=model_id,
                status=TaskStatus.PENDING,
                created_at=datetime.now()
            )
        logger.info(f"Created training task {task_id} for model {model_id}")
        return task_id
    
    def update_task(self, task_id: str, **kwargs):
        with self._lock:
            if task_id in self.tasks:
                for key, value in kwargs.items():
                    if hasattr(self.tasks[task_id], key):
                        setattr(self.tasks[task_id], key, value)
    
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        with self._lock:
            return self.tasks.get(task_id)
    
    def get_task_dict(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self.get_task(task_id)
        if task:
            result = asdict(task)
            # Enum을 문자열로 변환
            result['status'] = task.status.value
            # datetime을 ISO 문자열로 변환
            result['created_at'] = task.created_at.isoformat()
            if task.started_at:
                result['started_at'] = task.started_at.isoformat()
            if task.completed_at:
                result['completed_at'] = task.completed_at.isoformat()
            return result
        return None

# 글로벌 태스크 매니저 인스턴스
task_manager = TaskManager()