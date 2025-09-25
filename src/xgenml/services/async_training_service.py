# /src/xgenml/services/async_training_service.py
import asyncio
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from .training_service import TrainingService
from .task_manager import task_manager, TaskStatus

logger = logging.getLogger(__name__)

class AsyncTrainingService:
    def __init__(self):
        self.training_service = TrainingService()
        self.executor = None  # 스레드 풀 실행기
    
    def start_async_training(
        self,
        model_id: str,
        task: str,
        hf_repo: str, 
        hf_filename: str, 
        hf_revision: Optional[str],
        target_column: str, 
        feature_columns: Optional[List[str]],
        model_names: List[str], 
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        test_size: float = 0.2, 
        validation_size: float = 0.1,
        use_cv: bool = False, 
        cv_folds: int = 5
    ) -> str:
        """비동기 학습 시작하고 task_id 반환"""
        
        task_id = task_manager.create_task(model_id)
        
        # 백그라운드에서 학습 실행
        def run_training():
            try:
                logger.info(f"Starting background training for task {task_id}")
                
                # 태스크 상태 업데이트
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.RUNNING,
                    started_at=datetime.now(),
                    message="Training started"
                )
                
                # 실제 학습 실행
                result = self.training_service.run(
                    model_id=model_id,
                    task=task,
                    hf_repo=hf_repo,
                    hf_filename=hf_filename,
                    hf_revision=hf_revision,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    model_names=model_names,
                    overrides=overrides,
                    test_size=test_size,
                    validation_size=validation_size,
                    use_cv=use_cv,
                    cv_folds=cv_folds
                )
                
                # 성공 시 태스크 완료
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    completed_at=datetime.now(),
                    progress=100.0,
                    message="Training completed successfully",
                    result=result
                )

                
                logger.info(f"Training task {task_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Training task {task_id} failed: {str(e)}")
                
                # 실패 시 태스크 에러 상태로 업데이트
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    completed_at=datetime.now(),
                    message="Training failed",
                    error=str(e)
                )
        # 백그라운드 스레드에서 실행
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """태스크 상태 조회"""
        return task_manager.get_task_dict(task_id)

# 글로벌 비동기 서비스 인스턴스
async_training_service = AsyncTrainingService()