# /src/xgenml/api/training_router.py
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from typing import Dict
import json
import logging

from src.xgenml.models.schemas import TrainRequest, TrainResponse
from src.xgenml.services.training_service import TrainingService
from src.xgenml.services.async_training_service import async_training_service
from mlflow.tracking import MlflowClient

router = APIRouter(prefix="/training", tags=["Training"])
logger = logging.getLogger("uvicorn.error")

@router.post("/sync", response_model=TrainResponse)
async def train_sync(request: Request):
    """동기 모델 학습 실행 - text/plain으로 온 JSON 데이터 처리"""
    try:
        # Content-Type 확인
        content_type = request.headers.get("content-type", "")
        logger.info(f"Received request with Content-Type: {content_type}")
        
        if content_type == "text/plain":
            body_bytes = await request.body()
            body_str = body_bytes.decode('utf-8')
            logger.info(f"Received body: {body_str}")
            
            try:
                data = json.loads(body_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                raise HTTPException(status_code=400, detail="Invalid JSON data")
                
        elif content_type == "application/json":
            data = await request.json()
        else:
            raise HTTPException(status_code=400, detail="Unsupported Content-Type")
        
        logger.info(f"Parsed data: {data}")
        
        # Pydantic 모델로 검증
        try:
            req = TrainRequest(**data)
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(status_code=422, detail=str(e))
        
        # 학습 서비스 호출
        training_service = TrainingService()
        result = training_service.run(
            model_id=req.model_id,
            task=req.task,
            # HuggingFace
            hf_repo=req.hf_repo,
            hf_filename=req.hf_filename,
            hf_revision=req.hf_revision,
            # MLflow (새로 추가)
            use_mlflow_dataset=req.use_mlflow_dataset,
            mlflow_run_id=req.mlflow_run_id,
            mlflow_experiment_name=req.mlflow_experiment_name,
            mlflow_artifact_path=req.mlflow_artifact_path,
            # 나머지
            target_column=req.target_column,
            feature_columns=req.feature_columns,
            model_names=req.model_names,
            overrides=req.overrides,
            test_size=req.test_size,
            validation_size=req.validation_size,
            use_cv=req.use_cv,
            cv_folds=req.cv_folds,
            hpo_config=req.hpo_config,
        )

        
        logger.info(f"Training service returned: {type(result)}, {result}")
        
        # result의 타입을 확인하고 적절히 처리
        if hasattr(result, 'run_id'):
            response = result
            logger.info(f"Training completed with TrainResponse: run_id={response.run_id}")
        elif isinstance(result, dict):
            logger.info(f"Converting dict result to TrainResponse")
            
            run_id = result.get('run_id', result.get('best_run_id', 'unknown'))
            status = result.get('status', 'completed')
            message = result.get('message', 'Training completed successfully')
            
            results = {
                'best_model': result.get('best_model'),
                'best_accuracy': result.get('best_accuracy'),
                'trained_models_count': result.get('trained_models_count'),
                'duration_seconds': result.get('duration_seconds'),
                'models_results': result.get('models_results', [])
            }
            
            response = TrainResponse(
                run_id=run_id,
                status=status,
                message=message,
                results=results
            )
        else:
            logger.warning(f"Unexpected result type: {type(result)}")
            response = TrainResponse(
                run_id="unknown",
                status="completed",
                message="Training completed but result format is unexpected",
                results={"raw_result": str(result)}
            )
        
        logger.info(f"Final response: run_id={response.run_id}, status={response.status}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training failed with exception: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/async", response_model=Dict[str, str])
async def train_async(request: Request):
    """비동기 모델 학습 시작 - task_id 반환"""
    try:
        content_type = request.headers.get("content-type", "")
        
        if content_type == "text/plain":
            body_bytes = await request.body()
            body_str = body_bytes.decode('utf-8')
            data = json.loads(body_str)
        elif content_type == "application/json":
            data = await request.json()
        else:
            raise HTTPException(status_code=400, detail="Unsupported Content-Type")
        
        req = TrainRequest(**data)
        
        # 비동기 학습 시작 (MLflow 파라미터 추가)
        task_id = async_training_service.start_async_training(
            model_id=req.model_id,
            task=req.task,
            # HuggingFace
            hf_repo=req.hf_repo,
            hf_filename=req.hf_filename,
            hf_revision=req.hf_revision,
            # MLflow (추가)
            use_mlflow_dataset=req.use_mlflow_dataset,
            mlflow_run_id=req.mlflow_run_id,
            mlflow_experiment_name=req.mlflow_experiment_name,
            mlflow_artifact_path=req.mlflow_artifact_path,
            # 나머지
            target_column=req.target_column,
            feature_columns=req.feature_columns,
            model_names=req.model_names,
            overrides=req.overrides,
            test_size=req.test_size,
            validation_size=req.validation_size,
            use_cv=req.use_cv,
            cv_folds=req.cv_folds,
            hpo_config=req.hpo_config,
        )
        
        return {
            "task_id": task_id,
            "message": "Training started in background",
            "status_url": f"/api/training/task/{task_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Failed to start async training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/task/{task_id}/status")
def get_async_task_status(task_id: str):
    """비동기 학습 태스크 상태 조회"""
    try:
        status = async_training_service.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.get("/run/{run_id}/status")
def get_training_status(run_id: str):
    """MLflow run 상태 조회"""
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        
        return {
            "status": run.info.status,  # "RUNNING", "FINISHED", "FAILED" 등
            "progress": None,  # 진행률은 MLflow에서 직접 제공하지 않음
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Run not found: {str(e)}")

@router.get("/experiments")
def get_experiments():
    """MLflow 실험 목록 조회"""
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        
        return [
            {
                "id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "creation_time": exp.creation_time,
            }
            for exp in experiments
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))