# /src/xgenml/core/model_catalog.py
from src.xgenml.services.script_registry import get_script_registry

CATALOG = {
    # ========================================
    # 기존: Classification
    # ========================================
    "classification": [
        {"name": "logistic_regression", "cls": "sklearn.linear_model.LogisticRegression",
         "default": {"max_iter": 1000, "n_jobs": -1}},
        {"name": "svm", "cls": "sklearn.svm.SVC",
         "default": {"kernel": "rbf", "C": 1.0, "gamma": "scale"}},
        {"name": "random_forest", "cls": "sklearn.ensemble.RandomForestClassifier",
         "default": {"n_estimators": 300, "max_depth": 12, "n_jobs": -1}},
        {"name": "gradient_boosting", "cls": "sklearn.ensemble.GradientBoostingClassifier",
         "default": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}},
        {"name": "xgboost", "cls": "xgboost.XGBClassifier",
         "default": {
             "n_estimators": 300,
             "max_depth": 6,
             "learning_rate": 0.1,
             "subsample": 0.8,
             "colsample_bytree": 0.8,
             "random_state": 42,
             "n_jobs": -1,
             "eval_metric": "logloss"
         },
         "description": "Extreme Gradient Boosting classifier with optimized performance",
         "tags": ["ensemble", "boosting", "high_performance"]},
        {"name": "lightgbm", "cls": "lightgbm.LGBMClassifier",
         "default": {
             "n_estimators": 300,
             "learning_rate": 0.1,
             "max_depth": 6,
             "random_state": 42,
             "n_jobs": -1,
             "verbose": -1
         },
         "description": "Light Gradient Boosting Machine",
         "tags": ["ensemble", "boosting", "high_performance", "memory_efficient"]},
    ],
    
    # ========================================
    # 기존: Regression
    # ========================================
    "regression": [
        {"name": "linear_regression", "cls": "sklearn.linear_model.LinearRegression",
         "default": {}},
        {"name": "ridge", "cls": "sklearn.linear_model.Ridge",
         "default": {"alpha": 1.0}},
        {"name": "lasso", "cls": "sklearn.linear_model.Lasso",
         "default": {"alpha": 0.001, "max_iter": 5000}},
        {"name": "random_forest", "cls": "sklearn.ensemble.RandomForestRegressor",
         "default": {"n_estimators": 400, "max_depth": 14, "n_jobs": -1}},
        {"name": "xgboost", "cls": "xgboost.XGBRegressor",
         "default": {
             "n_estimators": 300,
             "max_depth": 6,
             "learning_rate": 0.1,
             "subsample": 0.8,
             "colsample_bytree": 0.8,
             "random_state": 42,
             "n_jobs": -1,
             "eval_metric": "rmse"
         },
         "description": "Extreme Gradient Boosting regressor with optimized performance",
         "tags": ["ensemble", "boosting", "high_performance"]},
        {"name": "lightgbm", "cls": "lightgbm.LGBMRegressor",
         "default": {
             "n_estimators": 300,
             "learning_rate": 0.1,
             "max_depth": 6,
             "random_state": 42,
             "n_jobs": -1,
             "verbose": -1
         },
         "description": "Light Gradient Boosting Machine",
         "tags": ["ensemble", "boosting", "high_performance", "memory_efficient"]},
    ],
    
    # ========================================
    # 🆕 Time Series
    # ========================================
    "timeseries": [
        {"name": "random_forest", "cls": "sklearn.ensemble.RandomForestRegressor",
         "default": {"n_estimators": 300, "max_depth": 10, "n_jobs": -1},
         "description": "Random Forest for time series regression",
         "tags": ["ensemble", "tree_based"]},
        
        {"name": "xgboost", "cls": "xgboost.XGBRegressor",
         "default": {
             "n_estimators": 200,
             "max_depth": 5,
             "learning_rate": 0.05,
             "subsample": 0.8,
             "colsample_bytree": 0.8,
             "random_state": 42,
             "n_jobs": -1,
         },
         "description": "XGBoost for time series forecasting",
         "tags": ["ensemble", "boosting", "high_performance"]},
        
        {"name": "lightgbm", "cls": "lightgbm.LGBMRegressor",
         "default": {
             "n_estimators": 200,
             "learning_rate": 0.05,
             "max_depth": 5,
             "random_state": 42,
             "n_jobs": -1,
             "verbose": -1
         },
         "description": "LightGBM for time series forecasting",
         "tags": ["ensemble", "boosting", "high_performance", "fast"]},
        
        {"name": "gradient_boosting", "cls": "sklearn.ensemble.GradientBoostingRegressor",
         "default": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
         "description": "Gradient Boosting for time series",
         "tags": ["ensemble", "boosting"]},
        
        {"name": "linear_regression", "cls": "sklearn.linear_model.LinearRegression",
         "default": {},
         "description": "Simple linear regression baseline",
         "tags": ["linear", "baseline"]},
        
        # 시계열 전용 모델 (추가 라이브러리 필요)
        {"name": "arima", "cls": "src.xgenml.models.timeseries_models.ARIMAWrapper",
         "default": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
         "description": "ARIMA model for univariate time series",
         "tags": ["statistical", "univariate", "classical"],
         "requires": ["statsmodels"]},
        
        {"name": "prophet", "cls": "src.xgenml.models.timeseries_models.ProphetWrapper",
         "default": {"seasonality_mode": "multiplicative"},
         "description": "Facebook Prophet for trend and seasonality",
         "tags": ["statistical", "trend", "seasonality"],
         "requires": ["prophet"]},
    ],
    
    # ========================================
    # 🆕 Anomaly Detection
    # ========================================
    "anomaly_detection": [
        # Unsupervised methods
        {"name": "isolation_forest", "cls": "sklearn.ensemble.IsolationForest",
         "default": {
             "contamination": 0.1,
             "n_estimators": 200,
             "max_samples": 256,
             "random_state": 42,
             "n_jobs": -1
         },
         "description": "Isolation Forest for anomaly detection",
         "tags": ["unsupervised", "ensemble", "fast"]},
        
        {"name": "one_class_svm", "cls": "sklearn.svm.OneClassSVM",
         "default": {"nu": 0.1, "kernel": "rbf", "gamma": "auto"},
         "description": "One-Class SVM for novelty detection",
         "tags": ["unsupervised", "svm"]},
        
        {"name": "local_outlier_factor", "cls": "sklearn.neighbors.LocalOutlierFactor",
         "default": {
             "contamination": 0.1,
             "n_neighbors": 20,
             "novelty": True,
             "n_jobs": -1
         },
         "description": "Local Outlier Factor for anomaly detection",
         "tags": ["unsupervised", "density_based"]},
        
        {"name": "elliptic_envelope", "cls": "sklearn.covariance.EllipticEnvelope",
         "default": {"contamination": 0.1, "support_fraction": None},
         "description": "Gaussian distribution based anomaly detection",
         "tags": ["unsupervised", "gaussian"]},
        
        # Supervised methods (라벨이 있는 경우)
        {"name": "random_forest_supervised", "cls": "sklearn.ensemble.RandomForestClassifier",
         "default": {"n_estimators": 200, "max_depth": 10, "n_jobs": -1, "class_weight": "balanced"},
         "description": "Random Forest for supervised anomaly detection",
         "tags": ["supervised", "ensemble"]},
        
        {"name": "xgboost_supervised", "cls": "xgboost.XGBClassifier",
         "default": {
             "n_estimators": 200,
             "max_depth": 5,
             "learning_rate": 0.1,
             "scale_pos_weight": 10,  # 불균형 데이터 처리
             "random_state": 42,
             "n_jobs": -1,
             "eval_metric": "aucpr"  # precision-recall AUC
         },
         "description": "XGBoost for supervised anomaly detection",
         "tags": ["supervised", "boosting", "imbalanced"]},
    ],
    
    # ========================================
    # 🆕 Clustering
    # ========================================
    "clustering": [
        {"name": "kmeans", "cls": "sklearn.cluster.KMeans",
         "default": {"n_clusters": 3, "n_init": 10, "max_iter": 300, "random_state": 42},
         "description": "K-Means clustering",
         "tags": ["centroid_based", "fast"]},
        
        {"name": "dbscan", "cls": "sklearn.cluster.DBSCAN",
         "default": {"eps": 0.5, "min_samples": 5, "n_jobs": -1},
         "description": "Density-based clustering",
         "tags": ["density_based", "noise_robust"]},
        
        {"name": "hierarchical", "cls": "sklearn.cluster.AgglomerativeClustering",
         "default": {"n_clusters": 3, "linkage": "ward"},
         "description": "Hierarchical clustering",
         "tags": ["hierarchical", "deterministic"]},
        
        {"name": "gaussian_mixture", "cls": "sklearn.mixture.GaussianMixture",
         "default": {"n_components": 3, "covariance_type": "full", "random_state": 42},
         "description": "Gaussian Mixture Model",
         "tags": ["probabilistic", "soft_clustering"]},
        
        {"name": "meanshift", "cls": "sklearn.cluster.MeanShift",
         "default": {"bandwidth": None, "n_jobs": -1},
         "description": "Mean Shift clustering",
         "tags": ["density_based", "auto_k"]},
    ],
}


# Primary metrics for each task
PRIMARY_METRICS = {
    "classification": "accuracy",
    "regression": "r2",
    "timeseries": "rmse",
    "anomaly_detection": "roc_auc",
    "clustering": "silhouette_score",
}


# ========================================
# 유틸리티 함수들
# ========================================

def get_all_models():
    """모든 사용 가능한 모델 반환"""
    return CATALOG


def get_models_by_task(task: str):
    """특정 태스크의 모델들만 반환"""
    return CATALOG.get(task, [])


def get_model_info(task: str, model_name: str):
    """특정 모델의 상세 정보 반환"""
    # Check built-in models
    models = get_models_by_task(task)
    for model in models:
        if model["name"] == model_name:
            return model

    # Check user script models
    if "@" in model_name:
        parts = model_name.split("@")
        if len(parts) == 2:
            name, version = parts
            registry = get_script_registry()
            script_info = registry.get_script(name, version)
            if script_info and script_info.get("task") == task:
                return {
                    "name": model_name,
                    "is_user_script": True,
                    "task": script_info.get("task"),
                    "requires": script_info.get("metadata", {}).get("requires", []),
                    "script_path": script_info.get("absolute_path"),
                    "content": registry.get_script_content(name, version)
                }

    return None



def get_models_by_tag(tag: str):
    """특정 태그를 가진 모든 모델 반환"""
    result = {task: [] for task in CATALOG.keys()}
    
    for task, models in CATALOG.items():
        for model in models:
            if tag in model.get("tags", []):
                result[task].append(model)
    
    return result


def validate_model_name(task: str, model_name: str) -> bool:
    """모델 이름이 유효한지 확인"""
    from src.xgenml.utils.logger_config import setup_logger
    logger = setup_logger(__name__)

    # Check built-in models first
    if any(m["name"] == model_name for m in get_models_by_task(task)):
        logger.info(f"✅ '{model_name}'은 빌트인 모델입니다 (task={task})")
        return True

    # Check for user script models (e.g., "my_model@1.0.0")
    if "@" in model_name:
        parts = model_name.split("@")
        if len(parts) == 2:
            name, version = parts
            registry = get_script_registry()
            exists = registry.script_exists(name, version)

            if exists:
                # 추가로 task가 일치하는지 확인
                script_info = registry.get_script(name, version)
                if script_info:
                    script_task = script_info.get("task")
                    if script_task == task:
                        logger.info(f"✅ '{model_name}'은 등록된 UserScript입니다 (task={task})")
                        return True
                    else:
                        logger.warning(f"❌ '{model_name}'의 태스크가 일치하지 않습니다. 스크립트 task={script_task}, 요청된 task={task}")
                        return False
                else:
                    logger.warning(f"❌ '{model_name}' 스크립트 정보를 가져올 수 없습니다")
                    return False
            else:
                logger.warning(f"❌ '{model_name}' 스크립트가 레지스트리에 등록되지 않았습니다")
                registered_scripts = [f"{s['name']}@{s['version']}" for s in registry.list_scripts()]
                logger.info(f"등록된 스크립트 목록: {registered_scripts}")
                return False

    logger.warning(f"❌ '{model_name}'은 인식할 수 없는 모델 이름입니다 (task={task})")
    return False


def get_available_tasks():
    """사용 가능한 태스크 목록"""
    return list(CATALOG.keys())


def get_primary_metric(task: str) -> str:
    """태스크의 주요 평가 지표"""
    return PRIMARY_METRICS.get(task, "accuracy")


def get_models_requiring_package(package: str):
    """특정 패키지가 필요한 모델 목록"""
    result = {}
    for task, models in CATALOG.items():
        matching_models = [
            model for model in models 
            if package in model.get("requires", [])
        ]
        if matching_models:
            result[task] = matching_models
    return result


def check_model_requirements(task: str, model_name: str) -> dict:
    """모델이 필요로 하는 패키지 확인"""
    model_info = get_model_info(task, model_name)
    if not model_info:
        return {"available": False, "error": f"Model {model_name} not found"}
    
    requirements = model_info.get("requires", [])
    missing = []
    
    for package in requirements:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return {
        "available": len(missing) == 0,
        "required_packages": requirements,
        "missing_packages": missing
    }