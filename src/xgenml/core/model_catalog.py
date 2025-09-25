# /src/xgenml/core/model_catalog.py
CATALOG = {
    "classification": [
        {"name": "logistic_regression", "cls": "sklearn.linear_model.LogisticRegression",
         "default": {"max_iter": 1000, "n_jobs": -1}},
        {"name": "svm", "cls": "sklearn.svm.SVC",
         "default": {"kernel": "rbf", "C": 1.0, "gamma": "scale"}},
        {"name": "random_forest", "cls": "sklearn.ensemble.RandomForestClassifier",
         "default": {"n_estimators": 300, "max_depth": 12, "n_jobs": -1}},
        {"name": "gradient_boosting", "cls": "sklearn.ensemble.GradientBoostingClassifier",
         "default": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}},
        # XGBoost 추가
        {"name": "xgboost", "cls": "xgboost.XGBClassifier",
         "default": {
             "n_estimators": 300,
             "max_depth": 6,
             "learning_rate": 0.1,
             "subsample": 0.8,
             "colsample_bytree": 0.8,
             "random_state": 42,
             "n_jobs": -1,
             "eval_metric": "logloss"  # 경고 메시지 방지
         },
         "description": "Extreme Gradient Boosting classifier with optimized performance",
         "tags": ["ensemble", "boosting", "high_performance"]},
    ],
    "regression": [
        {"name": "linear_regression", "cls": "sklearn.linear_model.LinearRegression",
         "default": {}},
        {"name": "ridge", "cls": "sklearn.linear_model.Ridge",
         "default": {"alpha": 1.0}},
        {"name": "lasso", "cls": "sklearn.linear_model.Lasso",
         "default": {"alpha": 0.001, "max_iter": 5000}},
        {"name": "random_forest", "cls": "sklearn.ensemble.RandomForestRegressor",
         "default": {"n_estimators": 400, "max_depth": 14, "n_jobs": -1}},
        # XGBoost 추가
        {"name": "xgboost", "cls": "xgboost.XGBRegressor",
         "default": {
             "n_estimators": 300,
             "max_depth": 6,
             "learning_rate": 0.1,
             "subsample": 0.8,
             "colsample_bytree": 0.8,
             "random_state": 42,
             "n_jobs": -1,
             "eval_metric": "rmse"  # 회귀용 평가 지표
         },
         "description": "Extreme Gradient Boosting regressor with optimized performance",
         "tags": ["ensemble", "boosting", "high_performance"]},
    ]
}

# 모델 카탈로그 유틸리티 함수들
def get_all_models():
    """모든 사용 가능한 모델 반환"""
    return CATALOG

def get_models_by_task(task: str):
    """특정 태스크의 모델들만 반환"""
    return CATALOG.get(task, [])

def get_model_info(task: str, model_name: str):
    """특정 모델의 상세 정보 반환"""
    models = get_models_by_task(task)
    for model in models:
        if model["name"] == model_name:
            return model
    return None

def get_models_by_tag(tag: str):
    """특정 태그를 가진 모든 모델 반환"""
    result = {"classification": [], "regression": []}
    
    for task, models in CATALOG.items():
        for model in models:
            if tag in model.get("tags", []):
                result[task].append(model)
    
    return result

def validate_model_name(task: str, model_name: str) -> bool:
    """모델 이름이 유효한지 확인"""
    return any(m["name"] == model_name for m in get_models_by_task(task))