from typing import Dict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_squared_error, r2_score
)
import numpy as np

def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
    }

def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """회귀 평가 지표 계산"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # squared 파라미터 대신 직접 계산
    r2 = r2_score(y_true, y_pred)
    
    return {
        "mse": float(mse), 
        "rmse": float(rmse), 
        "r2": float(r2)
    }


def timeseries_metrics(y_true, y_pred) -> Dict[str, float]:
    """시계열 예측 메트릭"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape)
    }


def anomaly_detection_metrics(y_true, y_pred) -> Dict[str, float]:
    """이상 탐지 메트릭"""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except:
        roc_auc = 0.0
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }