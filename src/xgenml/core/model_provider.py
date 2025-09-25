import importlib
from typing import Dict, Any
from .model_catalog import CATALOG

def create_estimator(task: str, name: str, override: Dict[str, Any] | None = None):
    spec = next((m for m in CATALOG[task] if m["name"] == name), None)
    if spec is None:
        raise ValueError(f"모델 '{name}'이(가) task '{task}' 카탈로그에 없습니다.")
    module, cls_name = spec["cls"].rsplit(".", 1)
    Estimator = getattr(importlib.import_module(module), cls_name)
    params = {**spec.get("default", {}), **(override or {})}
    return Estimator(**params)
