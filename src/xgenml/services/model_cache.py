# /src/xgenml/services/model_cache.py
import time
from typing import Dict, Tuple, Any, Optional
import threading
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CachedModel:
    model: Any
    feature_names: list
    loaded_at: float
    access_count: int = 0
    last_accessed: float = 0.0

class ModelCache:
    def __init__(self, max_size: int = 10, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CachedModel] = {}
        self._lock = threading.RLock()
    
    def get(self, model_id: str) -> Optional[Tuple[Any, list]]:
        """캐시에서 모델 조회"""
        with self._lock:
            if model_id not in self.cache:
                return None
            
            cached = self.cache[model_id]
            current_time = time.time()
            
            # TTL 체크
            if current_time - cached.loaded_at > self.ttl_seconds:
                logger.info(f"Model {model_id} expired from cache (TTL: {self.ttl_seconds}s)")
                del self.cache[model_id]
                return None
            
            # 액세스 정보 업데이트
            cached.access_count += 1
            cached.last_accessed = current_time
            
            logger.debug(f"Cache hit for model {model_id} (access count: {cached.access_count})")
            return cached.model, cached.feature_names
    
    def put(self, model_id: str, model: Any, feature_names: list):
        """모델을 캐시에 저장"""
        with self._lock:
            current_time = time.time()
            
            # 캐시 크기 제한 체크
            if len(self.cache) >= self.max_size and model_id not in self.cache:
                self._evict_lru()
            
            self.cache[model_id] = CachedModel(
                model=model,
                feature_names=feature_names,
                loaded_at=current_time,
                last_accessed=current_time
            )
            
            logger.info(f"Model {model_id} cached (cache size: {len(self.cache)}/{self.max_size})")
    
    def _evict_lru(self):
        """LRU 정책으로 캐시에서 제거"""
        if not self.cache:
            return
        
        # 가장 오래된 액세스 시간을 가진 모델 찾기
        lru_model_id = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed
        )
        
        del self.cache[lru_model_id]
        logger.info(f"Evicted model {lru_model_id} from cache (LRU)")
    
    def invalidate(self, model_id: str):
        """특정 모델 캐시 무효화"""
        with self._lock:
            if model_id in self.cache:
                del self.cache[model_id]
                logger.info(f"Invalidated cache for model {model_id}")
    
    def clear(self):
        """전체 캐시 클리어"""
        with self._lock:
            self.cache.clear()
            logger.info("Cleared all model cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self._lock:
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "models": {
                    model_id: {
                        "loaded_at": cached.loaded_at,
                        "access_count": cached.access_count,
                        "last_accessed": cached.last_accessed
                    }
                    for model_id, cached in self.cache.items()
                }
            }

# 글로벌 모델 캐시 인스턴스
model_cache = ModelCache(max_size=10, ttl_seconds=3600)  # 1시간 TTL