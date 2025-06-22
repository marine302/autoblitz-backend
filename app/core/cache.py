# 파일: app/core/cache.py
# 경로: /workspaces/autoblitz-backend/app/core/cache.py

"""
오토블리츠 Redis 캐시 시스템
메모리 최적화 및 성능 최적화 (t4g.nano 대응)
"""
import redis.asyncio as redis
import json
import logging
from typing import Any, Optional, Union
from datetime import timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis 캐시 관리자 - 메모리 최적화"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False
        self.memory_cache = {}  # Redis 실패시 메모리 캐시 폴백
    
    async def connect(self):
        """Redis 연결 (실패시 메모리 캐시로 폴백)"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=settings.max_redis_connections
            )
            
            # 연결 테스트
            await self.redis_client.ping()
            self.is_connected = True
            logger.info("✅ Redis 연결 성공")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis 연결 실패: {e}")
            logger.info("💡 메모리 캐시 모드로 동작합니다")
            self.redis_client = None
            self.is_connected = False
    
    async def disconnect(self):
        """Redis 연결 해제"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("🛑 Redis 연결 종료")
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """캐시 데이터 저장 (Redis 또는 메모리)"""
        
        # JSON 직렬화
        try:
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, ensure_ascii=False)
            elif not isinstance(value, str):
                serialized_value = str(value)
            else:
                serialized_value = value
        except Exception as e:
            logger.error(f"❌ 직렬화 실패: {key} - {e}")
            return False
        
        # Redis 사용 가능시
        if self.is_connected:
            try:
                # 만료시간 처리
                if isinstance(expire, timedelta):
                    expire = int(expire.total_seconds())
                
                await self.redis_client.set(key, serialized_value, ex=expire)
                logger.debug(f"✅ Redis 캐시 저장: {key}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Redis 저장 실패: {key} - {e}")
        
        # 메모리 캐시 폴백
        try:
            self.memory_cache[key] = serialized_value
            logger.debug(f"✅ 메모리 캐시 저장: {key}")
            return True
        except Exception as e:
            logger.error(f"❌ 메모리 캐시 저장 실패: {key} - {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """캐시 데이터 조회 (Redis 우선, 메모리 폴백)"""
        
        # Redis 우선 시도
        if self.is_connected:
            try:
                value = await self.redis_client.get(key)
                if value is not None:
                    return self._deserialize(value)
            except Exception as e:
                logger.error(f"❌ Redis 조회 실패: {key} - {e}")
        
        # 메모리 캐시 폴백
        try:
            value = self.memory_cache.get(key)
            if value is not None:
                return self._deserialize(value)
        except Exception as e:
            logger.error(f"❌ 메모리 캐시 조회 실패: {key} - {e}")
        
        return None
    
    def _deserialize(self, value: str) -> Any:
        """값 역직렬화"""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    async def delete(self, key: str) -> bool:
        """캐시 데이터 삭제"""
        success = False
        
        # Redis 삭제
        if self.is_connected:
            try:
                result = await self.redis_client.delete(key)
                success = bool(result)
            except Exception as e:
                logger.error(f"❌ Redis 삭제 실패: {key} - {e}")
        
        # 메모리 캐시 삭제
        try:
            if key in self.memory_cache:
                del self.memory_cache[key]
                success = True
        except Exception as e:
            logger.error(f"❌ 메모리 캐시 삭제 실패: {key} - {e}")
        
        if success:
            logger.debug(f"🗑️ 캐시 삭제: {key}")
        
        return success
    
    async def exists(self, key: str) -> bool:
        """캐시 키 존재 확인"""
        # Redis 확인
        if self.is_connected:
            try:
                return bool(await self.redis_client.exists(key))
            except Exception as e:
                logger.error(f"❌ Redis 존재 확인 실패: {key} - {e}")
        
        # 메모리 캐시 확인
        return key in self.memory_cache
    
    async def health_check(self) -> dict:
        """캐시 시스템 상태 확인"""
        status = {
            "redis_connected": self.is_connected,
            "memory_cache_keys": len(self.memory_cache),
            "cache_mode": "redis" if self.is_connected else "memory"
        }
        
        if self.is_connected:
            try:
                await self.redis_client.ping()
                status["redis_ping"] = "success"
            except Exception as e:
                status["redis_ping"] = f"failed: {e}"
                self.is_connected = False
        
        return status


# 전역 캐시 매니저 인스턴스
cache_manager = CacheManager()