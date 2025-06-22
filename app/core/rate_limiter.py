# 파일: app/core/rate_limiter.py
# 경로: /workspaces/autoblitz-backend/app/core/rate_limiter.py
"""
AutoBlitz Rate Limiter
요청 제한 및 DDoS 방지
"""

import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict, deque
import logging
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class RateLimiter:
    """요청 제한 관리자"""
    
    def __init__(
        self,
        max_requests: int = None,
        window_seconds: int = None
    ):
        self.max_requests = max_requests or settings.rate_limit_requests
        self.window_seconds = window_seconds or settings.rate_limit_window
        
        # IP별 요청 기록
        self.requests: Dict[str, deque] = defaultdict(deque)
        
        # 차단된 IP 목록
        self.blocked_ips: Dict[str, float] = {}
        
        # 정리 작업을 위한 마지막 시간
        self.last_cleanup = time.time()
        
        logger.info(f"Rate Limiter 초기화: {self.max_requests}req/{self.window_seconds}s")
    
    async def is_allowed(self, client_ip: str) -> bool:
        """요청 허용 여부 확인"""
        current_time = time.time()
        
        # 주기적 정리
        await self._cleanup_old_requests(current_time)
        
        # 차단된 IP 확인
        if self._is_blocked(client_ip, current_time):
            logger.warning(f"차단된 IP 요청: {client_ip}")
            return False
        
        # 현재 IP의 요청 기록
        ip_requests = self.requests[client_ip]
        
        # 윈도우 시간 이전의 요청 제거
        cutoff_time = current_time - self.window_seconds
        while ip_requests and ip_requests[0] < cutoff_time:
            ip_requests.popleft()
        
        # 요청 수 확인
        if len(ip_requests) >= self.max_requests:
            # 제한 초과 - 임시 차단
            self._block_ip(client_ip, current_time)
            logger.warning(f"Rate limit 초과: {client_ip} ({len(ip_requests)} requests)")
            return False
        
        # 요청 기록
        ip_requests.append(current_time)
        return True
    
    def _is_blocked(self, client_ip: str, current_time: float) -> bool:
        """IP 차단 여부 확인"""
        if client_ip not in self.blocked_ips:
            return False
        
        block_time = self.blocked_ips[client_ip]
        block_duration = 300  # 5분 차단
        
        if current_time - block_time > block_duration:
            # 차단 해제
            del self.blocked_ips[client_ip]
            logger.info(f"IP 차단 해제: {client_ip}")
            return False
        
        return True
    
    def _block_ip(self, client_ip: str, current_time: float):
        """IP 차단"""
        self.blocked_ips[client_ip] = current_time
        logger.warning(f"IP 차단: {client_ip}")
    
    async def _cleanup_old_requests(self, current_time: float):
        """오래된 요청 기록 정리"""
        # 10분마다 정리
        if current_time - self.last_cleanup < 600:
            return
        
        self.last_cleanup = current_time
        cutoff_time = current_time - (self.window_seconds * 2)
        
        # 오래된 IP 기록 제거
        ips_to_remove = []
        for ip, requests in self.requests.items():
            # 오래된 요청 제거
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # 빈 기록 제거
            if not requests:
                ips_to_remove.append(ip)
        
        for ip in ips_to_remove:
            del self.requests[ip]
        
        # 오래된 차단 기록 제거
        expired_blocks = [
            ip for ip, block_time in self.blocked_ips.items()
            if current_time - block_time > 3600  # 1시간
        ]
        
        for ip in expired_blocks:
            del self.blocked_ips[ip]
        
        if ips_to_remove or expired_blocks:
            logger.info(f"정리 완료: {len(ips_to_remove)} IP 기록, {len(expired_blocks)} 차단 기록")
    
    def get_stats(self) -> Dict:
        """Rate Limiter 통계"""
        current_time = time.time()
        
        # 활성 IP 수
        active_ips = len([ip for ip, requests in self.requests.items() if requests])
        
        # 차단된 IP 수
        blocked_count = len(self.blocked_ips)
        
        # 최근 1분간 요청 수
        recent_requests = 0
        cutoff_time = current_time - 60
        
        for requests in self.requests.values():
            recent_requests += sum(1 for req_time in requests if req_time > cutoff_time)
        
        return {
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "active_ips": active_ips,
            "blocked_ips": blocked_count,
            "recent_requests_1min": recent_requests,
            "total_tracked_ips": len(self.requests)
        }
    
    def whitelist_ip(self, client_ip: str):
        """IP 화이트리스트 추가 (향후 확장용)"""
        # 현재는 단순 차단 해제
        if client_ip in self.blocked_ips:
            del self.blocked_ips[client_ip]
            logger.info(f"IP 화이트리스트 추가: {client_ip}")
    
    def block_ip_manual(self, client_ip: str):
        """수동 IP 차단"""
        self.blocked_ips[client_ip] = time.time()
        logger.warning(f"IP 수동 차단: {client_ip}")
    
    def reset_ip(self, client_ip: str):
        """IP 기록 초기화"""
        if client_ip in self.requests:
            del self.requests[client_ip]
        if client_ip in self.blocked_ips:
            del self.blocked_ips[client_ip]
        logger.info(f"IP 기록 초기화: {client_ip}")

# 글로벌 Rate Limiter 인스턴스
rate_limiter = RateLimiter()

# 편의 함수
async def check_rate_limit(client_ip: str) -> bool:
    """요청 제한 확인"""
    return await rate_limiter.is_allowed(client_ip)

def get_rate_limit_stats() -> Dict:
    """Rate Limit 통계 조회"""
    return rate_limiter.get_stats()