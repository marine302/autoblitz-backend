# 파일: app/main.py (모니터링 시스템 통합 버전)
# 경로: /workspaces/autoblitz-backend/app/main.py
"""
AutoBlitz 백엔드 메인 애플리케이션
모니터링 시스템 통합
"""

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import time
import logging

from .core.config import get_settings
from .core.database import init_db
from .core.rate_limiter import RateLimiter
from .api.v1.router import api_router
from .monitoring import (
    init_monitoring, 
    shutdown_monitoring, 
    get_monitoring_status,
    measure_api_call,
    record_api_performance
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시 실행
    logger.info("AutoBlitz 백엔드 서버 시작")
    
    try:
        # 데이터베이스 초기화
        await init_db()
        logger.info("데이터베이스 초기화 완료")
        
        # 모니터링 시스템 초기화
        await init_monitoring()
        logger.info("모니터링 시스템 초기화 완료")
        
        logger.info("🚀 AutoBlitz 백엔드 서버 준비 완료!")
        
        yield
        
    except Exception as e:
        logger.error(f"서버 시작 실패: {e}")
        raise
    finally:
        # 종료 시 실행
        logger.info("AutoBlitz 백엔드 서버 종료 시작")
        
        try:
            # 모니터링 시스템 종료
            await shutdown_monitoring()
            logger.info("모니터링 시스템 종료 완료")
            
        except Exception as e:
            logger.error(f"서버 종료 중 오류: {e}")
        
        logger.info("AutoBlitz 백엔드 서버 종료 완료")

# FastAPI 앱 생성
app = FastAPI(
    title="AutoBlitz Backend API",
    description="암호화폐 자동매매 SaaS 백엔드 서버 (모니터링 통합)",
    version="1.1.0",
    lifespan=lifespan
)

# Rate Limiter 초기화
rate_limiter = RateLimiter()

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용, 운영에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 신뢰할 수 있는 호스트 미들웨어
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # 개발용, 운영에서는 제한 필요
)

# 모니터링 미들웨어
@app.middleware("http")
# Rate Limiting 미들웨어
@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """요청 제한 미들웨어"""
    client_ip = request.client.host
    
    if not await rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    
    response = await call_next(request)
    return response

# 기본 라우트
@app.get("/")
async def read_root():
    """기본 엔드포인트"""
    return {
        "message": "Hello AutoBlitz Backend!",
        "version": "1.1.0",
        "features": [
            "JWT Authentication",
            "Database Integration",
            "Cache System",
            "Rate Limiting",
            "CloudWatch Monitoring",
            "Real-time Alerts"
        ]
    }

@app.get("/health")
async def health_check():
    """상세 헬스체크 (모니터링 포함)"""
    try:
        monitoring_status = await get_monitoring_status()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "database": "connected",
            "cache": "available",
            "monitoring": monitoring_status
        }
    except Exception as e:
        logger.error(f"헬스체크 실패: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

@app.get("/monitoring/status")
async def monitoring_status():
    """모니터링 시스템 상태 조회"""
    return await get_monitoring_status()

@app.get("/monitoring/dashboard")
async def monitoring_dashboard():
    """모니터링 대시보드 데이터"""
    from .monitoring import monitoring_system
    return monitoring_system.get_dashboard_data()

# API 라우터 등록 (기존 v1 구조 활용)
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
# 전략 API 라우터 추가
from app.api.v1.strategies import router as strategies_router
app.include_router(strategies_router, prefix="/api/v1", tags=["strategies"])
