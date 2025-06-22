# 파일: app/api/v1/router.py
# 경로: /workspaces/autoblitz-backend/app/api/v1/router.py

"""
오토블리츠 API v1 메인 라우터
모든 하위 라우터를 통합하는 중앙 라우터
"""
from fastapi import APIRouter

# 하위 라우터들 임포트
from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router
from app.api.v1.bots import router as bots_router

# 메인 API v1 라우터
api_router = APIRouter(prefix="/api/v1")

# 하위 라우터들 등록
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(bots_router)

# API 정보 엔드포인트
@api_router.get("/", summary="API 정보")
async def api_info():
    """API v1 기본 정보"""
    return {
        "message": "🚀 오토블리츠 API v1",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/api/v1/auth",
            "users": "/api/v1/users", 
            "bots": "/api/v1/bots"
        },
        "documentation": "/docs",
        "status": "operational"
    }