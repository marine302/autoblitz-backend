# partner_service/main.py
# 작업: 파트너 서버 main.py 파일 생성
# 설명: 사용자 관리 및 프론트엔드 API 서버
# 작업: 파트너 서버 - 사용자 관리 및 프론트엔드 API
# 설명: 사용자 가입/관리, 화이트라벨 솔루션, 데이터 저장

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="AutoBlitz Partner API",
    description="파트너 서버 - 사용자 관리 및 프론트엔드 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "partner_service",
        "version": "1.0.0",
        "description": "사용자 관리 및 프론트엔드 API"
    }

@app.post("/api/v1/users/register")
async def register_user(user_data: dict):
    """사용자 가입 및 서버 할당"""
    return {
        "user_id": "user_12345",
        "server_url": "https://user-12345.autoblitz.com",
        "api_key": "ak_test_key_12345",
        "status": "active"
    }

@app.get("/api/v1/users/{user_id}/server-info")
async def get_user_server_info(user_id: str):
    """사용자 서버 정보 조회"""
    return {
        "user_id": user_id,
        "server_url": f"https://user-{user_id}.autoblitz.com",
        "status": "running",
        "last_active": "2025-06-25T04:30:00Z"
    }

@app.get("/api/v1/configs/strategies")
async def get_available_strategies():
    """본사에서 가져온 전략 목록"""
    # 실제로는 company_service API 호출
    return {
        "strategies": [
            {"id": "dantaro", "name": "7단계 물타기", "tier": "premium"},
            {"id": "scalping", "name": "스캘핑 전략", "tier": "basic"},
            {"id": "momentum", "name": "모멘텀 전략", "tier": "pro"}
        ]
    }

@app.post("/api/v1/users/{user_id}/trading-data")
async def store_trading_data(user_id: str, trading_data: dict):
    """사용자 거래 데이터 저장"""
    return {
        "user_id": user_id,
        "stored": True,
        "timestamp": "2025-06-25T04:30:00Z"
    }

@app.put("/api/v1/users/{user_id}/settings")
async def update_user_settings(user_id: str, settings: dict):
    """사용자 설정 업데이트"""
    return {
        "user_id": user_id,
        "settings_updated": True,
        "timestamp": "2025-06-25T04:30:00Z"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8002,
        reload=True,
        log_level="info"
    )