# 파일: app/api/v1/users.py
# 경로: /workspaces/autoblitz-backend/app/api/v1/users.py

"""
오토블리츠 사용자 관리 API 라우터
사용자 프로필, 설정, 거래소 연동 관리
"""
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
import logging

from app.api.v1.auth import verify_token
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)

# 라우터 인스턴스
router = APIRouter(prefix="/users", tags=["사용자 관리"])


# Pydantic 모델들
class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    timezone: Optional[str] = None


class ExchangeCredentials(BaseModel):
    exchange: str  # okx, upbit
    api_key: str
    secret_key: str
    passphrase: Optional[str] = None  # OKX용


class NotificationSettings(BaseModel):
    email_alerts: bool = True
    sms_alerts: bool = False
    discord_webhook: Optional[str] = None
    telegram_chat_id: Optional[str] = None


class UserStats(BaseModel):
    total_bots: int
    active_bots: int
    total_trades: int
    total_pnl: float
    win_rate: float


# API 엔드포인트들
@router.get("/me", summary="내 프로필 조회")
async def get_my_profile(current_user: dict = Depends(verify_token)):
    """현재 사용자의 상세 프로필 정보"""
    
    # 캐시에서 상세 정보 조회
    user_details = await cache_manager.get(f"user_details:{current_user['id']}")
    
    if not user_details:
        # 임시 데이터 생성
        user_details = {
            **current_user,
            "phone": None,
            "timezone": "Asia/Seoul",
            "created_at": "2025-06-22T00:00:00Z",
            "last_login": "2025-06-22T08:00:00Z",
            "exchange_count": 0
        }
        # 캐시에 저장 (10분)
        await cache_manager.set(f"user_details:{current_user['id']}", user_details, expire=600)
    
    return user_details


@router.put("/me", summary="내 프로필 수정")
async def update_my_profile(
    user_update: UserUpdate,
    current_user: dict = Depends(verify_token)
):
    """사용자 프로필 정보 수정"""
    
    logger.info(f"프로필 수정: {current_user['id']}")
    
    # 기존 정보 조회
    user_details = await cache_manager.get(f"user_details:{current_user['id']}") or current_user
    
    # 업데이트
    if user_update.full_name is not None:
        user_details["full_name"] = user_update.full_name
    if user_update.phone is not None:
        user_details["phone"] = user_update.phone
    if user_update.timezone is not None:
        user_details["timezone"] = user_update.timezone
    
    # 캐시 업데이트
    await cache_manager.set(f"user_details:{current_user['id']}", user_details, expire=600)
    
    return {
        "message": "프로필이 성공적으로 수정되었습니다",
        "updated_data": user_update.dict(exclude_none=True)
    }


@router.get("/stats", summary="내 거래 통계")
async def get_my_stats(current_user: dict = Depends(verify_token)):
    """사용자의 거래 통계 정보"""
    
    # 캐시에서 통계 조회
    stats_key = f"user_stats:{current_user['id']}"
    cached_stats = await cache_manager.get(stats_key)
    
    if not cached_stats:
        # 임시 통계 데이터
        stats = {
            "total_bots": 2,
            "active_bots": 1,
            "total_trades": 150,
            "total_pnl": 1250.75,
            "win_rate": 68.5,
            "best_strategy": "단타로 전략",
            "monthly_performance": {
                "profit": 850.25,
                "trades": 45,
                "win_rate": 72.3
            }
        }
        # 캐시에 저장 (5분)
        await cache_manager.set(stats_key, stats, expire=300)
        cached_stats = stats
    
    return cached_stats


@router.post("/exchanges", summary="거래소 연동")
async def add_exchange_credentials(
    credentials: ExchangeCredentials,
    current_user: dict = Depends(verify_token)
):
    """거래소 API 키 등록"""
    
    logger.info(f"거래소 연동: {current_user['id']} -> {credentials.exchange}")
    
    # 지원 거래소 확인
    supported_exchanges = ["okx", "upbit"]
    if credentials.exchange not in supported_exchanges:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 거래소입니다. 지원 거래소: {supported_exchanges}"
        )
    
    # OKX는 passphrase 필수
    if credentials.exchange == "okx" and not credentials.passphrase:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OKX 거래소는 passphrase가 필요합니다"
        )
    
    # 임시로 캐시에 저장 (실제로는 암호화해서 DB 저장)
    exchange_key = f"exchange:{current_user['id']}:{credentials.exchange}"
    
    # 민감 정보는 마스킹해서 저장
    masked_credentials = {
        "exchange": credentials.exchange,
        "api_key": credentials.api_key[:8] + "***",
        "secret_key": "***" + credentials.secret_key[-4:],
        "passphrase": "***" if credentials.passphrase else None,
        "status": "connected",
        "connected_at": "2025-06-22T08:00:00Z"
    }
    
    await cache_manager.set(exchange_key, masked_credentials, expire=3600)
    
    return {
        "message": f"{credentials.exchange.upper()} 거래소 연동이 완료되었습니다",
        "exchange": credentials.exchange,
        "status": "connected"
    }


@router.get("/exchanges", summary="연동된 거래소 목록")
async def get_connected_exchanges(current_user: dict = Depends(verify_token)):
    """사용자가 연동한 거래소 목록"""
    
    exchanges = []
    
    # 캐시에서 연동 정보 조회
    for exchange in ["okx", "upbit"]:
        exchange_key = f"exchange:{current_user['id']}:{exchange}"
        exchange_data = await cache_manager.get(exchange_key)
        
        if exchange_data:
            exchanges.append(exchange_data)
    
    return {
        "connected_exchanges": exchanges,
        "total_count": len(exchanges)
    }


@router.put("/notifications", summary="알림 설정 변경")
async def update_notification_settings(
    settings: NotificationSettings,
    current_user: dict = Depends(verify_token)
):
    """사용자 알림 설정 수정"""
    
    logger.info(f"알림 설정 변경: {current_user['id']}")
    
    # 캐시에 저장
    settings_key = f"notifications:{current_user['id']}"
    await cache_manager.set(settings_key, settings.dict(), expire=3600)
    
    return {
        "message": "알림 설정이 성공적으로 변경되었습니다",
        "settings": settings.dict()
    }


@router.get("/notifications", summary="현재 알림 설정")
async def get_notification_settings(current_user: dict = Depends(verify_token)):
    """사용자의 현재 알림 설정 조회"""
    
    settings_key = f"notifications:{current_user['id']}"
    settings = await cache_manager.get(settings_key)
    
    if not settings:
        # 기본 설정
        settings = {
            "email_alerts": True,
            "sms_alerts": False,
            "discord_webhook": None,
            "telegram_chat_id": None
        }
        await cache_manager.set(settings_key, settings, expire=3600)
    
    return settings


@router.delete("/account", summary="계정 삭제")
async def delete_account(current_user: dict = Depends(verify_token)):
    """사용자 계정 삭제 (7일 후 완전 삭제)"""
    
    logger.warning(f"계정 삭제 요청: {current_user['id']}")
    
    # 실제로는 DB에서 삭제 예약 상태로 변경
    # 임시로 캐시에서만 제거
    user_id = current_user['id']
    
    # 관련 캐시 삭제
    keys_to_delete = [
        f"user_details:{user_id}",
        f"user_stats:{user_id}",
        f"notifications:{user_id}"
    ]
    
    for key in keys_to_delete:
        await cache_manager.delete(key)
    
    return {
        "message": "계정 삭제가 예약되었습니다",
        "deletion_date": "2025-06-29",
        "note": "7일 이내에 로그인하면 삭제가 취소됩니다"
    }