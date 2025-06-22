# 파일: app/api/v1/bots.py
# 경로: /workspaces/autoblitz-backend/app/api/v1/bots.py

"""
오토블리츠 봇 관리 API 라우터
자동매매 봇 생성, 제어, 모니터링
"""
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, validator
from typing import Optional, List
from enum import Enum
import logging

from app.api.v1.auth import verify_token
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)

# 라우터 인스턴스
router = APIRouter(prefix="/bots", tags=["봇 관리"])


# Enum 클래스들
class BotStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class BotType(str, Enum):
    SPOT = "spot"
    FUTURES = "futures"
    HYBRID = "hybrid"


class Exchange(str, Enum):
    OKX = "okx"
    UPBIT = "upbit"


# Pydantic 모델들
class BotCreate(BaseModel):
    name: str
    description: Optional[str] = None
    bot_type: BotType = BotType.SPOT
    exchange: Exchange
    trading_pair: str
    strategy_id: int
    allocated_amount: float
    min_order_amount: float = 10.0
    max_position_size: float = 1000.0
    stop_loss_percent: float = 5.0
    take_profit_percent: float = 10.0
    is_paper_trading: bool = True
    
    @validator('allocated_amount')
    def validate_allocated_amount(cls, v):
        if v <= 0:
            raise ValueError('할당 자금은 0보다 커야 합니다')
        return v
    
    @validator('trading_pair')
    def validate_trading_pair(cls, v):
        if '/' not in v:
            raise ValueError('거래쌍 형식이 올바르지 않습니다 (예: BTC/USDT)')
        return v.upper()


class BotUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    allocated_amount: Optional[float] = None
    stop_loss_percent: Optional[float] = None
    take_profit_percent: Optional[float] = None


class BotResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    status: BotStatus
    bot_type: BotType
    exchange: Exchange
    trading_pair: str
    allocated_amount: float
    total_pnl: float
    total_trades: int
    win_rate: float
    is_paper_trading: bool
    created_at: str
    started_at: Optional[str]


# API 엔드포인트들
@router.get("/", summary="내 봇 목록 조회")
async def get_my_bots(
    status: Optional[BotStatus] = None,
    exchange: Optional[Exchange] = None,
    current_user: dict = Depends(verify_token)
):
    """사용자의 봇 목록 조회 (필터링 가능)"""
    
    # 캐시에서 봇 목록 조회
    bots_key = f"user_bots:{current_user['id']}"
    cached_bots = await cache_manager.get(bots_key)
    
    if not cached_bots:
        # 임시 봇 데이터
        bots = [
            {
                "id": 1,
                "name": "BTC 단타 봇",
                "description": "비트코인 단타 전략",
                "status": "running",
                "bot_type": "spot",
                "exchange": "okx",
                "trading_pair": "BTC/USDT",
                "allocated_amount": 1000.0,
                "total_pnl": 125.75,
                "total_trades": 45,
                "win_rate": 68.9,
                "is_paper_trading": True,
                "created_at": "2025-06-20T10:00:00Z",
                "started_at": "2025-06-22T08:00:00Z"
            },
            {
                "id": 2,
                "name": "ETH 스윙 봇",
                "description": "이더리움 스윙 트레이딩",
                "status": "paused",
                "bot_type": "spot",
                "exchange": "upbit",
                "trading_pair": "ETH/KRW",
                "allocated_amount": 500.0,
                "total_pnl": -25.30,
                "total_trades": 12,
                "win_rate": 41.7,
                "is_paper_trading": True,
                "created_at": "2025-06-21T14:00:00Z",
                "started_at": None
            }
        ]
        
        # 캐시에 저장 (5분)
        await cache_manager.set(bots_key, bots, expire=300)
        cached_bots = bots
    
    # 필터링 적용
    filtered_bots = cached_bots
    
    if status:
        filtered_bots = [bot for bot in filtered_bots if bot["status"] == status]
    
    if exchange:
        filtered_bots = [bot for bot in filtered_bots if bot["exchange"] == exchange]
    
    return {
        "bots": filtered_bots,
        "total_count": len(filtered_bots),
        "filters": {
            "status": status,
            "exchange": exchange
        }
    }


@router.post("/", response_model=BotResponse, summary="새 봇 생성")
async def create_bot(
    bot_data: BotCreate,
    current_user: dict = Depends(verify_token)
):
    """새로운 자동매매 봇 생성"""
    
    logger.info(f"봇 생성: {current_user['id']} -> {bot_data.name}")
    
    # 거래소 연동 확인
    exchange_key = f"exchange:{current_user['id']}:{bot_data.exchange}"
    exchange_info = await cache_manager.get(exchange_key)
    
    if not exchange_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{bot_data.exchange.upper()} 거래소가 연동되지 않았습니다"
        )
    
    # 새 봇 ID 생성 (임시)
    new_bot_id = hash(f"{current_user['id']}{bot_data.name}") % 10000
    
    # 봇 데이터 생성
    new_bot = {
        "id": new_bot_id,
        "name": bot_data.name,
        "description": bot_data.description,
        "status": "stopped",
        "bot_type": bot_data.bot_type,
        "exchange": bot_data.exchange,
        "trading_pair": bot_data.trading_pair,
        "allocated_amount": bot_data.allocated_amount,
        "total_pnl": 0.0,
        "total_trades": 0,
        "win_rate": 0.0,
        "is_paper_trading": bot_data.is_paper_trading,
        "created_at": "2025-06-22T08:00:00Z",
        "started_at": None,
        "strategy_id": bot_data.strategy_id,
        "config": {
            "min_order_amount": bot_data.min_order_amount,
            "max_position_size": bot_data.max_position_size,
            "stop_loss_percent": bot_data.stop_loss_percent,
            "take_profit_percent": bot_data.take_profit_percent
        }
    }
    
    # 캐시에 저장
    bot_key = f"bot:{new_bot_id}"
    await cache_manager.set(bot_key, new_bot, expire=3600)
    
    # 사용자 봇 목록 업데이트
    bots_key = f"user_bots:{current_user['id']}"
    cached_bots = await cache_manager.get(bots_key) or []
    cached_bots.append(new_bot)
    await cache_manager.set(bots_key, cached_bots, expire=300)
    
    return new_bot


@router.get("/{bot_id}", response_model=BotResponse, summary="봇 상세 정보")
async def get_bot_detail(
    bot_id: int,
    current_user: dict = Depends(verify_token)
):
    """특정 봇의 상세 정보 조회"""
    
    bot_key = f"bot:{bot_id}"
    bot_data = await cache_manager.get(bot_key)
    
    if not bot_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="봇을 찾을 수 없습니다"
        )
    
    return bot_data


@router.put("/{bot_id}", summary="봇 설정 수정")
async def update_bot(
    bot_id: int,
    bot_update: BotUpdate,
    current_user: dict = Depends(verify_token)
):
    """봇 설정 수정"""
    
    bot_key = f"bot:{bot_id}"
    bot_data = await cache_manager.get(bot_key)
    
    if not bot_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="봇을 찾을 수 없습니다"
        )
    
    # 실행 중인 봇은 수정 제한
    if bot_data["status"] == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="실행 중인 봇은 설정을 수정할 수 없습니다. 먼저 봇을 정지해주세요."
        )
    
    # 업데이트 적용
    update_data = bot_update.dict(exclude_none=True)
    bot_data.update(update_data)
    
    # 캐시 업데이트
    await cache_manager.set(bot_key, bot_data, expire=3600)
    
    return {
        "message": "봇 설정이 성공적으로 수정되었습니다",
        "bot_id": bot_id,
        "updated_fields": list(update_data.keys())
    }


@router.post("/{bot_id}/start", summary="봇 시작")
async def start_bot(
    bot_id: int,
    current_user: dict = Depends(verify_token)
):
    """봇 실행 시작"""
    
    bot_key = f"bot:{bot_id}"
    bot_data = await cache_manager.get(bot_key)
    
    if not bot_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="봇을 찾을 수 없습니다"
        )
    
    if bot_data["status"] == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="봇이 이미 실행 중입니다"
        )
    
    # 봇 상태 변경
    bot_data["status"] = "running"
    bot_data["started_at"] = "2025-06-22T08:00:00Z"
    
    # 캐시 업데이트
    await cache_manager.set(bot_key, bot_data, expire=3600)
    
    logger.info(f"봇 시작: {bot_id} by {current_user['id']}")
    
    return {
        "message": f"봇 '{bot_data['name']}'이 시작되었습니다",
        "bot_id": bot_id,
        "status": "running"
    }


@router.post("/{bot_id}/stop", summary="봇 정지")
async def stop_bot(
    bot_id: int,
    current_user: dict = Depends(verify_token)
):
    """봇 실행 정지"""
    
    bot_key = f"bot:{bot_id}"
    bot_data = await cache_manager.get(bot_key)
    
    if not bot_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="봇을 찾을 수 없습니다"
        )
    
    if bot_data["status"] == "stopped":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="봇이 이미 정지되어 있습니다"
        )
    
    # 봇 상태 변경
    bot_data["status"] = "stopped"
    bot_data["started_at"] = None
    
    # 캐시 업데이트
    await cache_manager.set(bot_key, bot_data, expire=3600)
    
    logger.info(f"봇 정지: {bot_id} by {current_user['id']}")
    
    return {
        "message": f"봇 '{bot_data['name']}'이 정지되었습니다",
        "bot_id": bot_id,
        "status": "stopped"
    }


@router.delete("/{bot_id}", summary="봇 삭제")
async def delete_bot(
    bot_id: int,
    current_user: dict = Depends(verify_token)
):
    """봇 삭제"""
    
    bot_key = f"bot:{bot_id}"
    bot_data = await cache_manager.get(bot_key)
    
    if not bot_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="봇을 찾을 수 없습니다"
        )
    
    if bot_data["status"] == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="실행 중인 봇은 삭제할 수 없습니다. 먼저 봇을 정지해주세요."
        )
    
    # 캐시에서 삭제
    await cache_manager.delete(bot_key)
    
    # 사용자 봇 목록에서도 제거
    bots_key = f"user_bots:{current_user['id']}"
    cached_bots = await cache_manager.get(bots_key) or []
    cached_bots = [bot for bot in cached_bots if bot["id"] != bot_id]
    await cache_manager.set(bots_key, cached_bots, expire=300)
    
    logger.warning(f"봇 삭제: {bot_id} by {current_user['id']}")
    
    return {
        "message": f"봇 '{bot_data['name']}'이 삭제되었습니다",
        "bot_id": bot_id
    }