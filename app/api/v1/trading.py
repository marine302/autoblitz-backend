# app/api/v1/trading.py
"""
거래 시스템 API 엔드포인트 - 실제 거래 실행을 위한 REST API
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import asyncio

from ...strategies.core.trading_system import (
    TradingSystem, TradingConfig, TradingMode, 
    create_trading_system
)
from ...strategies.utils.okx_private_client import OKXConfig
from ...strategies.plugins.dantaro_strategy import DantaroConfig
from ...core.database import get_db
# from ...core.auth import get_current_user  # 임시 비활성화

logger = logging.getLogger(__name__)

# =================================
# Pydantic 모델들
# =================================

class TradingConfigRequest(BaseModel):
    """거래 설정 요청"""
    mode: str = Field(..., description="거래 모드: simulation 또는 live")
    max_positions: int = Field(5, description="최대 동시 포지션 수")
    max_daily_loss: float = Field(1000.0, description="일일 최대 손실 (USDT)")
    position_size_limit: float = Field(100.0, description="포지션당 최대 크기 (USDT)")
    stop_loss_percentage: float = Field(5.0, description="스탑로스 비율 (%)")
    take_profit_percentage: float = Field(2.0, description="익절 비율 (%)")

class OKXCredentials(BaseModel):
    """OKX API 인증 정보"""
    api_key: str = Field(..., description="OKX API Key")
    secret_key: str = Field(..., description="OKX Secret Key")
    passphrase: str = Field(..., description="OKX Passphrase")
    sandbox: bool = Field(True, description="샌드박스 모드 여부")

class TradingSignalRequest(BaseModel):
    """거래 신호 요청"""
    symbol: str = Field(..., description="거래 심볼 (예: BTC-USDT-SWAP)")
    signal: str = Field(..., description="신호 타입: BUY, ADD, SELL")
    current_price: float = Field(..., description="현재 가격")
    confidence: float = Field(0.0, description="신호 신뢰도 (0-1)")
    metadata: Optional[Dict] = Field({}, description="추가 메타데이터")

class EmergencyStopRequest(BaseModel):
    """비상 정지 요청"""
    reason: str = Field(..., description="정지 사유")
    close_all_positions: bool = Field(True, description="모든 포지션 청산 여부")

# =================================
# 전역 거래 시스템 관리
# =================================

class TradingSystemManager:
    """거래 시스템 매니저 - 사용자별 거래 시스템 관리"""
    
    def __init__(self):
        self.systems: Dict[str, TradingSystem] = {}
        self.system_configs: Dict[str, Dict] = {}
    
    async def get_or_create_system(self, user_id: str, okx_credentials: OKXCredentials, 
                                 trading_config: TradingConfigRequest) -> TradingSystem:
        """사용자별 거래 시스템 조회 또는 생성"""
        try:
            # 기존 시스템이 있고 설정이 동일한 경우 재사용
            if user_id in self.systems:
                existing_config = self.system_configs.get(user_id, {})
                if (existing_config.get('api_key') == okx_credentials.api_key and
                    existing_config.get('mode') == trading_config.mode):
                    return self.systems[user_id]
                else:
                    # 설정이 변경된 경우 기존 시스템 정리
                    await self._cleanup_system(user_id)
            
            # 새 거래 시스템 생성
            system = create_trading_system(
                api_key=okx_credentials.api_key,
                secret_key=okx_credentials.secret_key,
                passphrase=okx_credentials.passphrase,
                sandbox=okx_credentials.sandbox,
                max_positions=trading_config.max_positions,
                max_daily_loss=trading_config.max_daily_loss
            )
            
            # 시스템 모드 설정
            system.trading_config.mode = TradingMode.SIMULATION if trading_config.mode == "simulation" else TradingMode.LIVE
            system.trading_config.position_size_limit = trading_config.position_size_limit
            system.trading_config.stop_loss_percentage = trading_config.stop_loss_percentage
            system.trading_config.take_profit_percentage = trading_config.take_profit_percentage
            
            # 시스템 초기화
            await system.__aenter__()
            
            # 저장
            self.systems[user_id] = system
            self.system_configs[user_id] = {
                'api_key': okx_credentials.api_key,
                'mode': trading_config.mode,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"거래 시스템 생성 완료: {user_id} ({trading_config.mode} 모드)")
            return system
            
        except Exception as e:
            logger.error(f"거래 시스템 생성 실패: {e}")
            raise HTTPException(status_code=500, detail=f"거래 시스템 생성 실패: {str(e)}")
    
    async def _cleanup_system(self, user_id: str):
        """거래 시스템 정리"""
        if user_id in self.systems:
            try:
                await self.systems[user_id].__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"시스템 정리 중 에러: {e}")
            finally:
                del self.systems[user_id]
                if user_id in self.system_configs:
                    del self.system_configs[user_id]
    
    def get_system(self, user_id: str) -> Optional[TradingSystem]:
        """거래 시스템 조회"""
        return self.systems.get(user_id)
    
    async def cleanup_all(self):
        """모든 거래 시스템 정리"""
        for user_id in list(self.systems.keys()):
            await self._cleanup_system(user_id)

# 전역 매니저 인스턴스
trading_manager = TradingSystemManager()

# =================================
# API 라우터 설정
# =================================

router = APIRouter(prefix="/api/v1/trading", tags=["Trading"])

# =================================
# 거래 시스템 관리 엔드포인트
# =================================

@router.post("/system/initialize")
async def initialize_trading_system(
    okx_credentials: OKXCredentials,
    trading_config: TradingConfigRequest,
    current_user: dict = None  # 임시로 None 처리
):
    """거래 시스템 초기화"""
    try:
        user_id = current_user.get("user_id", "demo_user")
        
        system = await trading_manager.get_or_create_system(
            user_id=user_id,
            okx_credentials=okx_credentials,
            trading_config=trading_config
        )
        
        # 시스템 상태 조회
        status = await system.get_system_status()
        
        return {
            "status": "success",
            "message": "거래 시스템이 성공적으로 초기화되었습니다",
            "system_status": status,
            "config": {
                "mode": trading_config.mode,
                "max_positions": trading_config.max_positions,
                "max_daily_loss": trading_config.max_daily_loss,
                "environment": "sandbox" if okx_credentials.sandbox else "production"
            }
        }
        
    except Exception as e:
        logger.error(f"거래 시스템 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/status")
async def get_trading_system_status(current_user: dict = None):
    """거래 시스템 상태 조회"""
    try:
        user_id = current_user.get("user_id", "demo_user")
        system = trading_manager.get_system(user_id)
        
        if not system:
            raise HTTPException(status_code=404, detail="거래 시스템이 초기화되지 않았습니다")
        
        status = await system.get_system_status()
        return {
            "status": "success",
            "data": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"시스템 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================================
# 거래 실행 엔드포인트
# =================================

@router.post("/execute/signal")
async def execute_trading_signal(
    signal_request: TradingSignalRequest,
    current_user: dict = None  # 임시로 None 처리
):
    """거래 신호 실행"""
    try:
        user_id = current_user.get("user_id", "demo_user")
        system = trading_manager.get_system(user_id)
        
        if not system:
            raise HTTPException(status_code=404, detail="거래 시스템이 초기화되지 않았습니다")
        
        # 신호 데이터 준비
        signal_data = {
            "signal": signal_request.signal,
            "current_price": signal_request.current_price,
            "confidence": signal_request.confidence,
            "metadata": signal_request.metadata
        }
        
        # 신호 실행
        result = await system.execute_dantaro_signal(signal_request.symbol, signal_data)
        
        return {
            "status": "success",
            "signal_result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"거래 신호 실행 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute/batch_signals")
async def execute_batch_signals(
    signals: List[TradingSignalRequest],
    current_user: dict = None  # 임시로 None 처리
):
    """배치 거래 신호 실행"""
    try:
        user_id = current_user.get("user_id", "demo_user")
        system = trading_manager.get_system(user_id)
        
        if not system:
            raise HTTPException(status_code=404, detail="거래 시스템이 초기화되지 않았습니다")
        
        results = []
        for signal_request in signals:
            try:
                signal_data = {
                    "signal": signal_request.signal,
                    "current_price": signal_request.current_price,
                    "confidence": signal_request.confidence,
                    "metadata": signal_request.metadata
                }
                
                result = await system.execute_dantaro_signal(signal_request.symbol, signal_data)
                results.append({
                    "symbol": signal_request.symbol,
                    "result": result
                })
                
            except Exception as e:
                results.append({
                    "symbol": signal_request.symbol,
                    "result": {"status": "error", "error": str(e)}
                })
        
        return {
            "status": "success",
            "executed_signals": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"배치 신호 실행 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================================
# 포지션 관리 엔드포인트
# =================================

@router.get("/positions")
async def get_positions(current_user: dict = None):
    """현재 포지션 조회"""
    try:
        user_id = current_user.get("user_id", "demo_user")
        system = trading_manager.get_system(user_id)
        
        if not system:
            raise HTTPException(status_code=404, detail="거래 시스템이 초기화되지 않았습니다")
        
        status = await system.get_system_status()
        
        return {
            "status": "success",
            "active_positions": status.get("active_positions", 0),
            "positions": status.get("positions", []),
            "total_unrealized_pnl": status.get("unrealized_pnl", 0),
            "daily_pnl": status.get("daily_pnl", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"포지션 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/positions/close/{symbol}")
async def close_position(
    symbol: str,
    current_user: dict = None  # 임시로 None 처리
):
    """특정 포지션 청산"""
    try:
        user_id = current_user.get("user_id", "demo_user")
        system = trading_manager.get_system(user_id)
        
        if not system:
            raise HTTPException(status_code=404, detail="거래 시스템이 초기화되지 않았습니다")
        
        # 현재 포지션 확인
        if symbol not in system.positions:
            raise HTTPException(status_code=404, detail=f"포지션을 찾을 수 없습니다: {symbol}")
        
        position = system.positions[symbol]
        current_price = float(position.current_price)
        
        # 강제 매도 신호 생성
        signal_data = {
            "signal": "SELL",
            "current_price": current_price,
            "confidence": 1.0,
            "metadata": {"type": "manual_close"}
        }
        
        result = await system.execute_dantaro_signal(symbol, signal_data)
        
        return {
            "status": "success",
            "message": f"{symbol} 포지션이 청산되었습니다",
            "close_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"포지션 청산 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================================
# 위험 관리 엔드포인트
# =================================

@router.post("/emergency/stop")
async def emergency_stop(
    emergency_request: EmergencyStopRequest,
    current_user: dict = None  # 임시로 None 처리
):
    """비상 정지"""
    try:
        user_id = current_user.get("user_id", "demo_user")
        system = trading_manager.get_system(user_id)
        
        if not system:
            raise HTTPException(status_code=404, detail="거래 시스템이 초기화되지 않았습니다")
        
        logger.warning(f"비상 정지 요청: {emergency_request.reason}")
        
        result = {"status": "emergency_stop_activated", "reason": emergency_request.reason}
        
        if emergency_request.close_all_positions:
            close_result = await system.emergency_close_all()
            result["close_result"] = close_result
        else:
            system.emergency_stop = True
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"비상 정지 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emergency/resume")
async def resume_trading(current_user: dict = None):
    """거래 재개"""
    try:
        user_id = current_user.get("user_id", "demo_user")
        system = trading_manager.get_system(user_id)
        
        if not system:
            raise HTTPException(status_code=404, detail="거래 시스템이 초기화되지 않았습니다")
        
        system.emergency_stop = False
        logger.info("거래 재개됨")
        
        return {
            "status": "success",
            "message": "거래가 재개되었습니다",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"거래 재개 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================================
# 성과 분석 엔드포인트
# =================================

@router.get("/performance/summary")
async def get_performance_summary(current_user: dict = None):
    """성과 요약 조회"""
    try:
        user_id = current_user.get("user_id", "demo_user")
        system = trading_manager.get_system(user_id)
        
        if not system:
            raise HTTPException(status_code=404, detail="거래 시스템이 초기화되지 않았습니다")
        
        status = await system.get_system_status()
        
        # 성과 지표 계산
        total_pnl = status.get("total_pnl", 0)
        total_trades = status.get("total_trades", 0)
        win_rate = 0.0  # TODO: 승률 계산 로직 추가
        
        return {
            "status": "success",
            "performance": {
                "total_pnl": total_pnl,
                "daily_pnl": status.get("daily_pnl", 0),
                "unrealized_pnl": status.get("unrealized_pnl", 0),
                "total_trades": total_trades,
                "win_rate": win_rate,
                "active_positions": status.get("active_positions", 0),
                "average_trade_size": 0.0,  # TODO: 평균 거래 크기 계산
                "max_drawdown": 0.0,  # TODO: 최대 낙폭 계산
            },
            "risk_metrics": status.get("risk_status", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"성과 요약 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================================
# 헬스체크 및 시스템 정보
# =================================

@router.get("/health")
async def trading_health_check():
    """거래 시스템 헬스체크"""
    try:
        active_systems = len(trading_manager.systems)
        
        return {
            "status": "healthy",
            "active_trading_systems": active_systems,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"헬스체크 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

# =================================
# 시스템 종료 시 정리
# =================================

async def cleanup_trading_systems():
    """시스템 종료시 모든 거래 시스템 정리"""
    try:
        await trading_manager.cleanup_all()
        logger.info("모든 거래 시스템이 정리되었습니다")
    except Exception as e:
        logger.error(f"거래 시스템 정리 중 에러: {e}")

# FastAPI 애플리케이션 종료 이벤트에 등록
# (main.py에서 app.add_event_handler("shutdown", cleanup_trading_systems) 추가 필요)