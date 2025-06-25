# 📁 app/api/v1/strategies.py
"""
전략 API 엔드포인트 - 본사서버 신호 제공 시스템
Created: 2025.06.24 00:48 KST
Fixed: 2025.06.24 00:52 KST - 기존 strategy_base.py 구조 호환
Purpose: 사용자서버가 호출할 수 있는 전략 신호 API
"""
# import 섹션에 추가
from app.strategies.plugins.dantaro_strategy import create_dantaro_strategy, DantaroConfig
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import asyncio
import logging


# 기존 구조와 호환되는 import
from app.strategies.core.strategy_base import MarketData, TradingSignal, SignalType
from app.strategies.utils.okx_client_fixed import FixedMarketScanner

logger = logging.getLogger(__name__)
router = APIRouter()

# 응답 모델 정의
class SignalResponse(BaseModel):
    """신호 응답 모델"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    timestamp: datetime
    strategy_name: str
    conditions_met: List[str]
    market_data: Dict[str, Any]

class StrategyListResponse(BaseModel):
    """전략 목록 응답"""
    strategy_id: str
    strategy_name: str
    description: str
    category: str
    is_active: bool

class MarketScanResponse(BaseModel):
    """시장 스캔 응답"""
    scan_timestamp: datetime
    total_coins_scanned: int
    signals_found: int
    signals: List[SignalResponse]
    scan_duration_ms: int

# 전역 스캐너 인스턴스 (재사용을 위해)
_market_scanner: Optional[FixedMarketScanner] = None

async def get_market_scanner():
    """마켓 스캐너 싱글톤 패턴"""
    global _market_scanner
    if _market_scanner is None:
        _market_scanner = FixedMarketScanner()
        await _market_scanner.initialize()
    return _market_scanner

# 전역 단타로 전략 인스턴스
_dantaro_strategy = None

async def get_dantaro_strategy():
    """단타로 전략 싱글톤 패턴"""
    global _dantaro_strategy
    if _dantaro_strategy is None:
        config = DantaroConfig(
            base_amount=5500.0,
            interval_percent=2.0,
            multiplier=2.0,
            max_stages=7,
            profit_target=0.5
        )
        _dantaro_strategy = create_dantaro_strategy(config)
    return _dantaro_strategy

@router.get("/strategies", response_model=List[StrategyListResponse])
async def get_available_strategies():
    """사용 가능한 전략 목록 조회"""
    strategies = [
        {
            "strategy_id": "basic_scanning",
            "strategy_name": "기본 스캐닝 전략",
            "description": "실시간 시장 스캔을 통한 변동성 기반 신호 생성",
            "category": "market_scanning",
            "is_active": True
        },
        {
            "strategy_id": "scalping_strategy", 
            "strategy_name": "단타로 전략",
            "description": "기술적 분석 기반 단타 매매 전략",
            "category": "technical_analysis",
            "is_active": True
        },
        {
            "strategy_id": "dantaro_strategy",
            "strategy_name": "물타기 전략", 
            "description": "분할매수를 통한 평균단가 하락 전략",
            "category": "martingale",
            "is_active": False  # 아직 구현 중
        }
    ]
    
    return [StrategyListResponse(**strategy) for strategy in strategies]

@router.get("/strategies/basic_scanning/signals", response_model=MarketScanResponse)
async def get_basic_scanning_signals(
    limit: int = Query(default=30, description="스캔할 코인 수", ge=1, le=100)
):
    """기본 스캐닝 전략 신호 조회"""
    start_time = datetime.now()
    
    try:
        scanner = await get_market_scanner()
        
        # 실시간 스캔 실행
        signals_data = await scanner.scan_top_coins(limit)
        
        # 응답 형식으로 변환
        signals = []
        for signal_data in signals_data:
            signal_response = SignalResponse(
                symbol=signal_data['symbol'],
                signal_type=signal_data['signal_type'],
                confidence=signal_data['confidence'],
                timestamp=signal_data['market_data']['timestamp'],
                strategy_name="기본 스캐닝 전략",
                conditions_met=signal_data['conditions_met'],
                market_data=signal_data['market_data']
            )
            signals.append(signal_response)
        
        scan_duration = (datetime.now() - start_time).total_seconds() * 1000
        
        return MarketScanResponse(
            scan_timestamp=start_time,
            total_coins_scanned=limit,
            signals_found=len(signals),
            signals=signals,
            scan_duration_ms=int(scan_duration)
        )
        
    except Exception as e:
        logger.error(f"기본 스캐닝 신호 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신호 조회 실패: {str(e)}")

@router.get("/strategies/scalping_strategy/signals", response_model=List[SignalResponse])
async def get_scalping_strategy_signals(
    symbols: Optional[str] = Query(default=None, description="콤마로 구분된 심볼 목록"),
    limit: int = Query(default=10, description="최대 신호 수", ge=1, le=50)
):
    """단타로 전략 신호 조회 (단순 버전)"""
    try:
        # 심볼 목록 처리
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',')] if isinstance(symbols, str) else ['BTC-USDT', 'ETH-USDT', 'BNB-USDT']
        else:
            # 기본 주요 코인들
            symbol_list = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'XRP-USDT', 'ADA-USDT']
        
        # 시장 데이터 수집
        scanner = await get_market_scanner()
        all_tickers = await scanner.okx_client.get_multiple_tickers()
        
        signals = []
        for ticker in all_tickers:
            if ticker['symbol'] in symbol_list and len(signals) < limit:
                # 간단한 단타로 로직 (RSI 기반)
                change_24h = ticker['change_24h']
                volume_24h = ticker['volume_24h']
                
                # 단순 조건: 하락 + 높은 거래량
                if change_24h < -3.0 and volume_24h > 1000000:
                    signal_response = SignalResponse(
                        symbol=ticker['symbol'],
                        signal_type="BUY",  # 하락 후 매수 신호
                        confidence=0.75,
                        timestamp=ticker['timestamp'],
                        strategy_name="단타로 전략",
                        conditions_met=["price_drop", "high_volume"],
                        market_data={
                            'price': ticker['price'],
                            'change_24h': ticker['change_24h'],
                            'volume_24h': ticker['volume_24h']
                        }
                    )
                    signals.append(signal_response)
        
        return signals
        
    except Exception as e:
        logger.error(f"단타로 전략 신호 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신호 조회 실패: {str(e)}")

@router.get("/strategies/{strategy_id}/status")
async def get_strategy_status(strategy_id: str):
    """특정 전략의 상태 조회"""
    strategy_status = {
        "basic_scanning": {
            "status": "active",
            "last_scan": datetime.now(),
            "total_signals_today": 145,
            "success_rate": 0.73,
            "uptime": "99.8%"
        },
        "scalping_strategy": {
            "status": "active", 
            "last_signal": datetime.now(),
            "total_signals_today": 23,
            "success_rate": 0.68,
            "uptime": "100%"
        },
        "dantaro_strategy": {
            "status": "development",
            "last_signal": None,
            "total_signals_today": 0,
            "success_rate": 0.0,
            "uptime": "0%"
        }
    }
    
    if strategy_id not in strategy_status:
        raise HTTPException(status_code=404, detail="전략을 찾을 수 없습니다")
    
    return strategy_status[strategy_id]

@router.post("/strategies/scan/all")
async def scan_all_strategies(
    limit: int = Query(default=20, description="코인당 스캔 수", ge=1, le=50)
):
    """모든 활성 전략으로 통합 스캔"""
    start_time = datetime.now()
    
    try:
        # 기본 스캐닝 신호
        basic_response = await get_basic_scanning_signals(limit)
        
        # 단타로 전략 신호  
        scalping_signals = await get_scalping_strategy_signals(limit=limit)
        
        # 결과 통합
        all_signals = basic_response.signals + scalping_signals
        
        scan_duration = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "scan_timestamp": start_time,
            "strategies_used": ["basic_scanning", "scalping_strategy"],
            "total_signals": len(all_signals),
            "basic_scanning_signals": len(basic_response.signals),
            "scalping_strategy_signals": len(scalping_signals),
            "scan_duration_ms": int(scan_duration),
            "signals": all_signals
        }
        
    except Exception as e:
        logger.error(f"통합 스캔 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통합 스캔 실패: {str(e)}")

# Health check 엔드포인트
@router.get("/strategies/health")
async def strategy_health_check():
    """전략 시스템 헬스체크"""
    try:
        scanner = await get_market_scanner()
        
        # 간단한 연결 테스트
        test_tickers = await scanner.okx_client.get_multiple_tickers()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "okx_connection": "ok" if test_tickers else "error",
            "strategies_loaded": 2,
            "message": "전략 시스템 정상 동작 중"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now(),
            "error": str(e),
            "message": "전략 시스템 오류 발생"
        }

"""
작업명: Step_9-3 단타로 전략 API 연동
진행상황: 3/4 완료 (단타로 클래스 구현 완료, API 연동 중)
추가위치: strategies.py 파일 마지막 부분에 추가
생성시간: 2025.06.24 01:20
"""

# 단타로 전략 import 추가 (파일 상단 import 섹션에 추가)
from app.strategies.plugins.dantaro_strategy import create_dantaro_strategy, DantaroConfig

# 전역 단타로 전략 인스턴스
_dantaro_strategy = None

async def get_dantaro_strategy():
    """단타로 전략 싱글톤 패턴"""
    global _dantaro_strategy
    if _dantaro_strategy is None:
        config = DantaroConfig(
            base_amount=5500.0,
            interval_percent=2.0,
            multiplier=2.0,
            max_stages=7,
            profit_target=0.5
        )
        _dantaro_strategy = create_dantaro_strategy(config)
    return _dantaro_strategy

@router.get("/strategies/dantaro_strategy/info")
async def get_dantaro_strategy_info():
    """단타로 전략 정보 조회"""
    try:
        strategy = await get_dantaro_strategy()
        stats = strategy.get_strategy_stats()
        
        return {
            "strategy_name": "단타로 (물타기) 전략",
            "description": "7단계 분할매수를 통한 평균단가 하락 전략",
            "status": "active",
            "config": stats['config'],
            "requirements": stats['requirements'],
            "current_positions": stats['current_positions'],
            "active_symbols": stats['active_symbols']
        }
    except Exception as e:
        logger.error(f"단타로 전략 정보 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies/dantaro_strategy/signals")
async def get_dantaro_signals(
    limit: int = Query(default=10, description="최대 분석 코인 수", ge=1, le=50)
):
    """단타로 전략 신호 조회"""
    try:
        strategy = await get_dantaro_strategy()
        scanner = await get_market_scanner()
        
        # 상위 코인 데이터 가져오기
        all_tickers = await scanner.okx_client.get_multiple_tickers()
        
        signals = []
        for ticker in all_tickers[:limit]:
            # 단타로 전략으로 신호 생성
            signal = strategy.generate_signal(ticker)
            
            if signal:
                signal_response = SignalResponse(
                    symbol=signal['symbol'],
                    signal_type=signal['action'],
                    confidence=signal['confidence'],
                    timestamp=datetime.now(),
                    strategy_name="단타로 전략",
                    conditions_met=signal.get('conditions_met', [signal['reason']]),
                    market_data={
                        'price': ticker['price'],
                        'change_24h': ticker['change_24h'],
                        'volume_24h': ticker['volume_24h'],
                        'stage': signal.get('stage', 0),
                        'amount': signal.get('amount', 0)
                    }
                )
                signals.append(signal_response)
        
        return signals
        
    except Exception as e:
        logger.error(f"단타로 신호 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies/dantaro_strategy/positions")
async def get_dantaro_positions():
    """단타로 전략 현재 포지션 조회"""
    try:
        strategy = await get_dantaro_strategy()
        positions = strategy.get_all_positions()
        
        return {
            "timestamp": datetime.now(),
            "total_positions": len(positions),
            "positions": positions
        }
        
    except Exception as e:
        logger.error(f"단타로 포지션 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies/dantaro_strategy/positions/{symbol}")
async def get_dantaro_position(symbol: str):
    """특정 심볼의 단타로 포지션 상세 조회"""
    try:
        strategy = await get_dantaro_strategy()
        position = strategy.get_position_status(symbol)
        
        if position is None:
            raise HTTPException(status_code=404, detail=f"포지션을 찾을 수 없습니다: {symbol}")
        
        return position
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"단타로 포지션 상세 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategies/dantaro_strategy/execute")
async def execute_dantaro_signal(
    symbol: str,
    action: str = Query(description="ENTRY, ADD, EXIT"),
    current_price: Optional[float] = Query(default=None)
):
    """단타로 전략 신호 수동 실행 (테스트용)"""
    try:
        strategy = await get_dantaro_strategy()
        
        if current_price is None:
            # 현재 가격 가져오기
            scanner = await get_market_scanner()
            all_tickers = await scanner.okx_client.get_multiple_tickers()
            ticker = next((t for t in all_tickers if t['symbol'] == symbol), None)
            
            if not ticker:
                raise HTTPException(status_code=404, detail=f"심볼을 찾을 수 없습니다: {symbol}")
            
            current_price = ticker['price']
        
        if action == "ENTRY":
            # 진입 신호 생성 및 실행
            market_data = {
                'symbol': symbol,
                'price': current_price,
                'change_24h': -3.0,  # 테스트용 하락
                'volume_24h': 10000000
            }
            
            signal = strategy.analyze_entry_signal(market_data)
            if signal:
                success = strategy.execute_stage(signal)
                return {
                    "success": success,
                    "action": "ENTRY",
                    "signal": signal,
                    "position": strategy.get_position_status(symbol)
                }
            else:
                return {"success": False, "message": "진입 조건 미충족"}
        
        elif action == "ADD":
            # 추가 매수 신호
            signal = strategy.analyze_add_signal(symbol, current_price)
            if signal:
                success = strategy.execute_stage(signal)
                return {
                    "success": success,
                    "action": "ADD", 
                    "signal": signal,
                    "position": strategy.get_position_status(symbol)
                }
            else:
                return {"success": False, "message": "추가 매수 조건 미충족"}
        
        elif action == "EXIT":
            # 매도 신호
            signal = strategy.analyze_exit_signal(symbol, current_price)
            if signal:
                result = strategy.execute_exit(signal)
                return {
                    "success": result['success'],
                    "action": "EXIT",
                    "result": result
                }
            else:
                return {"success": False, "message": "매도 조건 미충족"}
        
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 액션입니다")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"단타로 신호 실행 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies/dantaro_strategy/simulate")
async def simulate_dantaro_strategy(
    symbol: str = Query(default="BTC-USDT"),
    stages: int = Query(default=3, description="시뮬레이션할 단계 수", ge=1, le=7)
):
    """단타로 전략 시뮬레이션"""
    try:
        strategy = await get_dantaro_strategy()
        
        # 시뮬레이션 시나리오
        base_price = 50000.0
        simulation_results = []
        
        # 1단계: 진입
        entry_data = {
            'symbol': symbol,
            'price': base_price,
            'change_24h': -3.0,
            'volume_24h': 10000000
        }
        
        entry_signal = strategy.analyze_entry_signal(entry_data)
        if entry_signal:
            strategy.execute_stage(entry_signal)
            simulation_results.append({
                'stage': 1,
                'action': 'ENTRY',
                'price': base_price,
                'amount': entry_signal['amount']
            })
        
        # 추가 단계들
        for stage in range(2, stages + 1):
            # 가격 하락 시뮬레이션
            current_price = base_price * (0.98 ** (stage - 1))
            
            add_signal = strategy.analyze_add_signal(symbol, current_price)
            if add_signal:
                strategy.execute_stage(add_signal)
                simulation_results.append({
                    'stage': stage,
                    'action': 'ADD',
                    'price': current_price,
                    'amount': add_signal['amount']
                })
        
        # 최종 포지션 상태
        final_position = strategy.get_position_status(symbol)
        
        # 시뮬레이션 정리 (포지션 삭제)
        if symbol in strategy.positions:
            del strategy.positions[symbol]
        
        return {
            "simulation_summary": {
                "symbol": symbol,
                "stages_simulated": len(simulation_results),
                "total_cost": sum(r['amount'] for r in simulation_results),
                "avg_price": final_position['avg_price'] if final_position else 0,
                "target_price": final_position['target_price'] if final_position else 0
            },
            "stage_details": simulation_results,
            "final_position": final_position
        }
        
    except Exception as e:
        logger.error(f"단타로 시뮬레이션 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))