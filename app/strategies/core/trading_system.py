# app/strategies/core/trading_system.py
"""
통합 거래 시스템 - 단타로 전략과 OKX Private API 연동
실제 거래 실행, 포지션 관리, 리스크 제어 등의 기능 제공
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import json

from ..utils.okx_private_client import (
    OKXPrivateClient, OKXConfig, OrderRequest, PositionInfo,
    create_market_order, create_limit_order
)
from ..plugins.dantaro_strategy import DantaroStrategy, DantaroConfig
from ..utils.okx_client_fixed import OKXClient

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """거래 모드"""
    SIMULATION = "simulation"  # 시뮬레이션 모드
    LIVE = "live"  # 실제 거래 모드
    
class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class TradingConfig:
    """거래 시스템 설정"""
    mode: TradingMode = TradingMode.SIMULATION
    max_positions: int = 5
    max_daily_loss: float = 1000.0  # 일일 최대 손실 (USDT)
    position_size_limit: float = 100.0  # 포지션당 최대 크기 (USDT)
    stop_loss_percentage: float = 5.0  # 스탑로스 비율 (%)
    take_profit_percentage: float = 2.0  # 익절 비율 (%)
    
@dataclass
class TradingPosition:
    """거래 포지션"""
    symbol: str
    side: str  # 'long', 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    unrealized_pnl: Decimal
    stage: int  # 물타기 단계
    orders: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def percentage_change(self) -> float:
        """수익률 계산"""
        if self.entry_price == 0:
            return 0.0
        return float((self.current_price - self.entry_price) / self.entry_price * 100)

class TradingSystem:
    """통합 거래 시스템"""
    
    def __init__(self, 
                 okx_config: OKXConfig,
                 dantaro_config: DantaroConfig,
                 trading_config: TradingConfig):
        self.okx_config = okx_config
        self.dantaro_config = dantaro_config
        self.trading_config = trading_config
        
        # 클라이언트 초기화
        self.private_client = None
        self.public_client = OKXClient()
        self.dantaro_strategy = DantaroStrategy(dantaro_config)
        
        # 상태 관리
        self.positions: Dict[str, TradingPosition] = {}
        self.daily_pnl: float = 0.0
        self.total_trades: int = 0
        self.is_running: bool = False
        
        # 리스크 관리
        self.emergency_stop: bool = False
        self.last_trade_time: Optional[datetime] = None
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.private_client = OKXPrivateClient(self.okx_config)
        await self.private_client.__aenter__()
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.private_client:
            await self.private_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def initialize(self) -> bool:
        """시스템 초기화"""
        try:
            logger.info("거래 시스템 초기화 시작...")
            
            # OKX 연결 테스트
            connection_test = await self.private_client.test_connection()
            if connection_test.get('status') != 'connected':
                logger.error(f"OKX 연결 실패: {connection_test}")
                return False
            
            logger.info(f"OKX 연결 성공 - 환경: {connection_test.get('environment')}")
            
            # 계정 정보 조회
            account_info = await self.private_client.get_account_info()
            logger.info(f"계정 레벨: {account_info.get('account_level')}")
            
            # 기존 포지션 로드
            await self._load_existing_positions()
            
            logger.info("거래 시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            return False
    
    async def _load_existing_positions(self):
        """기존 포지션 로드"""
        try:
            positions = await self.private_client.get_positions()
            logger.info(f"기존 포지션 {len(positions)}개 발견")
            
            for pos in positions:
                trading_pos = TradingPosition(
                    symbol=pos.symbol,
                    side='long' if pos.size > 0 else 'short',
                    entry_price=pos.average_price,
                    current_price=pos.average_price,  # 현재가는 별도 업데이트
                    size=abs(pos.size),
                    unrealized_pnl=pos.unrealized_pnl,
                    stage=1  # 기존 포지션은 1단계로 간주
                )
                self.positions[pos.symbol] = trading_pos
                
        except Exception as e:
            logger.error(f"기존 포지션 로드 실패: {e}")
    
    # =================================
    # 거래 실행 메서드
    # =================================
    
    async def execute_dantaro_signal(self, symbol: str, signal_data: Dict) -> Dict[str, Any]:
        """단타로 신호 실행"""
        try:
            signal_type = signal_data.get('signal')
            current_price = signal_data.get('current_price', 0)
            
            if signal_type == 'BUY':
                return await self._execute_buy_signal(symbol, current_price, signal_data)
            elif signal_type == 'ADD':
                return await self._execute_add_signal(symbol, current_price, signal_data)
            elif signal_type == 'SELL':
                return await self._execute_sell_signal(symbol, current_price, signal_data)
            else:
                return {'status': 'ignored', 'reason': f'Unknown signal: {signal_type}'}
                
        except Exception as e:
            logger.error(f"신호 실행 실패 ({symbol}): {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _execute_buy_signal(self, symbol: str, price: float, signal_data: Dict) -> Dict[str, Any]:
        """매수 신호 실행 (1단계 진입)"""
        try:
            # 리스크 체크
            if not await self._check_risk_limits(symbol, 'buy'):
                return {'status': 'rejected', 'reason': 'Risk limits exceeded'}
            
            # 주문 크기 계산
            base_amount_usdt = self.dantaro_config.base_amount / 1000  # 원 -> USDT 근사 변환
            order_size = self._calculate_order_size(symbol, base_amount_usdt, price)
            
            if self.trading_config.mode == TradingMode.SIMULATION:
                # 시뮬레이션 모드
                result = await self._simulate_order(symbol, 'buy', order_size, price)
            else:
                # 실제 거래 모드
                order = create_market_order(symbol, 'buy', str(order_size))
                result = await self.private_client.place_order(order)
            
            if result.get('order_id') or result.get('status') == 'simulated':
                # 포지션 생성
                trading_pos = TradingPosition(
                    symbol=symbol,
                    side='long',
                    entry_price=Decimal(str(price)),
                    current_price=Decimal(str(price)),
                    size=Decimal(str(order_size)),
                    unrealized_pnl=Decimal('0'),
                    stage=1
                )
                self.positions[symbol] = trading_pos
                
                logger.info(f"매수 주문 실행: {symbol} at {price} USDT")
                return {
                    'status': 'success',
                    'action': 'buy',
                    'symbol': symbol,
                    'price': price,
                    'size': order_size,
                    'stage': 1,
                    'order_id': result.get('order_id', 'simulated')
                }
            else:
                return {'status': 'failed', 'error': result.get('error', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"매수 신호 실행 실패: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _execute_add_signal(self, symbol: str, price: float, signal_data: Dict) -> Dict[str, Any]:
        """추가 매수 신호 실행 (물타기)"""
        try:
            if symbol not in self.positions:
                return {'status': 'rejected', 'reason': 'No existing position'}
            
            position = self.positions[symbol]
            next_stage = position.stage + 1
            
            if next_stage > self.dantaro_config.max_stages:
                return {'status': 'rejected', 'reason': 'Max stages reached'}
            
            # 추가 매수 금액 계산
            stage_amount = self.dantaro_config.base_amount * (self.dantaro_config.multiplier ** (next_stage - 1))
            stage_amount_usdt = stage_amount / 1000  # 원 -> USDT 근사 변환
            order_size = self._calculate_order_size(symbol, stage_amount_usdt, price)
            
            if self.trading_config.mode == TradingMode.SIMULATION:
                result = await self._simulate_order(symbol, 'buy', order_size, price)
            else:
                order = create_market_order(symbol, 'buy', str(order_size))
                result = await self.private_client.place_order(order)
            
            if result.get('order_id') or result.get('status') == 'simulated':
                # 포지션 업데이트 (평균단가 계산)
                total_cost = float(position.entry_price * position.size) + (price * order_size)
                total_size = float(position.size) + order_size
                new_avg_price = total_cost / total_size
                
                position.entry_price = Decimal(str(new_avg_price))
                position.size = Decimal(str(total_size))
                position.stage = next_stage
                
                logger.info(f"추가 매수 실행: {symbol} Stage {next_stage} at {price} USDT")
                return {
                    'status': 'success',
                    'action': 'add_buy',
                    'symbol': symbol,
                    'price': price,
                    'size': order_size,
                    'stage': next_stage,
                    'new_avg_price': new_avg_price,
                    'order_id': result.get('order_id', 'simulated')
                }
            else:
                return {'status': 'failed', 'error': result.get('error', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"추가 매수 실행 실패: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _execute_sell_signal(self, symbol: str, price: float, signal_data: Dict) -> Dict[str, Any]:
        """매도 신호 실행"""
        try:
            if symbol not in self.positions:
                return {'status': 'rejected', 'reason': 'No position to sell'}
            
            position = self.positions[symbol]
            sell_size = float(position.size)
            
            if self.trading_config.mode == TradingMode.SIMULATION:
                result = await self._simulate_order(symbol, 'sell', sell_size, price)
            else:
                order = create_market_order(symbol, 'sell', str(sell_size))
                result = await self.private_client.place_order(order)
            
            if result.get('order_id') or result.get('status') == 'simulated':
                # 수익 계산
                profit = (price - float(position.entry_price)) * sell_size
                profit_percentage = position.percentage_change
                
                # 포지션 제거
                del self.positions[symbol]
                
                # 일일 손익 업데이트
                self.daily_pnl += profit
                self.total_trades += 1
                
                logger.info(f"매도 완료: {symbol} at {price} USDT, 수익: {profit:.2f} USDT ({profit_percentage:.2f}%)")
                return {
                    'status': 'success',
                    'action': 'sell',
                    'symbol': symbol,
                    'price': price,
                    'size': sell_size,
                    'profit': profit,
                    'profit_percentage': profit_percentage,
                    'order_id': result.get('order_id', 'simulated')
                }
            else:
                return {'status': 'failed', 'error': result.get('error', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"매도 실행 실패: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # =================================
    # 리스크 관리 메서드
    # =================================
    
    async def _check_risk_limits(self, symbol: str, action: str) -> bool:
        """리스크 한도 체크"""
        try:
            # 비상 정지 상태 체크
            if self.emergency_stop:
                logger.warning("비상 정지 상태 - 거래 차단")
                return False
            
            # 최대 포지션 수 체크
            if action == 'buy' and len(self.positions) >= self.trading_config.max_positions:
                logger.warning(f"최대 포지션 수 초과: {len(self.positions)}/{self.trading_config.max_positions}")
                return False
            
            # 일일 손실 한도 체크
            if self.daily_pnl <= -self.trading_config.max_daily_loss:
                logger.warning(f"일일 손실 한도 초과: {self.daily_pnl:.2f} USDT")
                self.emergency_stop = True
                return False
            
            # 거래 빈도 체크 (과도한 거래 방지)
            if self.last_trade_time:
                time_since_last = datetime.now() - self.last_trade_time
                if time_since_last < timedelta(seconds=10):  # 10초 간격 제한
                    logger.warning("거래 빈도 제한 - 너무 빈번한 거래")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"리스크 체크 실패: {e}")
            return False
    
    def _calculate_order_size(self, symbol: str, amount_usdt: float, price: float) -> float:
        """주문 크기 계산"""
        try:
            # USDT 금액을 코인 수량으로 변환
            coin_amount = amount_usdt / price
            
            # 최소 주문 단위로 반올림 (보통 소수점 6자리)
            rounded_amount = round(coin_amount, 6)
            
            return max(rounded_amount, 0.000001)  # 최소 주문량 보장
            
        except Exception as e:
            logger.error(f"주문 크기 계산 실패: {e}")
            return 0.0
    
    async def _simulate_order(self, symbol: str, side: str, size: float, price: float) -> Dict[str, Any]:
        """주문 시뮬레이션"""
        import uuid
        
        # 시뮬레이션 결과 생성
        return {
            'status': 'simulated',
            'order_id': f'sim_{uuid.uuid4().hex[:8]}',
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'timestamp': datetime.now().isoformat()
        }
    
    # =================================
    # 모니터링 및 상태 조회 메서드
    # =================================
    
    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            # 현재 포지션 업데이트
            await self._update_positions_price()
            
            # 총 손익 계산
            total_unrealized_pnl = sum(float(pos.unrealized_pnl) for pos in self.positions.values())
            
            return {
                'status': 'running' if self.is_running else 'stopped',
                'mode': self.trading_config.mode.value,
                'emergency_stop': self.emergency_stop,
                'active_positions': len(self.positions),
                'daily_pnl': self.daily_pnl,
                'unrealized_pnl': total_unrealized_pnl,
                'total_pnl': self.daily_pnl + total_unrealized_pnl,
                'total_trades': self.total_trades,
                'risk_status': {
                    'max_positions': f"{len(self.positions)}/{self.trading_config.max_positions}",
                    'daily_loss_limit': f"{self.daily_pnl:.2f}/{-self.trading_config.max_daily_loss:.2f}",
                },
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'size': float(pos.size),
                        'entry_price': float(pos.entry_price),
                        'current_price': float(pos.current_price),
                        'unrealized_pnl': float(pos.unrealized_pnl),
                        'percentage': pos.percentage_change,
                        'stage': pos.stage
                    }
                    for pos in self.positions.values()
                ]
            }
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 실패: {e}")
            return {'error': str(e)}
    
    async def _update_positions_price(self):
        """포지션 현재가 업데이트"""
        try:
            for symbol, position in self.positions.items():
                # Public API로 현재가 조회
                ticker_data = await self.public_client.get_ticker(symbol)
                if ticker_data and 'last' in ticker_data:
                    current_price = Decimal(str(ticker_data['last']))
                    position.current_price = current_price
                    
                    # 미실현 손익 계산
                    pnl = (current_price - position.entry_price) * position.size
                    position.unrealized_pnl = pnl
                    
        except Exception as e:
            logger.error(f"포지션 가격 업데이트 실패: {e}")
    
    async def emergency_close_all(self) -> Dict[str, Any]:
        """비상 청산"""
        logger.warning("비상 청산 시작...")
        self.emergency_stop = True
        
        results = []
        for symbol, position in list(self.positions.items()):
            try:
                current_price = float(position.current_price)
                sell_result = await self._execute_sell_signal(symbol, current_price, {'signal': 'EMERGENCY_SELL'})
                results.append({
                    'symbol': symbol,
                    'result': sell_result
                })
            except Exception as e:
                logger.error(f"비상 청산 실패 ({symbol}): {e}")
                results.append({
                    'symbol': symbol,
                    'result': {'status': 'error', 'error': str(e)}
                })
        
        return {
            'status': 'emergency_close_completed',
            'closed_positions': len(results),
            'results': results
        }

# =================================
# 헬퍼 함수들
# =================================

def create_trading_system(
    api_key: str,
    secret_key: str, 
    passphrase: str,
    sandbox: bool = True,
    max_positions: int = 5,
    max_daily_loss: float = 1000.0
) -> TradingSystem:
    """거래 시스템 생성"""
    
    okx_config = OKXConfig(
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
        sandbox=sandbox
    )
    
    dantaro_config = DantaroConfig()
    
    trading_config = TradingConfig(
        mode=TradingMode.SIMULATION,
        max_positions=max_positions,
        max_daily_loss=max_daily_loss
    )
    
    return TradingSystem(okx_config, dantaro_config, trading_config)

# =================================
# 테스트 및 데모 코드
# =================================

async def demo_trading_system():
    """거래 시스템 데모"""
    # 테스트용 설정 (실제 API 키 필요)
    system = create_trading_system(
        api_key="test_key",
        secret_key="test_secret",
        passphrase="test_pass",
        sandbox=True
    )
    
    async with system:
        # 시스템 상태 확인
        status = await system.get_system_status()
        print(f"시스템 상태: {status}")
        
        # 테스트 신호 실행
        test_signal = {
            'signal': 'BUY',
            'current_price': 42000.0,
            'confidence': 0.85
        }
        
        result = await system.execute_dantaro_signal('BTC-USDT-SWAP', test_signal)
        print(f"신호 실행 결과: {result}")

if __name__ == "__main__":
    print("🚀 통합 거래 시스템 테스트")
    print("실제 사용을 위해서는 API 키 설정이 필요합니다.")
    
    # asyncio.run(demo_trading_system())  # 실제 키가 있을 때만 실행