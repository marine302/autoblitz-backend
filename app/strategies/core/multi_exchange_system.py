# app/strategies/core/multi_exchange_system.py
"""
다중 거래소 통합 시스템 - OKX와 업비트 동시 지원
거래소별 최적화된 전략 실행 및 통합 관리
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import json

# OKX 관련 임포트
from ..utils.okx_private_client import (
    OKXPrivateClient, OKXConfig, OrderRequest as OKXOrderRequest
)
from ..utils.okx_client_fixed import OKXClient

# 업비트 관련 임포트
from ..utils.upbit_client import (
    UpbitClient, UpbitConfig, UpbitOrderRequest,
    create_buy_order, create_sell_order, create_market_buy_order
)

# 전략 관련 임포트
from ..plugins.dantaro_strategy import DantaroStrategy, DantaroConfig

logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """거래소 타입"""
    OKX = "okx"
    UPBIT = "upbit"

class MarketType(Enum):
    """마켓 타입"""
    SPOT = "spot"      # 현물
    FUTURES = "futures" # 선물
    
@dataclass
class ExchangeConfig:
    """거래소 설정"""
    exchange_type: ExchangeType
    market_type: MarketType
    credentials: Dict[str, str]
    enabled: bool = True
    
@dataclass
class UnifiedPosition:
    """통합 포지션 정보"""
    exchange: ExchangeType
    symbol: str
    market_type: MarketType
    side: str  # 'long', 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    unrealized_pnl: Decimal
    stage: int  # 물타기 단계
    currency: str  # 'USDT', 'KRW'
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def percentage_change(self) -> float:
        """수익률 계산"""
        if self.entry_price == 0:
            return 0.0
        return float((self.current_price - self.entry_price) / self.entry_price * 100)
    
    @property
    def value_krw(self) -> Decimal:
        """KRW 기준 가치 (환율 1300 가정)"""
        if self.currency == 'KRW':
            return self.current_price * self.size
        else:  # USDT
            return self.current_price * self.size * Decimal('1300')

@dataclass
class MultiExchangeConfig:
    """다중 거래소 설정"""
    okx_config: Optional[ExchangeConfig] = None
    upbit_config: Optional[ExchangeConfig] = None
    dantaro_config: DantaroConfig = field(default_factory=DantaroConfig)
    preferred_exchange: ExchangeType = ExchangeType.OKX
    max_total_positions: int = 10
    max_daily_loss_krw: float = 1000000.0  # 100만원
    currency_conversion_rate: float = 1300.0  # USDT to KRW

class MultiExchangeSystem:
    """다중 거래소 통합 시스템"""
    
    def __init__(self, config: MultiExchangeConfig):
        self.config = config
        
        # 클라이언트 초기화
        self.okx_private_client = None
        self.okx_public_client = OKXClient()
        self.upbit_client = None
        
        # 전략 시스템
        self.dantaro_strategy = DantaroStrategy(config.dantaro_config)
        
        # 상태 관리
        self.positions: Dict[str, UnifiedPosition] = {}  # key: f"{exchange}:{symbol}"
        self.daily_pnl_krw: float = 0.0
        self.total_trades: int = 0
        self.is_running: bool = False
        
        # 거래소별 상태
        self.exchange_status: Dict[ExchangeType, Dict] = {
            ExchangeType.OKX: {'connected': False, 'last_error': None},
            ExchangeType.UPBIT: {'connected': False, 'last_error': None}
        }
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.cleanup()
    
    async def initialize(self) -> bool:
        """시스템 초기화"""
        try:
            logger.info("다중 거래소 시스템 초기화 시작...")
            
            # OKX 초기화
            if self.config.okx_config and self.config.okx_config.enabled:
                await self._initialize_okx()
            
            # 업비트 초기화
            if self.config.upbit_config and self.config.upbit_config.enabled:
                await self._initialize_upbit()
            
            # 기존 포지션 로드
            await self._load_existing_positions()
            
            self.is_running = True
            logger.info("다중 거래소 시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            return False
    
    async def _initialize_okx(self):
        """OKX 초기화"""
        try:
            if not self.config.okx_config:
                return
                
            creds = self.config.okx_config.credentials
            okx_config = OKXConfig(
                api_key=creds['api_key'],
                secret_key=creds['secret_key'],
                passphrase=creds['passphrase'],
                sandbox=creds.get('sandbox', True)
            )
            
            self.okx_private_client = OKXPrivateClient(okx_config)
            await self.okx_private_client.__aenter__()
            
            # 연결 테스트
            test_result = await self.okx_private_client.test_connection()
            if test_result.get('status') == 'connected':
                self.exchange_status[ExchangeType.OKX]['connected'] = True
                logger.info(f"OKX 연결 성공 - 환경: {test_result.get('environment')}")
            else:
                raise Exception(f"OKX 연결 실패: {test_result}")
                
        except Exception as e:
            logger.error(f"OKX 초기화 실패: {e}")
            self.exchange_status[ExchangeType.OKX]['last_error'] = str(e)
    
    async def _initialize_upbit(self):
        """업비트 초기화"""
        try:
            if not self.config.upbit_config:
                return
                
            creds = self.config.upbit_config.credentials
            upbit_config = UpbitConfig(
                access_key=creds['access_key'],
                secret_key=creds['secret_key']
            )
            
            self.upbit_client = UpbitClient(upbit_config)
            await self.upbit_client.__aenter__()
            
            # 연결 테스트
            test_result = await self.upbit_client.test_connection()
            if test_result.get('status') == 'connected':
                self.exchange_status[ExchangeType.UPBIT]['connected'] = True
                logger.info(f"업비트 연결 성공 - 마켓: {test_result.get('krw_markets')}개")
            else:
                raise Exception(f"업비트 연결 실패: {test_result}")
                
        except Exception as e:
            logger.error(f"업비트 초기화 실패: {e}")
            self.exchange_status[ExchangeType.UPBIT]['last_error'] = str(e)
    
    async def _load_existing_positions(self):
        """기존 포지션 로드"""
        try:
            # OKX 포지션 로드
            if self.okx_private_client and self.exchange_status[ExchangeType.OKX]['connected']:
                okx_positions = await self.okx_private_client.get_positions()
                for pos in okx_positions:
                    unified_pos = UnifiedPosition(
                        exchange=ExchangeType.OKX,
                        symbol=pos.symbol,
                        market_type=MarketType.FUTURES if 'SWAP' in pos.symbol else MarketType.SPOT,
                        side='long' if pos.size > 0 else 'short',
                        entry_price=pos.average_price,
                        current_price=pos.average_price,
                        size=abs(pos.size),
                        unrealized_pnl=pos.unrealized_pnl,
                        stage=1,
                        currency='USDT'
                    )
                    key = f"{ExchangeType.OKX.value}:{pos.symbol}"
                    self.positions[key] = unified_pos
            
            # 업비트 포지션은 현물만이므로 잔고에서 확인
            if self.upbit_client and self.exchange_status[ExchangeType.UPBIT]['connected']:
                balances = await self.upbit_client.get_accounts()
                for balance in balances:
                    if balance.currency != 'KRW' and balance.balance > 0:
                        # 암호화폐 보유량이 있는 경우 포지션으로 간주
                        market_symbol = f"KRW-{balance.currency}"
                        current_price = await self.upbit_client.get_market_price(market_symbol)
                        
                        if current_price:
                            unified_pos = UnifiedPosition(
                                exchange=ExchangeType.UPBIT,
                                symbol=market_symbol,
                                market_type=MarketType.SPOT,
                                side='long',
                                entry_price=balance.avg_buy_price,
                                current_price=current_price,
                                size=balance.balance,
                                unrealized_pnl=(current_price - balance.avg_buy_price) * balance.balance,
                                stage=1,
                                currency='KRW'
                            )
                            key = f"{ExchangeType.UPBIT.value}:{market_symbol}"
                            self.positions[key] = unified_pos
            
            logger.info(f"기존 포지션 {len(self.positions)}개 로드 완료")
            
        except Exception as e:
            logger.error(f"기존 포지션 로드 실패: {e}")
    
    # =================================
    # 거래 실행 메서드
    # =================================
    
    async def execute_signal(self, exchange: ExchangeType, symbol: str, 
                           signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """통합 신호 실행"""
        try:
            if exchange == ExchangeType.OKX:
                return await self._execute_okx_signal(symbol, signal_data)
            elif exchange == ExchangeType.UPBIT:
                return await self._execute_upbit_signal(symbol, signal_data)
            else:
                return {'status': 'error', 'error': f'지원하지 않는 거래소: {exchange}'}
                
        except Exception as e:
            logger.error(f"신호 실행 실패 ({exchange.value}:{symbol}): {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _execute_okx_signal(self, symbol: str, signal_data: Dict) -> Dict[str, Any]:
        """OKX 신호 실행"""
        try:
            if not self.okx_private_client or not self.exchange_status[ExchangeType.OKX]['connected']:
                return {'status': 'error', 'error': 'OKX 연결되지 않음'}
            
            signal_type = signal_data.get('signal')
            current_price = signal_data.get('current_price', 0)
            
            position_key = f"{ExchangeType.OKX.value}:{symbol}"
            
            if signal_type == 'BUY':
                # 1단계 진입
                base_amount_usdt = self.config.dantaro_config.base_amount / self.config.currency_conversion_rate
                order_size = self._calculate_order_size(current_price, base_amount_usdt)
                
                # OKX 주문 실행 (시뮬레이션)
                result = await self._simulate_okx_order(symbol, 'buy', order_size, current_price)
                
                if result.get('status') == 'simulated':
                    # 포지션 생성
                    unified_pos = UnifiedPosition(
                        exchange=ExchangeType.OKX,
                        symbol=symbol,
                        market_type=MarketType.FUTURES if 'SWAP' in symbol else MarketType.SPOT,
                        side='long',
                        entry_price=Decimal(str(current_price)),
                        current_price=Decimal(str(current_price)),
                        size=Decimal(str(order_size)),
                        unrealized_pnl=Decimal('0'),
                        stage=1,
                        currency='USDT'
                    )
                    self.positions[position_key] = unified_pos
                    
                    return {
                        'status': 'success',
                        'exchange': 'OKX',
                        'action': 'buy',
                        'symbol': symbol,
                        'price': current_price,
                        'size': order_size,
                        'stage': 1
                    }
            
            elif signal_type == 'ADD':
                # 추가 매수 (물타기)
                if position_key not in self.positions:
                    return {'status': 'error', 'error': '기존 포지션 없음'}
                
                position = self.positions[position_key]
                next_stage = position.stage + 1
                
                if next_stage > self.config.dantaro_config.max_stages:
                    return {'status': 'error', 'error': '최대 단계 초과'}
                
                # 추가 매수 금액 계산
                stage_amount = self.config.dantaro_config.base_amount * (
                    self.config.dantaro_config.multiplier ** (next_stage - 1)
                )
                stage_amount_usdt = stage_amount / self.config.currency_conversion_rate
                order_size = self._calculate_order_size(current_price, stage_amount_usdt)
                
                result = await self._simulate_okx_order(symbol, 'buy', order_size, current_price)
                
                if result.get('status') == 'simulated':
                    # 평균단가 계산
                    total_cost = float(position.entry_price * position.size) + (current_price * order_size)
                    total_size = float(position.size) + order_size
                    new_avg_price = total_cost / total_size
                    
                    position.entry_price = Decimal(str(new_avg_price))
                    position.size = Decimal(str(total_size))
                    position.stage = next_stage
                    
                    return {
                        'status': 'success',
                        'exchange': 'OKX',
                        'action': 'add_buy',
                        'symbol': symbol,
                        'price': current_price,
                        'size': order_size,
                        'stage': next_stage,
                        'new_avg_price': new_avg_price
                    }
            
            elif signal_type == 'SELL':
                # 매도
                if position_key not in self.positions:
                    return {'status': 'error', 'error': '매도할 포지션 없음'}
                
                position = self.positions[position_key]
                sell_size = float(position.size)
                
                result = await self._simulate_okx_order(symbol, 'sell', sell_size, current_price)
                
                if result.get('status') == 'simulated':
                    # 수익 계산
                    profit_usdt = (current_price - float(position.entry_price)) * sell_size
                    profit_krw = profit_usdt * self.config.currency_conversion_rate
                    
                    # 포지션 제거
                    del self.positions[position_key]
                    self.daily_pnl_krw += profit_krw
                    self.total_trades += 1
                    
                    return {
                        'status': 'success',
                        'exchange': 'OKX',
                        'action': 'sell',
                        'symbol': symbol,
                        'price': current_price,
                        'size': sell_size,
                        'profit_usdt': profit_usdt,
                        'profit_krw': profit_krw,
                        'profit_percentage': position.percentage_change
                    }
            
            return {'status': 'ignored', 'reason': f'알 수 없는 신호: {signal_type}'}
            
        except Exception as e:
            logger.error(f"OKX 신호 실행 실패: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _execute_upbit_signal(self, symbol: str, signal_data: Dict) -> Dict[str, Any]:
        """업비트 신호 실행"""
        try:
            if not self.upbit_client or not self.exchange_status[ExchangeType.UPBIT]['connected']:
                return {'status': 'error', 'error': '업비트 연결되지 않음'}
            
            signal_type = signal_data.get('signal')
            current_price = signal_data.get('current_price', 0)
            
            position_key = f"{ExchangeType.UPBIT.value}:{symbol}"
            
            if signal_type == 'BUY':
                # KRW 기준 매수
                base_amount_krw = self.config.dantaro_config.base_amount
                
                # 시뮬레이션 주문
                result = await self._simulate_upbit_order(symbol, 'buy', base_amount_krw, current_price)
                
                if result.get('status') == 'simulated':
                    # 매수 수량 계산
                    volume = base_amount_krw / current_price
                    
                    unified_pos = UnifiedPosition(
                        exchange=ExchangeType.UPBIT,
                        symbol=symbol,
                        market_type=MarketType.SPOT,
                        side='long',
                        entry_price=Decimal(str(current_price)),
                        current_price=Decimal(str(current_price)),
                        size=Decimal(str(volume)),
                        unrealized_pnl=Decimal('0'),
                        stage=1,
                        currency='KRW'
                    )
                    self.positions[position_key] = unified_pos
                    
                    return {
                        'status': 'success',
                        'exchange': 'UPBIT',
                        'action': 'buy',
                        'symbol': symbol,
                        'price': current_price,
                        'volume': volume,
                        'amount_krw': base_amount_krw,
                        'stage': 1
                    }
            
            elif signal_type == 'ADD':
                # 업비트 물타기
                if position_key not in self.positions:
                    return {'status': 'error', 'error': '기존 포지션 없음'}
                
                position = self.positions[position_key]
                next_stage = position.stage + 1
                
                if next_stage > self.config.dantaro_config.max_stages:
                    return {'status': 'error', 'error': '최대 단계 초과'}
                
                stage_amount_krw = self.config.dantaro_config.base_amount * (
                    self.config.dantaro_config.multiplier ** (next_stage - 1)
                )
                
                result = await self._simulate_upbit_order(symbol, 'buy', stage_amount_krw, current_price)
                
                if result.get('status') == 'simulated':
                    # 추가 매수량 계산
                    add_volume = stage_amount_krw / current_price
                    
                    # 평균단가 계산 (KRW 기준)
                    total_value = float(position.entry_price * position.size) + (current_price * add_volume)
                    total_volume = float(position.size) + add_volume
                    new_avg_price = total_value / total_volume
                    
                    position.entry_price = Decimal(str(new_avg_price))
                    position.size = Decimal(str(total_volume))
                    position.stage = next_stage
                    
                    return {
                        'status': 'success',
                        'exchange': 'UPBIT',
                        'action': 'add_buy',
                        'symbol': symbol,
                        'price': current_price,
                        'volume': add_volume,
                        'amount_krw': stage_amount_krw,
                        'stage': next_stage,
                        'new_avg_price': new_avg_price
                    }
            
            elif signal_type == 'SELL':
                # 업비트 매도
                if position_key not in self.positions:
                    return {'status': 'error', 'error': '매도할 포지션 없음'}
                
                position = self.positions[position_key]
                sell_volume = float(position.size)
                
                result = await self._simulate_upbit_order(symbol, 'sell', sell_volume, current_price)
                
                if result.get('status') == 'simulated':
                    # 수익 계산 (KRW 기준)
                    profit_krw = (current_price - float(position.entry_price)) * sell_volume
                    
                    # 포지션 제거
                    del self.positions[position_key]
                    self.daily_pnl_krw += profit_krw
                    self.total_trades += 1
                    
                    return {
                        'status': 'success',
                        'exchange': 'UPBIT',
                        'action': 'sell',
                        'symbol': symbol,
                        'price': current_price,
                        'volume': sell_volume,
                        'profit_krw': profit_krw,
                        'profit_percentage': position.percentage_change
                    }
            
            return {'status': 'ignored', 'reason': f'알 수 없는 신호: {signal_type}'}
            
        except Exception as e:
            logger.error(f"업비트 신호 실행 실패: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # =================================
    # 유틸리티 메서드
    # =================================
    
    def _calculate_order_size(self, price: float, amount_usdt: float) -> float:
        """주문 크기 계산 (USDT 기준)"""
        return round(amount_usdt / price, 8)
    
    async def _simulate_okx_order(self, symbol: str, side: str, size: float, price: float) -> Dict:
        """OKX 주문 시뮬레이션"""
        return {
            'status': 'simulated',
            'exchange': 'OKX',
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _simulate_upbit_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        """업비트 주문 시뮬레이션"""
        return {
            'status': 'simulated',
            'exchange': 'UPBIT',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            # 포지션 현재가 업데이트
            await self._update_positions_price()
            
            # 통계 계산
            total_positions = len(self.positions)
            total_unrealized_pnl_krw = sum(float(pos.unrealized_pnl) * 
                                         (1 if pos.currency == 'KRW' else self.config.currency_conversion_rate)
                                         for pos in self.positions.values())
            
            return {
                'status': 'running' if self.is_running else 'stopped',
                'total_positions': total_positions,
                'daily_pnl_krw': self.daily_pnl_krw,
                'unrealized_pnl_krw': total_unrealized_pnl_krw,
                'total_pnl_krw': self.daily_pnl_krw + total_unrealized_pnl_krw,
                'total_trades': self.total_trades,
                'exchange_status': self.exchange_status,
                'positions_by_exchange': {
                    'OKX': len([p for p in self.positions.values() if p.exchange == ExchangeType.OKX]),
                    'UPBIT': len([p for p in self.positions.values() if p.exchange == ExchangeType.UPBIT])
                },
                'positions': [
                    {
                        'exchange': pos.exchange.value,
                        'symbol': pos.symbol,
                        'market_type': pos.market_type.value,
                        'side': pos.side,
                        'size': float(pos.size),
                        'entry_price': float(pos.entry_price),
                        'current_price': float(pos.current_price),
                        'unrealized_pnl': float(pos.unrealized_pnl),
                        'percentage': pos.percentage_change,
                        'stage': pos.stage,
                        'currency': pos.currency,
                        'value_krw': float(pos.value_krw)
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
            for position in self.positions.values():
                try:
                    if position.exchange == ExchangeType.OKX:
                        # OKX 현재가 조회
                        ticker_data = await self.okx_public_client.get_ticker(position.symbol)
                        if ticker_data and 'last' in ticker_data:
                            current_price = Decimal(str(ticker_data['last']))
                            position.current_price = current_price
                            pnl = (current_price - position.entry_price) * position.size
                            position.unrealized_pnl = pnl
                    
                    elif position.exchange == ExchangeType.UPBIT:
                        # 업비트 현재가 조회
                        current_price = await self.upbit_client.get_market_price(position.symbol)
                        if current_price:
                            position.current_price = current_price
                            pnl = (current_price - position.entry_price) * position.size
                            position.unrealized_pnl = pnl
                            
                except Exception as e:
                    logger.error(f"포지션 가격 업데이트 실패 ({position.symbol}): {e}")
                    
        except Exception as e:
            logger.error(f"전체 포지션 가격 업데이트 실패: {e}")
    
    async def cleanup(self):
        """시스템 정리"""
        try:
            if self.okx_private_client:
                await self.okx_private_client.__aexit__(None, None, None)
            
            if self.upbit_client:
                await self.upbit_client.__aexit__(None, None, None)
                
            logger.info("다중 거래소 시스템 정리 완료")
            
        except Exception as e:
            logger.error(f"시스템 정리 중 에러: {e}")

# =================================
# 헬퍼 함수들
# =================================

def create_multi_exchange_system(
    okx_credentials: Optional[Dict] = None,
    upbit_credentials: Optional[Dict] = None,
    preferred_exchange: ExchangeType = ExchangeType.OKX
) -> MultiExchangeSystem:
    """다중 거래소 시스템 생성"""
    
    config = MultiExchangeConfig(preferred_exchange=preferred_exchange)
    
    if okx_credentials:
        config.okx_config = ExchangeConfig(
            exchange_type=ExchangeType.OKX,
            market_type=MarketType.FUTURES,
            credentials=okx_credentials,
            enabled=True
        )
    
    if upbit_credentials:
        config.upbit_config = ExchangeConfig(
            exchange_type=ExchangeType.UPBIT,
            market_type=MarketType.SPOT,
            credentials=upbit_credentials,
            enabled=True
        )
    
    return MultiExchangeSystem(config)

# =================================
# 테스트 및 데모 코드
# =================================

async def demo_multi_exchange():
    """다중 거래소 시스템 데모"""
    print("🚀 다중 거래소 통합 시스템 테스트")
    
    # 테스트용 설정
    okx_creds = {
        'api_key': 'test_okx_key',
        'secret_key': 'test_okx_secret',
        'passphrase': 'test_okx_pass',
        'sandbox': True
    }
    
    upbit_creds = {
        'access_key': 'test_upbit_access',
        'secret_key': 'test_upbit_secret'
    }
    
    system = create_multi_exchange_system(
        okx_credentials=okx_creds,
        upbit_credentials=upbit_creds,
        preferred_exchange=ExchangeType.OKX
    )
    
    async with system:
        # 시스템 상태 확인
        status = await system.get_system_status()
        print(f"시스템 상태: {status}")
        
        # OKX 신호 테스트
        okx_signal = {
            'signal': 'BUY',
            'current_price': 42000.0,
            'confidence': 0.85
        }
        
        okx_result = await system.execute_signal(
            ExchangeType.OKX, 
            'BTC-USDT-SWAP', 
            okx_signal
        )
        print(f"OKX 거래 결과: {okx_result}")
        
        # 업비트 신호 테스트
        upbit_signal = {
            'signal': 'BUY',
            'current_price': 55000000.0,  # 5500만원
            'confidence': 0.85
        }
        
        upbit_result = await system.execute_signal(
            ExchangeType.UPBIT,
            'KRW-BTC',
            upbit_signal
        )
        print(f"업비트 거래 결과: {upbit_result}")

if __name__ == "__main__":
    print("🌍 다중 거래소 통합 시스템")
    print("OKX (글로벌 USDT) + 업비트 (한국 KRW) 동시 지원")
    
    # asyncio.run(demo_multi_exchange())  # 테스트 실행