# 파일명: app/strategies/core/backtest_engine.py
"""
백테스트 엔진
과거 데이터를 사용하여 거래 전략의 성능을 시뮬레이션하고 분석합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum

from .strategy_base import StrategyBase, MarketData, TradingSignal, Position, SignalType
from ..utils.risk_management import RiskManager

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """주문 유형"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class BacktestConfig:
    """백테스트 설정"""
    start_date: str
    end_date: str
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1% 수수료
    slippage_rate: float = 0.0001   # 0.01% 슬리피지
    max_positions: int = 10
    enable_margin: bool = False
    margin_ratio: float = 1.0
    data_frequency: str = "1m"      # 1분봉


@dataclass
class Trade:
    """거래 기록"""
    trade_id: str
    symbol: str
    side: str  # LONG, SHORT
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    commission: float
    slippage: float
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    holding_time: Optional[timedelta] = None
    entry_reason: str = ""
    exit_reason: str = ""


@dataclass
class BacktestResults:
    """백테스트 결과"""
    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    
    # 성과 지표
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_return: float = 0.0
    daily_return: float = 0.0
    
    # 리스크 지표
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    
    # 거래 통계
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # 시간 통계
    avg_holding_time: Optional[timedelta] = None
    max_holding_time: Optional[timedelta] = None
    min_holding_time: Optional[timedelta] = None


class BacktestEngine:
    """백테스트 엔진"""
    
    def __init__(self, strategy: StrategyBase, config: BacktestConfig):
        self.strategy = strategy
        self.config = config
        self.risk_manager = RiskManager()
        
        # 백테스트 상태
        self.current_balance = config.initial_balance
        self.current_positions: Dict[str, Position] = {}
        self.open_trades: Dict[str, Trade] = {}
        self.completed_trades: List[Trade] = []
        self.equity_curve: List[float] = [config.initial_balance]
        self.timestamps: List[datetime] = []
        
        # 통계
        self.trade_counter = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        # 과거 데이터 저장용 추가
        self.historical_data: List[Dict] = []
        self.max_history = 200  # 최대 200개 봉 저장
        
        logger.info(f"백테스트 엔진 초기화: {config.start_date} ~ {config.end_date}")
    
    def run(self, market_data: List[Dict]) -> BacktestResults:
        """백테스트 실행"""
        
        logger.info("백테스트 시작")
        
        try:
            # 데이터 검증
            if not market_data:
                raise ValueError("시장 데이터가 비어있습니다")
            
            # 시뮬레이션 실행
            for i, data_point in enumerate(market_data):
                self._process_data_point(data_point, i)
                
                # 진행상황 로깅 (1000개마다)
                if i > 0 and i % 1000 == 0:
                    logger.info(f"진행률: {i}/{len(market_data)} ({i/len(market_data)*100:.1f}%)")
            
            # 남은 포지션 청산
            self._close_all_positions(market_data[-1])
            
            # 결과 분석
            results = self._analyze_results()
            
            logger.info(f"백테스트 완료: 총 {len(self.completed_trades)}건 거래")
            logger.info(f"최종 수익률: {results.total_return:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"백테스트 실행 중 오류: {str(e)}")
            raise
    
    def _process_data_point(self, data_point: Dict, index: int):
        """단일 데이터 포인트 처리"""
        
        # 현재 시간과 가격
        timestamp = datetime.fromisoformat(data_point['timestamp'].replace('Z', '+00:00'))
        current_price = float(data_point['close'])
        
        # 과거 데이터에 현재 데이터 추가
        self.historical_data.append(data_point)
        if len(self.historical_data) > self.max_history:
            self.historical_data.pop(0)  # 오래된 데이터 제거
        
        # MarketData 객체 생성 (현재 데이터)
        market_data = MarketData(
            symbol=data_point.get('symbol', 'BTC/USDT'),
            timestamp=timestamp,
            open=float(data_point['open']),
            high=float(data_point['high']),
            low=float(data_point['low']),
            close=float(data_point['close']),
            volume=float(data_point['volume'])
        )
        
        # 과거 데이터를 포함한 분석용 데이터 생성
        analysis_data = MarketData(
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            open=market_data.open,
            high=market_data.high,
            low=market_data.low,
            close=market_data.close,
            volume=market_data.volume
        )
        
        # historical_data를 ohlcv 형태로 추가 (strategy에서 사용)
        analysis_data.ohlcv = self.historical_data.copy()
        
        # 기존 포지션 업데이트
        self._update_positions(current_price, timestamp)
        
        # 전략 시그널 생성 (과거 데이터 포함)
        try:
            signal = self.strategy.generate_signal(analysis_data)
            
            # 시그널 처리
            if signal and signal.signal_type != SignalType.HOLD:
                self._process_signal(signal, current_price, timestamp)
                
        except Exception as e:
            logger.warning(f"시그널 생성 오류 ({timestamp}): {str(e)}")
        
        # 포트폴리오 가치 업데이트
        portfolio_value = self._calculate_portfolio_value(current_price)
        self.equity_curve.append(portfolio_value)
        self.timestamps.append(timestamp)
    
    def _update_positions(self, current_price: float, timestamp: datetime):
        """기존 포지션 업데이트 및 청산 조건 확인"""
        
        positions_to_close = []
        
        for symbol, position in self.current_positions.items():
            # 현재 가격 업데이트
            position.current_price = current_price
            
            # P&L 계산
            if position.side == "LONG":
                position.unrealized_pnl = (current_price - position.entry_price) * position.size
            else:  # SHORT
                position.unrealized_pnl = (position.entry_price - current_price) * position.size
            
            # 청산 조건 확인 (시간 제한, 손절/익절 등)
            if self._should_close_position(position, timestamp):
                positions_to_close.append(symbol)
        
        # 청산 실행
        for symbol in positions_to_close:
            self._close_position(symbol, current_price, timestamp, "AUTO_CLOSE")
    
    def _should_close_position(self, position: Position, current_time: datetime) -> bool:
        """포지션 청산 조건 확인"""
        
        if not position.current_price:
            return False
        
        # 시간 기반 청산 (단타로 전략: 최대 30분)
        holding_time = current_time - position.entry_time
        if holding_time.total_seconds() > 1800:  # 30분
            return True
        
        # 손익 기반 청산
        pnl_percent = position.unrealized_pnl / (position.entry_price * position.size) if position.entry_price > 0 else 0
        
        # 익절 (0.8%)
        if pnl_percent > 0.008:
            return True
        
        # 손절 (-0.3%)
        if pnl_percent < -0.003:
            return True
        
        return False
    
    def _process_signal(self, signal: TradingSignal, current_price: float, timestamp: datetime):
        """거래 시그널 처리"""
        
        symbol = signal.symbol
        
        if signal.signal_type == SignalType.CLOSE:
            # 기존 포지션 청산
            if symbol in self.current_positions:
                self._close_position(symbol, current_price, timestamp, "CLOSE_SIGNAL")
        
        elif signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            # 새 포지션 오픈
            if len(self.current_positions) < self.config.max_positions:
                self._open_position(signal, current_price, timestamp)
            else:
                logger.debug(f"최대 포지션 수 초과로 신호 무시: {signal.signal_type.value} {symbol}")
    
    def _open_position(self, signal: TradingSignal, current_price: float, timestamp: datetime):
        """새 포지션 오픈"""
        
        symbol = signal.symbol
        side = "LONG" if signal.signal_type == SignalType.BUY else "SHORT"
        
        # 포지션 크기 계산
        position_size = self.strategy.get_position_size(signal.confidence, self.current_balance)
        
        # 수수료 및 슬리피지 계산
        commission = position_size * self.config.commission_rate
        slippage = position_size * self.config.slippage_rate
        
        # 실제 진입 가격 (슬리피지 반영)
        if side == "LONG":
            entry_price = current_price * (1 + self.config.slippage_rate)
        else:
            entry_price = current_price * (1 - self.config.slippage_rate)
        
        # 잔고 확인
        required_balance = position_size + commission
        if required_balance > self.current_balance:
            logger.debug(f"잔고 부족으로 포지션 오픈 실패: {symbol}")
            return
        
        # 포지션 생성
        position = Position(
            symbol=symbol,
            side=side,
            size=position_size / entry_price,  # 수량 계산
            entry_price=entry_price,
            entry_time=timestamp
        )
        
        # 거래 기록 생성
        trade_id = f"T{self.trade_counter:06d}"
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_time=timestamp,
            exit_time=None,  # 추가
            entry_price=entry_price,
            exit_price=None,  # 추가
            quantity=position.size,
            commission=commission,
            slippage=slippage,
            entry_reason=signal.metadata.get('reason', 'Unknown') if signal.metadata else 'Unknown'
        )
        
        
        # 상태 업데이트
        self.current_positions[symbol] = position
        self.open_trades[symbol] = trade
        self.current_balance -= required_balance
        self.total_commission += commission
        self.total_slippage += slippage
        self.trade_counter += 1
        
        logger.debug(f"포지션 오픈: {side} {symbol} @{entry_price:.4f} (크기: ${position_size:.2f})")
    
    def _close_position(self, symbol: str, current_price: float, timestamp: datetime, reason: str):
        """포지션 청산"""
        
        if symbol not in self.current_positions:
            return
        
        position = self.current_positions[symbol]
        trade = self.open_trades[symbol]
        
        # 청산 가격 (슬리피지 반영)
        if position.side == "LONG":
            exit_price = current_price * (1 - self.config.slippage_rate)
        else:
            exit_price = current_price * (1 + self.config.slippage_rate)
        
        # P&L 계산
        if position.side == "LONG":
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        
        # 수수료 차감
        exit_commission = position.size * exit_price * self.config.commission_rate
        pnl -= (trade.commission + exit_commission)
        
        # 거래 기록 완성
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_percent = (pnl / (position.entry_price * position.size)) * 100
        trade.holding_time = timestamp - position.entry_time
        trade.exit_reason = reason
        trade.commission += exit_commission
        
        # 잔고 업데이트
        self.current_balance += (position.size * exit_price) + pnl
        self.total_commission += exit_commission
        
        # 기록 이동
        self.completed_trades.append(trade)
        del self.current_positions[symbol]
        del self.open_trades[symbol]
        
        logger.debug(f"포지션 청산: {position.side} {symbol} @{exit_price:.4f} "
                    f"P&L: ${pnl:.2f} ({trade.pnl_percent:.2f}%)")
    
    def _close_all_positions(self, last_data_point: Dict):
        """모든 남은 포지션 청산"""
        
        if not self.current_positions:
            return
        
        current_price = float(last_data_point['close'])
        timestamp = datetime.fromisoformat(last_data_point['timestamp'].replace('Z', '+00:00'))
        
        symbols_to_close = list(self.current_positions.keys())
        for symbol in symbols_to_close:
            self._close_position(symbol, current_price, timestamp, "BACKTEST_END")
        
        logger.info(f"백테스트 종료시 {len(symbols_to_close)}개 포지션 청산")
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """현재 포트폴리오 가치 계산"""
        
        total_value = self.current_balance
        
        # 오픈 포지션 가치 추가
        for position in self.current_positions.values():
            if position.side == "LONG":
                unrealized_pnl = (current_price - position.entry_price) * position.size
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.size
            
            total_value += unrealized_pnl
        
        return total_value
    
    def _analyze_results(self) -> BacktestResults:
        """백테스트 결과 분석"""
        
        results = BacktestResults(config=self.config)
        
        # 기본 정보
        results.trades = self.completed_trades
        results.equity_curve = self.equity_curve
        results.timestamps = self.timestamps
        
        if not self.completed_trades:
            logger.warning("완료된 거래가 없습니다")
            return results
        
        # 수익률 계산
        initial_balance = self.config.initial_balance
        final_balance = self.equity_curve[-1]
        results.total_return = ((final_balance - initial_balance) / initial_balance) * 100
        
        # 기간 계산
        start_date = datetime.fromisoformat(self.config.start_date)
        end_date = datetime.fromisoformat(self.config.end_date)
        total_days = (end_date - start_date).days
        
        if total_days > 0:
            results.annual_return = results.total_return * (365 / total_days)
            results.monthly_return = results.total_return * (30 / total_days)
            results.daily_return = results.total_return / total_days
        
        # 거래 통계
        results.total_trades = len(self.completed_trades)
        winning_trades = [t for t in self.completed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.completed_trades if t.pnl and t.pnl < 0]
        
        results.winning_trades = len(winning_trades)
        results.losing_trades = len(losing_trades)
        results.win_rate = (results.winning_trades / results.total_trades) * 100 if results.total_trades > 0 else 0
        
        if winning_trades:
            results.avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
        
        if losing_trades:
            results.avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
            results.profit_factor = abs(results.avg_win / results.avg_loss) if results.avg_loss != 0 else 0
        
        # 시간 통계
        holding_times = [t.holding_time for t in self.completed_trades if t.holding_time]
        if holding_times:
            results.avg_holding_time = sum(holding_times, timedelta()) / len(holding_times)
            results.max_holding_time = max(holding_times)
            results.min_holding_time = min(holding_times)
        
        # 리스크 지표 계산
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            
            # 변동성
            results.volatility = np.std(returns) * np.sqrt(252) * 100  # 연환산
            
            # 샤프 비율 (무위험 수익률 2% 가정)
            if results.volatility > 0:
                excess_return = results.annual_return - 2.0
                results.sharpe_ratio = excess_return / results.volatility
            
            # 최대 낙폭
            peak = self.equity_curve[0]
            max_dd = 0
            dd_duration = 0
            max_dd_duration = 0
            
            for value in self.equity_curve:
                if value > peak:
                    peak = value
                    dd_duration = 0
                else:
                    dd_duration += 1
                    max_dd_duration = max(max_dd_duration, dd_duration)
                
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
            
            results.max_drawdown = max_dd * 100
            results.max_drawdown_duration = max_dd_duration
            
            # 칼마 비율
            if results.max_drawdown > 0:
                results.calmar_ratio = results.annual_return / results.max_drawdown
        
        return results


def create_sample_market_data(symbol: str = "BTC/USDT", days: int = 30) -> List[Dict]:
    """샘플 시장 데이터 생성 (테스트용) - 더 현실적인 변동성"""
    
    data = []
    base_price = 50000.0
    current_price = base_price
    
    start_time = datetime.now() - timedelta(days=days)
    
    for i in range(days * 24 * 60):  # 1분봉
        timestamp = start_time + timedelta(minutes=i)
        
        # 더 큰 변동성 (±2%)
        change = np.random.normal(0, 0.02)
        current_price *= (1 + change)
        
        # 더 현실적인 OHLC
        volatility = abs(np.random.normal(0, 0.01))
        high = current_price * (1 + volatility)
        low = current_price * (1 - volatility)
        
        # 가끔 급등/급락 (5% 확률로 ±5% 변동)
        if np.random.random() < 0.05:
            spike = np.random.choice([-0.05, 0.05])
            current_price *= (1 + spike)
            if spike > 0:
                high = max(high, current_price)
            else:
                low = min(low, current_price)
        
        # 더 다양한 거래량 (1배~10배)
        base_volume = 2000000
        volume_multiplier = np.random.uniform(1, 10)
        volume = base_volume * volume_multiplier
        
        data.append({
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'open': current_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
    
    return data


# 사용 예시
if __name__ == "__main__":
    from ..plugins.scalping_strategy import create_scalping_strategy
    
    # 전략 생성
    strategy = create_scalping_strategy()
    
    # 백테스트 설정
    config = BacktestConfig(
        start_date="2024-01-01T00:00:00",
        end_date="2024-01-31T23:59:59",
        initial_balance=10000,
        commission_rate=0.001
    )
    
    # 백테스트 엔진 생성
    engine = BacktestEngine(strategy, config)
    
    # 샘플 데이터 생성
    market_data = create_sample_market_data(days=30)
    
    # 백테스트 실행
    results = engine.run(market_data)
    
    # 결과 출력
    print(f"🎯 백테스트 결과 요약:")
    print(f"총 수익률: {results.total_return:.2f}%")
    print(f"연 수익률: {results.annual_return:.2f}%")
    print(f"샤프 비율: {results.sharpe_ratio:.2f}")
    print(f"최대 낙폭: {results.max_drawdown:.2f}%")
    print(f"승률: {results.win_rate:.1f}%")
    print(f"총 거래: {results.total_trades}건")