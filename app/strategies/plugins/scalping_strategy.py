# 파일명: app/strategies/plugins/scalping_strategy.py
"""
단타로(Scalping) 전략 구현
짧은 시간 내에 작은 수익을 노리는 고빈도 거래 전략
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

from ..core.strategy_base import StrategyBase, StrategyConfig, MarketData, TradingSignal, Position, SignalType
from ..utils.technical_analysis import TechnicalAnalysis
from ..utils.risk_management import RiskManager


@dataclass
class ScalpingConfig(StrategyConfig):
    """단타로 전략 설정"""
    
    # StrategyConfig 필수 필드들
    version: str = "1.0.0"
    description: str = "단타로 고빈도 거래 전략"
    parameters: Dict = None
    risk_level: str = "MEDIUM"
    min_balance: float = 100.0
    supported_exchanges: List[str] = None
    
    # 단타로 전략 전용 설정
    timeframe: str = "1m"  # 1분봉 기준
    max_position_size: float = 0.02  # 총 자산의 2%
    
    # 진입 조건
    rsi_oversold: float = 30.0  # RSI 과매도 기준
    rsi_overbought: float = 70.0  # RSI 과매수 기준
    volume_multiplier: float = 3.0  # 평균 거래량 대비 배수
    price_change_threshold: float = 0.02  # 2% 이상 급등/급락
    
    # 청산 조건
    target_profit: float = 0.008  # 0.8% 목표 수익률
    stop_loss: float = 0.003  # 0.3% 손절 기준
    max_holding_time: int = 30  # 최대 보유시간 (분)
    
    # 리스크 관리
    max_daily_trades: int = 50  # 일일 최대 거래 횟수
    daily_loss_limit: float = 0.05  # 일일 최대 손실 5%
    
    # 시장 조건
    min_volatility: float = 0.01  # 최소 변동성 (ATR 기준)
    min_volume: float = 100000  # 최소 거래량 (USDT)


class ScalpingStrategy(StrategyBase):
    """단타로 거래 전략"""
    
    def __init__(self, config: ScalpingConfig):
        super().__init__(config)
        self.config = config
        self.ta = TechnicalAnalysis()
        self.risk_manager = RiskManager()
        
        # 거래 추적
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self.positions: Dict[str, Position] = {}
        
        # 기술적 지표 저장
        self.indicators: Dict[str, Dict] = {}
        
        print(f"✅ 단타로 전략 초기화 완료")
        print(f"📊 설정: 목표수익 {config.target_profit*100:.1f}%, 손절 {config.stop_loss*100:.1f}%")
    
    def analyze_market(self, market_data: MarketData) -> Dict:
        """시장 분석 수행"""
        symbol = market_data.symbol
        
        # 단일 데이터 포인트 기반
        prices = np.array([market_data.close])
        volumes = np.array([market_data.volume])
        high_prices = np.array([market_data.high])
        low_prices = np.array([market_data.low])
        
        if len(prices) < 1:
            return {'signal': 'HOLD', 'reason': '데이터 부족'}
        
        # 기술적 지표 계산
        current_price = prices[-1]
        
        # RSI 계산 (단일 값이므로 0)
        rsi = 0.0
        
        # 이동평균선 (단일 값)
        sma_5 = current_price
        sma_20 = current_price
        
        # 볼린저 밴드 (단일 값)
        bb_upper = current_price
        bb_middle = current_price
        bb_lower = current_price
        
        # ATR (단일 값)
        atr = 0.0
        
        # 거래량 분석
        avg_volume = volumes[-1]
        current_volume = volumes[-1]
        volume_ratio = 1.0
        
        # 가격 변화율 (단일 값이므로 0)
        price_change = 0.0
        
        # 지표 저장
        self.indicators[symbol] = {
            'rsi': rsi,
            'sma_5': sma_5,
            'sma_20': sma_20,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'atr': atr,
            'volume_ratio': volume_ratio,
            'price_change': price_change,
            'current_price': current_price
        }
        
        return self.indicators[symbol]
    
    def generate_signal(self, market_data: MarketData) -> TradingSignal:
        """거래 시그널 생성"""
        
        # 강제 테스트 - 맨 앞에 추가
        import random
        rand = random.random()
        if rand < 0.5:  # 50% 확률
            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=SignalType.BUY,
                price=market_data.close,
                quantity=0.0,
                confidence=0.9,
                timestamp=datetime.now(),
                metadata={"reason": "강제 테스트"}
            )
        
        # 일일 거래 제한 확인
        if self.daily_trades >= self.config.max_daily_trades:
            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=SignalType.HOLD,
                price=market_data.close,
                quantity=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"reason": "일일 거래 한도 초과"}
            )
        
        # 일일 손실 제한 확인
        if self.daily_pnl < -self.config.daily_loss_limit:
            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=SignalType.HOLD,
                price=market_data.close,
                quantity=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"reason": "일일 손실 한도 초과"}
            )
        
        # 시장 분석
        analysis = self.analyze_market(market_data)
        
        if 'signal' in analysis and analysis['signal'] == 'HOLD':
            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=SignalType.HOLD,
                price=market_data.close,
                quantity=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"reason": analysis.get('reason', '분석 불가')}
            )
        
        symbol = market_data.symbol
        indicators = analysis
        
        # 시장 조건 확인
        if indicators['atr'] < self.config.min_volatility:
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=market_data.close,
                quantity=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"reason": "변동성 부족"}
            )
        
        # 현재 포지션 확인
        current_position = self.positions.get(symbol)
        
        if current_position:
            # 기존 포지션이 있는 경우 청산 조건 확인
            return self._check_exit_conditions(current_position, indicators)
        else:
            # 새로운 진입 조건 확인
            return self._check_entry_conditions(symbol, indicators)
    
    def _check_entry_conditions(self, symbol: str, indicators: Dict) -> TradingSignal:
        """진입 조건 확인 - 임시로 간단화"""
        
        rsi = indicators['rsi']
        current_price = indicators['current_price']
        
        # 임시로 매우 간단한 조건
        import random
        
        # 30% 확률로 매수, 30% 확률로 매도, 40% 확률로 홀드
        rand = random.random()
        
        if rand < 0.3:  # 매수
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                quantity=0.0,
                confidence=0.8,
                timestamp=datetime.now(),
                metadata={"reason": "테스트 매수"}
            )
        elif rand < 0.6:  # 매도
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                quantity=0.0,
                confidence=0.8,
                timestamp=datetime.now(),
                metadata={"reason": "테스트 매도"}
            )
        else:  # 홀드
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=current_price,
                quantity=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"reason": "테스트 홀드"}
            )
    
    def _check_exit_conditions(self, position: Position, indicators: Dict) -> TradingSignal:
        """청산 조건 확인"""
        
        current_price = indicators['current_price']
        entry_price = position.entry_price
        
        # 수익률 계산
        if position.side == "LONG":
            pnl_ratio = (current_price - entry_price) / entry_price
        else:  # SHORT
            pnl_ratio = (entry_price - current_price) / entry_price
        
        # 보유 시간 확인
        holding_time = datetime.now() - position.entry_time
        holding_minutes = holding_time.total_seconds() / 60
        
        # 청산 조건들
        take_profit = pnl_ratio >= self.config.target_profit
        stop_loss = pnl_ratio <= -self.config.stop_loss
        max_time = holding_minutes >= self.config.max_holding_time
        
        # RSI 반전 조건
        rsi = indicators['rsi']
        rsi_reversal = False
        if position.side == "LONG" and rsi > self.config.rsi_overbought:
            rsi_reversal = True
        elif position.side == "SHORT" and rsi < self.config.rsi_oversold:
            rsi_reversal = True
        
        # 청산 결정
        if take_profit:
            return TradingSignal(
                symbol=position.symbol,
                signal_type=SignalType.CLOSE,
                price=indicators['current_price'],
                quantity=position.size,
                confidence=1.0,
                timestamp=datetime.now(),
                metadata={"reason": f"목표수익 달성 ({pnl_ratio*100:.2f}%)"}
            )
        elif stop_loss:
            return TradingSignal(
                symbol=position.symbol,
                signal_type=SignalType.CLOSE,
                price=indicators['current_price'],
                quantity=position.size,
                confidence=1.0,
                timestamp=datetime.now(),
                metadata={"reason": f"손절 실행 ({pnl_ratio*100:.2f}%)"}
            )
        elif max_time:
            return TradingSignal(
                symbol=position.symbol,
                signal_type=SignalType.CLOSE,
                price=indicators['current_price'],
                quantity=position.size,
                confidence=0.8,
                timestamp=datetime.now(),
                metadata={"reason": f"최대보유시간 도달 ({holding_minutes:.0f}분)"}
            )
        elif rsi_reversal:
            return TradingSignal(
                symbol=position.symbol,
                signal_type=SignalType.CLOSE,
                price=indicators['current_price'],
                quantity=position.size,
                confidence=0.7,
                timestamp=datetime.now(),
                metadata={"reason": f"RSI 반전 신호 (RSI:{rsi:.1f})"}
            )
        return TradingSignal(
            symbol=position.symbol,
            signal_type=SignalType.HOLD,
            price=indicators['current_price'],
            quantity=position.size,
            confidence=0.0,
            timestamp=datetime.now(),
            metadata={"reason": f"보유중 (수익률:{pnl_ratio*100:.2f}%, {holding_minutes:.0f}분)"}
        )
    
    def execute_trade(self, signal: TradingSignal, market_data: MarketData) -> bool:
        """거래 실행"""
        
        if signal.signal_type == SignalType.HOLD:
            return True
        
        symbol = signal.symbol
        current_price = float(market_data.close)
        
        try:
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                # 새로운 포지션 생성
                position_size = self._calculate_position_size(current_price, signal.confidence)
                
                position = Position(
                    symbol=symbol,
                    side="LONG" if signal.signal_type == SignalType.BUY else "SHORT",
                    size=position_size,
                    entry_price=current_price,
                    entry_time=datetime.now()
                )
                
                self.positions[symbol] = position
                self.daily_trades += 1
                
                print(f"🔵 {signal.signal_type.value} 포지션 생성: {symbol} @{current_price} (신뢰도:{signal.confidence:.2f})")
                print(f"📝 사유: {signal.metadata.get('reason', '') if signal.metadata else ''}")
                
            elif signal.signal_type == SignalType.CLOSE:
                # 포지션 청산
                if symbol in self.positions:
                    position = self.positions[symbol]
                    
                    # 수익률 계산
                    if position.side == "LONG":
                        pnl_ratio = (current_price - position.entry_price) / position.entry_price
                    else:
                        pnl_ratio = (position.entry_price - current_price) / position.entry_price
                    
                    pnl_amount = position.size * pnl_ratio
                    self.daily_pnl += pnl_amount
                    
                    print(f"🔴 포지션 청산: {symbol} @{current_price}")
                    print(f"📊 수익률: {pnl_ratio*100:.2f}% (${pnl_amount:.2f})")
                    print(f"📝 사유: {signal.metadata.get('reason', '') if signal.metadata else ''}")
                    
                    # 포지션 제거
                    del self.positions[symbol]
                    self.daily_trades += 1
            
            return True
            
        except Exception as e:
            print(f"❌ 거래 실행 실패: {str(e)}")
            return False
    
    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """포지션 크기 계산"""
        
        # 기본 포지션 크기 (총 자산의 2%)
        base_size = self.config.max_position_size * 10000  # 가상의 계좌 잔고 $10,000
        
        # 신뢰도에 따른 조정 (50% ~ 100%)
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        # 최종 포지션 크기 (달러 기준)
        position_size = base_size * confidence_multiplier
        
        return position_size
    
    def _calculate_atr(self, high_prices: np.ndarray, low_prices: np.ndarray, 
                      close_prices: np.ndarray, period: int = 14) -> float:
        """ATR (Average True Range) 계산"""
        
        if len(high_prices) < period + 1:
            return 0.0
        
        # True Range 계산
        tr_list = []
        for i in range(1, len(high_prices)):
            high_low = high_prices[i] - low_prices[i]
            high_close_prev = abs(high_prices[i] - close_prices[i-1])
            low_close_prev = abs(low_prices[i] - close_prices[i-1])
            
            tr = max(high_low, high_close_prev, low_close_prev)
            tr_list.append(tr)
        
        # ATR 계산 (단순 이동평균)
        if len(tr_list) >= period:
            atr = np.mean(tr_list[-period:])
            return atr / close_prices[-1]  # 가격 대비 비율로 정규화
        
        return 0.0
    
    def get_performance_metrics(self) -> Dict:
        """성과 지표 조회"""
        
        total_positions = len([p for p in self.positions.values()])
        
        return {
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'active_positions': total_positions,
            'pnl_percentage': (self.daily_pnl / 10000) * 100,  # 가상 계좌 기준
            'avg_pnl_per_trade': self.daily_pnl / max(self.daily_trades, 1),
            'strategy_name': 'Scalping Strategy',
            'status': 'ACTIVE' if total_positions > 0 else 'IDLE'
        }
    
    def reset_daily_metrics(self):
        """일일 지표 초기화"""
        self.daily_trades = 0
        self.daily_pnl = 0.0
        print("\ud83d\udcca \uc77c\uc77c \uc9c0\ud45c \ucd08\uae30\ud654 \uc644\ub8cc")

    # ===== StrategyBase 추상 메서드 구현 =====
    
    def analyze(self, market_data: MarketData) -> Dict:
        """시장 분석 (추상 메서드 구현)"""
        return self.analyze_market(market_data)
    
    def get_position_size(self, signal_confidence: float, account_balance: float) -> float:
        """포지션 크기 계산 (추상 메서드 구현)"""
        return self._calculate_position_size(account_balance, signal_confidence)
    
    def get_required_parameters(self) -> List[str]:
        """필수 매개변수 목록 (추상 메서드 구현)"""
        return [
            'target_profit', 'stop_loss', 'max_daily_trades',
            'rsi_oversold', 'rsi_overbought', 'volume_multiplier',
            'price_change_threshold', 'max_holding_time'
        ]
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """시그널 검증 (추상 메서드 구현)"""
        # 기본 검증
        if not signal or not signal.symbol:
            return False
        
        if signal.signal_type not in [SignalType.BUY, SignalType.SELL, SignalType.HOLD, SignalType.CLOSE]:
            return False
        
        if not (0 <= signal.confidence <= 1):
            return False
        
        # 단타로 전략 특화 검증
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            # 신뢰도가 0.6 이상이어야 함
            if signal.confidence < 0.6:
                return False
        
        return True


# 사용 예시 및 테스트 함수
def create_scalping_strategy() -> ScalpingStrategy:
    """단타로 전략 인스턴스 생성"""
    
    config = ScalpingConfig(
        name="Scalping_Strategy_v1",
        timeframe="1m",
        target_profit=0.008,  # 0.8%
        stop_loss=0.003,      # 0.3%
        max_daily_trades=30,
        rsi_oversold=25,      # 더 보수적
        rsi_overbought=75,    # 더 보수적
        supported_exchanges=["OKX", "UPBIT"]
    )
    
    return ScalpingStrategy(config)


if __name__ == "__main__":
    # 전략 테스트
    strategy = create_scalping_strategy()
    print("🚀 단타로 전략 테스트 시작")
    print(f"📊 성과 지표: {strategy.get_performance_metrics()}")