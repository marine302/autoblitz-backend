# 파일: app/strategies/plugins/simple_momentum.py
# 경로: /workspaces/autoblitz-backend/app/strategies/plugins/simple_momentum.py

"""
단순 모멘텀 전략 (샘플 구현)

단타로 전략의 기본 구조를 보여주는 샘플 전략입니다.
RSI와 이동평균을 이용한 간단한 모멘텀 전략을 구현합니다.
"""

from typing import List, Optional
from datetime import datetime
import logging

from app.strategies.core.strategy_base import (
    StrategyBase, 
    StrategyConfig, 
    MarketData, 
    TradingSignal, 
    SignalType
)
from app.strategies.utils.strategy_utils import (
    TechnicalAnalysis,
    RiskManagement,
    MarketAnalysis,
    calculate_all_indicators
)

logger = logging.getLogger(__name__)


class SimpleMomentumStrategy(StrategyBase):
    """
    단순 모멘텀 전략
    
    기술적 지표를 이용한 기본적인 단타 전략입니다.
    - RSI를 이용한 과매수/과매도 감지
    - 이동평균을 이용한 트렌드 확인
    - 볼린저 밴드를 이용한 진입/청산 신호
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # 전략별 데이터 저장
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.signal_history: List[TradingSignal] = []
        
        # 기본 매개변수 설정
        self.default_params = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'sma_short': 10,
            'sma_long': 20,
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'risk_percentage': 0.02,  # 2% 리스크
            'max_position_size': 0.1,  # 최대 10% 포지션
            'min_confidence': 0.6,     # 최소 신뢰도 60%
            'stop_loss_percentage': 0.03,  # 3% 손절
            'take_profit_percentage': 0.06  # 6% 익절
        }
        
        # 매개변수 업데이트
        for key, value in self.default_params.items():
            if key not in self.config.parameters:
                self.config.parameters[key] = value
        
        self.logger.info(f"단순 모멘텀 전략 초기화 완료: {self.config.parameters}")
    
    def analyze(self, market_data: MarketData) -> Optional[TradingSignal]:
        """
        시장 데이터 분석 및 거래 신호 생성
        
        Args:
            market_data: 현재 시장 데이터
            
        Returns:
            TradingSignal: 생성된 거래 신호
        """
        try:
            # 가격 및 거래량 히스토리 업데이트
            self.price_history.append(market_data.close)
            self.volume_history.append(market_data.volume)
            
            # 히스토리 크기 제한 (최근 100개만 유지)
            if len(self.price_history) > 100:
                self.price_history = self.price_history[-100:]
                self.volume_history = self.volume_history[-100:]
            
            # 충분한 데이터가 없으면 신호 없음
            params = self.config.parameters
            min_data_points = max(
                params['rsi_period'],
                params['sma_long'],
                params['bollinger_period']
            )
            
            if len(self.price_history) < min_data_points:
                return None
            
            # 기술적 지표 계산
            indicators = calculate_all_indicators(
                self.price_history, 
                self.volume_history
            )
            
            # 신호 생성
            signal = self._generate_signal(market_data, indicators)
            
            if signal:
                self.signal_history.append(signal)
                # 신호 히스토리 크기 제한
                if len(self.signal_history) > 50:
                    self.signal_history = self.signal_history[-50:]
            
            return signal
            
        except Exception as e:
            self.logger.error(f"시장 데이터 분석 중 오류: {e}")
            return None
    
    def _generate_signal(self, market_data: MarketData, indicators) -> Optional[TradingSignal]:
        """
        기술적 지표를 바탕으로 거래 신호 생성
        
        Args:
            market_data: 시장 데이터
            indicators: 계산된 기술적 지표
            
        Returns:
            TradingSignal: 거래 신호
        """
        params = self.config.parameters
        current_price = market_data.close
        
        # 기본 신호 타입과 신뢰도
        signal_type = SignalType.HOLD
        confidence = 0.0
        
        # RSI 기반 과매수/과매도 확인
        rsi_signal = None
        if indicators.rsi is not None:
            if indicators.rsi < params['rsi_oversold']:
                rsi_signal = SignalType.BUY
            elif indicators.rsi > params['rsi_overbought']:
                rsi_signal = SignalType.SELL
        
        # 이동평균 크로스오버 확인
        ma_signal = None
        if indicators.sma is not None and len(self.price_history) >= params['sma_long']:
            sma_short = TechnicalAnalysis.simple_moving_average(
                self.price_history, params['sma_short']
            )
            sma_long = TechnicalAnalysis.simple_moving_average(
                self.price_history, params['sma_long']
            )
            
            if sma_short and sma_long:
                if sma_short > sma_long and current_price > sma_short:
                    ma_signal = SignalType.BUY
                elif sma_short < sma_long and current_price < sma_short:
                    ma_signal = SignalType.SELL
        
        # 볼린저 밴드 확인
        bb_signal = None
        if (indicators.bollinger_upper is not None and 
            indicators.bollinger_lower is not None):
            
            if current_price <= indicators.bollinger_lower:
                bb_signal = SignalType.BUY  # 하단 터치시 매수
            elif current_price >= indicators.bollinger_upper:
                bb_signal = SignalType.SELL  # 상단 터치시 매도
        
        # 트렌드 확인
        trend = MarketAnalysis.detect_trend(self.price_history, 20)
        
        # 신호 조합 및 신뢰도 계산
        buy_signals = sum([
            1 for signal in [rsi_signal, ma_signal, bb_signal] 
            if signal == SignalType.BUY
        ])
        
        sell_signals = sum([
            1 for signal in [rsi_signal, ma_signal, bb_signal] 
            if signal == SignalType.SELL
        ])
        
        # 매수 신호
        if buy_signals >= 2:
            signal_type = SignalType.BUY
            confidence = 0.5 + (buy_signals * 0.2)
            
            # 상승 트렌드에서 신뢰도 증가
            if trend == 'uptrend':
                confidence += 0.1
                
        # 매도 신호
        elif sell_signals >= 2:
            signal_type = SignalType.SELL
            confidence = 0.5 + (sell_signals * 0.2)
            
            # 하락 트렌드에서 신뢰도 증가
            if trend == 'downtrend':
                confidence += 0.1
        
        # 최소 신뢰도 확인
        if confidence < params['min_confidence']:
            return None
        
        # 신뢰도 상한 설정
        confidence = min(confidence, 1.0)
        
        # 매수/매도 신호만 반환 (HOLD 제외)
        if signal_type in [SignalType.BUY, SignalType.SELL]:
            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=signal_type,
                price=current_price,
                quantity=0,  # get_position_size에서 계산
                confidence=confidence,
                timestamp=market_data.timestamp,
                metadata={
                    'rsi': indicators.rsi,
                    'rsi_signal': rsi_signal.value if rsi_signal else None,
                    'ma_signal': ma_signal.value if ma_signal else None,
                    'bb_signal': bb_signal.value if bb_signal else None,
                    'trend': trend,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'bollinger_upper': indicators.bollinger_upper,
                    'bollinger_lower': indicators.bollinger_lower
                }
            )
        
        return None
    
    def get_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """
        포지션 크기 계산
        
        Args:
            signal: 거래 신호
            account_balance: 계좌 잔고
            
        Returns:
            float: 포지션 크기
        """
        try:
            params = self.config.parameters
            
            # 기본 위험 기반 포지션 크기 계산
            entry_price = signal.price
            stop_loss_percentage = params['stop_loss_percentage']
            
            if signal.signal_type == SignalType.BUY:
                stop_loss_price = entry_price * (1 - stop_loss_percentage)
            else:  # SELL
                stop_loss_price = entry_price * (1 + stop_loss_percentage)
            
            # 위험 기반 포지션 크기
            position_size = RiskManagement.calculate_position_size(
                account_balance=account_balance,
                risk_percentage=params['risk_percentage'],
                entry_price=entry_price,
                stop_loss_price=stop_loss_price
            )
            
            # 신뢰도에 따른 조정
            confidence_multiplier = signal.confidence
            position_size *= confidence_multiplier
            
            # 최대 포지션 크기 제한
            max_position_value = account_balance * params['max_position_size']
            max_position_size = max_position_value / entry_price
            
            position_size = min(position_size, max_position_size)
            
            # 최소 포지션 크기 (거래소 최소 주문 금액 고려)
            min_order_value = 10  # $10 최소 주문
            min_position_size = min_order_value / entry_price
            
            if position_size < min_position_size:
                return 0
            
            self.logger.debug(
                f"포지션 크기 계산: {position_size:.6f} "
                f"(신뢰도: {signal.confidence:.2f}, 가격: {entry_price})"
            )
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 중 오류: {e}")
            return 0
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        거래 신호 유효성 검증
        
        Args:
            signal: 검증할 거래 신호
            
        Returns:
            bool: 유효성 여부
        """
        try:
            params = self.config.parameters
            
            # 기본 유효성 검사
            if signal.confidence < params['min_confidence']:
                self.logger.debug(f"신뢰도 부족: {signal.confidence} < {params['min_confidence']}")
                return False
            
            if signal.price <= 0:
                self.logger.warning("유효하지 않은 가격")
                return False
            
            # 신호 타입 검증
            if signal.signal_type not in [SignalType.BUY, SignalType.SELL]:
                self.logger.debug(f"지원하지 않는 신호 타입: {signal.signal_type}")
                return False
            
            # 최근 신호와의 중복 확인 (과도한 거래 방지)
            if len(self.signal_history) > 0:
                last_signal = self.signal_history[-1]
                time_diff = (signal.timestamp - last_signal.timestamp).total_seconds()
                
                # 5분 이내 동일한 신호는 무시
                if (time_diff < 300 and 
                    signal.signal_type == last_signal.signal_type and
                    signal.symbol == last_signal.symbol):
                    self.logger.debug("최근 신호와 중복됨")
                    return False
            
            # 변동성 확인 (너무 낮은 변동성에서는 거래하지 않음)
            if len(self.price_history) >= 20:
                volatility = MarketAnalysis.calculate_volatility(self.price_history, 20)
                if volatility and volatility < 0.005:  # 0.5% 미만 변동성
                    self.logger.debug(f"변동성 부족: {volatility:.4f}")
                    return False
            
            self.logger.debug(f"신호 검증 통과: {signal.signal_type.value} @ {signal.price}")
            return True
            
        except Exception as e:
            self.logger.error(f"신호 검증 중 오류: {e}")
            return False
    
    def get_required_parameters(self) -> List[str]:
        """필수 매개변수 목록 반환"""
        return [
            'rsi_period',
            'rsi_overbought', 
            'rsi_oversold',
            'sma_short',
            'sma_long',
            'risk_percentage',
            'min_confidence'
        ]
    
    def get_strategy_info(self) -> dict:
        """전략 정보 반환"""
        return {
            'name': '단순 모멘텀 전략',
            'version': '1.0.0',
            'description': 'RSI, 이동평균, 볼린저밴드를 이용한 단타 전략',
            'indicators_used': ['RSI', 'SMA', 'Bollinger Bands'],
            'timeframe': '1분-5분',
            'risk_level': 'Medium',
            'suitable_for': ['BTC/USDT', 'ETH/USDT', '주요 암호화폐'],
            'parameters': self.config.parameters,
            'signals_generated': len(self.signal_history),
            'data_points': len(self.price_history)
        }
    
    def reset_data(self):
        """전략 데이터 초기화"""
        self.price_history.clear()
        self.volume_history.clear()
        self.signal_history.clear()
        self.performance_metrics.clear()
        self.logger.info("전략 데이터 초기화 완료")
    
    def get_current_indicators(self) -> dict:
        """현재 기술적 지표 상태 반환"""
        if len(self.price_history) < 20:
            return {}
        
        indicators = calculate_all_indicators(
            self.price_history, 
            self.volume_history
        )
        
        return {
            'rsi': indicators.rsi,
            'sma_20': indicators.sma,
            'ema_20': indicators.ema,
            'bollinger_upper': indicators.bollinger_upper,
            'bollinger_lower': indicators.bollinger_lower,
            'trend': MarketAnalysis.detect_trend(self.price_history, 20),
            'volatility': MarketAnalysis.calculate_volatility(self.price_history, 20),
            'current_price': self.price_history[-1] if self.price_history else None
        }