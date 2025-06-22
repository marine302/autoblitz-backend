# 파일: app/strategies/core/strategy_base.py
# 경로: /workspaces/autoblitz-backend/app/strategies/core/strategy_base.py

"""
오토블리츠 전략 플러그인 시스템 - 기본 인터페이스

모든 거래 전략이 상속받아야 하는 기본 클래스입니다.
플러그인 아키텍처를 통해 동적으로 전략을 로드하고 실행할 수 있습니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """전략 실행 상태"""
    INACTIVE = "inactive"      # 비활성
    ACTIVE = "active"         # 활성
    PAUSED = "paused"         # 일시정지
    ERROR = "error"           # 오류
    BACKTESTING = "backtesting"  # 백테스트 중


class SignalType(Enum):
    """거래 신호 타입"""
    BUY = "buy"              # 매수
    SELL = "sell"            # 매도
    HOLD = "hold"            # 보유
    CLOSE = "close"          # 포지션 닫기


@dataclass
class MarketData:
    """시장 데이터 구조"""
    symbol: str              # 심볼 (예: BTC/USDT)
    timestamp: datetime      # 시간
    open: float             # 시가
    high: float             # 고가
    low: float              # 저가
    close: float            # 종가
    volume: float           # 거래량
    bid: Optional[float] = None    # 매수호가
    ask: Optional[float] = None    # 매도호가


@dataclass
class TradingSignal:
    """거래 신호 구조"""
    symbol: str             # 심볼
    signal_type: SignalType # 신호 타입
    price: float           # 가격
    quantity: float        # 수량
    confidence: float      # 신뢰도 (0.0 ~ 1.0)
    timestamp: datetime    # 생성 시간
    metadata: Dict[str, Any] = None  # 추가 메타데이터


@dataclass
class StrategyConfig:
    """전략 설정"""
    name: str              # 전략명
    version: str           # 버전
    description: str       # 설명
    parameters: Dict[str, Any]  # 매개변수
    risk_level: float      # 위험도 (0.0 ~ 1.0)
    min_balance: float     # 최소 잔고
    supported_exchanges: List[str]  # 지원 거래소


@dataclass
class BacktestResult:
    """백테스트 결과"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_return: float        # 총 수익률
    sharpe_ratio: float       # 샤프 비율
    max_drawdown: float       # 최대 낙폭
    win_rate: float           # 승률
    total_trades: int         # 총 거래 수
    profit_trades: int        # 수익 거래 수
    loss_trades: int          # 손실 거래 수
    avg_profit: float         # 평균 수익
    avg_loss: float           # 평균 손실


class StrategyBase(ABC):
    """
    모든 거래 전략의 기본 클래스
    
    모든 전략은 이 클래스를 상속받아 구현해야 합니다.
    플러그인 시스템을 통해 동적으로 로드됩니다.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        전략 초기화
        
        Args:
            config: 전략 설정
        """
        self.config = config
        self.status = StrategyStatus.INACTIVE
        self.logger = logging.getLogger(f"strategy.{config.name}")
        self.last_signal: Optional[TradingSignal] = None
        self.performance_metrics: Dict[str, float] = {}
        
        # 전략별 데이터 저장소
        self.data_cache: Dict[str, Any] = {}
        
        self.logger.info(f"전략 '{self.config.name}' 초기화 완료")
    
    @abstractmethod
    def analyze(self, market_data: MarketData) -> Optional[TradingSignal]:
        """
        시장 데이터를 분석하여 거래 신호 생성
        
        Args:
            market_data: 현재 시장 데이터
            
        Returns:
            TradingSignal: 거래 신호 (없으면 None)
        """
        pass
    
    @abstractmethod
    def get_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """
        거래 신호에 따른 포지션 크기 계산
        
        Args:
            signal: 거래 신호
            account_balance: 계좌 잔고
            
        Returns:
            float: 포지션 크기
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        거래 신호 유효성 검증
        
        Args:
            signal: 거래 신호
            
        Returns:
            bool: 유효성 여부
        """
        pass
    
    def start(self) -> bool:
        """
        전략 시작
        
        Returns:
            bool: 시작 성공 여부
        """
        try:
            if self.status == StrategyStatus.ACTIVE:
                self.logger.warning("전략이 이미 활성 상태입니다")
                return True
                
            # 전략 시작 전 검증
            if not self._validate_config():
                return False
                
            self.status = StrategyStatus.ACTIVE
            self.logger.info(f"전략 '{self.config.name}' 시작됨")
            return True
            
        except Exception as e:
            self.logger.error(f"전략 시작 실패: {e}")
            self.status = StrategyStatus.ERROR
            return False
    
    def stop(self) -> bool:
        """
        전략 정지
        
        Returns:
            bool: 정지 성공 여부
        """
        try:
            self.status = StrategyStatus.INACTIVE
            self.logger.info(f"전략 '{self.config.name}' 정지됨")
            return True
            
        except Exception as e:
            self.logger.error(f"전략 정지 실패: {e}")
            return False
    
    def pause(self) -> bool:
        """
        전략 일시정지
        
        Returns:
            bool: 일시정지 성공 여부
        """
        if self.status == StrategyStatus.ACTIVE:
            self.status = StrategyStatus.PAUSED
            self.logger.info(f"전략 '{self.config.name}' 일시정지됨")
            return True
        return False
    
    def resume(self) -> bool:
        """
        전략 재개
        
        Returns:
            bool: 재개 성공 여부
        """
        if self.status == StrategyStatus.PAUSED:
            self.status = StrategyStatus.ACTIVE
            self.logger.info(f"전략 '{self.config.name}' 재개됨")
            return True
        return False
    
    def get_status(self) -> StrategyStatus:
        """현재 전략 상태 반환"""
        return self.status
    
    def get_config(self) -> StrategyConfig:
        """전략 설정 반환"""
        return self.config
    
    def update_config(self, new_params: Dict[str, Any]) -> bool:
        """
        전략 매개변수 업데이트
        
        Args:
            new_params: 새로운 매개변수
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            self.config.parameters.update(new_params)
            self.logger.info(f"전략 매개변수 업데이트: {new_params}")
            return True
        except Exception as e:
            self.logger.error(f"매개변수 업데이트 실패: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """성능 지표 반환"""
        return self.performance_metrics.copy()
    
    def backtest(
        self, 
        historical_data: List[MarketData], 
        initial_balance: float = 10000.0
    ) -> BacktestResult:
        """
        전략 백테스트 실행
        
        Args:
            historical_data: 과거 시장 데이터
            initial_balance: 초기 잔고
            
        Returns:
            BacktestResult: 백테스트 결과
        """
        self.logger.info(f"백테스트 시작: {len(historical_data)}개 데이터포인트")
        self.status = StrategyStatus.BACKTESTING
        
        try:
            balance = initial_balance
            trades = []
            positions = {}
            
            for data in historical_data:
                signal = self.analyze(data)
                
                if signal and self.validate_signal(signal):
                    position_size = self.get_position_size(signal, balance)
                    
                    # 거래 실행 시뮬레이션
                    trade_result = self._simulate_trade(signal, position_size, data)
                    if trade_result:
                        trades.append(trade_result)
                        balance += trade_result.get('pnl', 0)
            
            # 백테스트 결과 계산
            result = self._calculate_backtest_metrics(
                trades, initial_balance, balance, historical_data
            )
            
            self.status = StrategyStatus.INACTIVE
            self.logger.info(f"백테스트 완료: 총 수익률 {result.total_return:.2%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"백테스트 실행 중 오류: {e}")
            self.status = StrategyStatus.ERROR
            raise
    
    def _validate_config(self) -> bool:
        """전략 설정 유효성 검증"""
        try:
            required_params = self.get_required_parameters()
            for param in required_params:
                if param not in self.config.parameters:
                    self.logger.error(f"필수 매개변수 누락: {param}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"설정 검증 실패: {e}")
            return False
    
    def _simulate_trade(
        self, 
        signal: TradingSignal, 
        position_size: float, 
        market_data: MarketData
    ) -> Optional[Dict[str, Any]]:
        """거래 시뮬레이션"""
        try:
            # 간단한 거래 시뮬레이션 로직
            # 실제 구현에서는 더 복잡한 슬리피지, 수수료 등을 고려
            price = signal.price
            
            trade = {
                'symbol': signal.symbol,
                'type': signal.signal_type.value,
                'price': price,
                'quantity': position_size,
                'timestamp': signal.timestamp,
                'pnl': 0  # 손익은 포지션 청산시 계산
            }
            
            return trade
            
        except Exception as e:
            self.logger.error(f"거래 시뮬레이션 실패: {e}")
            return None
    
    def _calculate_backtest_metrics(
        self, 
        trades: List[Dict[str, Any]], 
        initial_balance: float, 
        final_balance: float,
        historical_data: List[MarketData]
    ) -> BacktestResult:
        """백테스트 성능 지표 계산"""
        
        total_return = (final_balance - initial_balance) / initial_balance
        total_trades = len(trades)
        profit_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        loss_trades = total_trades - profit_trades
        win_rate = profit_trades / total_trades if total_trades > 0 else 0
        
        # 기본적인 지표들 (실제로는 더 정교한 계산 필요)
        result = BacktestResult(
            strategy_name=self.config.name,
            start_date=historical_data[0].timestamp if historical_data else datetime.now(),
            end_date=historical_data[-1].timestamp if historical_data else datetime.now(),
            total_return=total_return,
            sharpe_ratio=0.0,  # 실제 계산 구현 필요
            max_drawdown=0.0,  # 실제 계산 구현 필요
            win_rate=win_rate,
            total_trades=total_trades,
            profit_trades=profit_trades,
            loss_trades=loss_trades,
            avg_profit=0.0,    # 실제 계산 구현 필요
            avg_loss=0.0       # 실제 계산 구현 필요
        )
        
        return result
    
    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        """
        전략에 필요한 필수 매개변수 목록 반환
        
        Returns:
            List[str]: 필수 매개변수 이름 목록
        """
        pass
    
    def __str__(self) -> str:
        """전략 정보 문자열 표현"""
        return f"Strategy(name={self.config.name}, status={self.status.value})"
    
    def __repr__(self) -> str:
        """전략 객체 표현"""
        return (f"StrategyBase(name='{self.config.name}', "
                f"version='{self.config.version}', "
                f"status='{self.status.value}')")