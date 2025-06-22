# 파일: tests/test_strategies.py
# 경로: /workspaces/autoblitz-backend/tests/test_strategies.py

"""
전략 플러그인 시스템 테스트

전략 기본 인터페이스, 매니저, 샘플 전략의 동작을 검증합니다.
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from typing import List

# 테스트를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.strategies.core.strategy_base import (
    StrategyBase, StrategyConfig, StrategyStatus, 
    MarketData, TradingSignal, SignalType, BacktestResult
)
from app.strategies.core.strategy_manager import StrategyManager
from app.strategies.utils.strategy_utils import (
    TechnicalAnalysis, RiskManagement, MarketAnalysis, calculate_all_indicators
)


class TestStrategyBase:
    """StrategyBase 클래스 테스트"""
    
    def create_sample_config(self) -> StrategyConfig:
        """테스트용 전략 설정 생성"""
        return StrategyConfig(
            name="test_strategy",
            version="1.0.0",
            description="테스트용 전략",
            parameters={
                'risk_percentage': 0.02,
                'min_confidence': 0.6
            },
            risk_level=0.5,
            min_balance=100.0,
            supported_exchanges=["okx", "upbit"]
        )
    
    def create_sample_market_data(self, price: float = 50000) -> MarketData:
        """테스트용 시장 데이터 생성"""
        return MarketData(
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            open=price * 0.99,
            high=price * 1.01,
            low=price * 0.98,
            close=price,
            volume=1000.0,
            bid=price * 0.999,
            ask=price * 1.001
        )
    
    def test_strategy_config_creation(self):
        """전략 설정 생성 테스트"""
        config = self.create_sample_config()
        
        assert config.name == "test_strategy"
        assert config.version == "1.0.0"
        assert config.risk_level == 0.5
        assert "risk_percentage" in config.parameters
    
    def test_market_data_creation(self):
        """시장 데이터 생성 테스트"""
        market_data = self.create_sample_market_data()
        
        assert market_data.symbol == "BTC/USDT"
        assert market_data.close == 50000
        assert market_data.high > market_data.low
        assert isinstance(market_data.timestamp, datetime)


class TestTechnicalAnalysis:
    """기술적 분석 유틸리티 테스트"""
    
    def create_sample_prices(self, count: int = 50) -> List[float]:
        """테스트용 가격 데이터 생성"""
        prices = []
        base_price = 50000
        
        for i in range(count):
            # 간단한 랜덤 워크
            change = (i % 3 - 1) * 100  # -100, 0, 100 변화
            base_price += change
            prices.append(max(base_price, 1000))  # 최소 1000
        
        return prices
    
    def test_simple_moving_average(self):
        """단순이동평균 계산 테스트"""
        prices = [100, 110, 120, 130, 140]
        
        sma_3 = TechnicalAnalysis.simple_moving_average(prices, 3)
        assert sma_3 == 130.0  # (120 + 130 + 140) / 3
        
        sma_5 = TechnicalAnalysis.simple_moving_average(prices, 5)
        assert sma_5 == 120.0  # (100 + 110 + 120 + 130 + 140) / 5
        
        # 데이터 부족시 None 반환
        sma_10 = TechnicalAnalysis.simple_moving_average(prices, 10)
        assert sma_10 is None
    
    def test_exponential_moving_average(self):
        """지수이동평균 계산 테스트"""
        prices = [100, 110, 120, 130, 140]
        
        ema = TechnicalAnalysis.exponential_moving_average(prices, 3)
        assert ema is not None
        assert isinstance(ema, float)
        
        # 데이터 부족시 None 반환
        ema_none = TechnicalAnalysis.exponential_moving_average([100], 5)
        assert ema_none is None
    
    def test_rsi_calculation(self):
        """RSI 계산 테스트"""
        # 상승 추세 데이터
        rising_prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 
                        150, 155, 160, 165, 170, 175]
        
        rsi = TechnicalAnalysis.rsi(rising_prices, 14)
        assert rsi is not None
        assert 0 <= rsi <= 100
        assert rsi > 50  # 상승 추세에서는 RSI가 50 이상
        
        # 데이터 부족시 None 반환
        rsi_none = TechnicalAnalysis.rsi([100, 110], 14)
        assert rsi_none is None
    
    def test_bollinger_bands(self):
        """볼린저 밴드 계산 테스트"""
        prices = self.create_sample_prices(30)
        
        upper, middle, lower = TechnicalAnalysis.bollinger_bands(prices, 20, 2.0)
        
        assert upper is not None
        assert middle is not None
        assert lower is not None
        assert upper > middle > lower
    
    def test_calculate_all_indicators(self):
        """모든 지표 계산 테스트"""
        prices = self.create_sample_prices(50)
        volumes = [1000 + i * 10 for i in range(50)]
        
        indicators = calculate_all_indicators(prices, volumes)
        
        assert indicators.sma is not None
        assert indicators.ema is not None
        assert indicators.rsi is not None
        assert indicators.volume_sma is not None


class TestRiskManagement:
    """위험 관리 유틸리티 테스트"""
    
    def test_position_size_calculation(self):
        """포지션 크기 계산 테스트"""
        account_balance = 10000
        risk_percentage = 0.02  # 2%
        entry_price = 50000
        stop_loss_price = 48500
        
        position_size = RiskManagement.calculate_position_size(
            account_balance, risk_percentage, entry_price, stop_loss_price
        )
        
        assert position_size > 0
        assert isinstance(position_size, float)
        
        # 위험 금액 검증
        risk_amount = account_balance * risk_percentage
        price_diff = abs(entry_price - stop_loss_price)
        expected_size = risk_amount / price_diff
        
        assert abs(position_size - expected_size) < 0.001 or position_size <= expected_size
    
    def test_kelly_criterion(self):
        """켈리 공식 테스트"""
        win_rate = 0.6
        avg_win = 150
        avg_loss = 100
        
        kelly_ratio = RiskManagement.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        
        assert kelly_ratio >= 0
        assert kelly_ratio <= 0.25  # 최대 25%로 제한
    
    def test_sharpe_ratio(self):
        """샤프 비율 계산 테스트"""
        returns = [0.01, 0.02, -0.01, 0.03, 0.02, -0.005, 0.015]
        
        sharpe = RiskManagement.calculate_sharpe_ratio(returns, 0.02)
        
        assert sharpe is not None
        assert isinstance(sharpe, float)
    
    def test_max_drawdown(self):
        """최대 낙폭 계산 테스트"""
        prices = [100, 110, 105, 120, 90, 95, 115, 85, 100]
        
        max_dd, start_idx, end_idx = RiskManagement.calculate_max_drawdown(prices)
        
        assert max_dd >= 0
        assert max_dd <= 1
        assert start_idx >= 0
        assert end_idx >= start_idx


class TestMarketAnalysis:
    """시장 분석 유틸리티 테스트"""
    
    def test_trend_detection(self):
        """트렌드 감지 테스트"""
        # 상승 추세
        uptrend_prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145,
                         150, 155, 160, 165, 170, 175, 180, 185, 190, 195]
        
        trend = MarketAnalysis.detect_trend(uptrend_prices, 20)
        assert trend == 'uptrend'
        
        # 하락 추세
        downtrend_prices = [200, 195, 190, 185, 180, 175, 170, 165, 160, 155,
                           150, 145, 140, 135, 130, 125, 120, 115, 110, 105]
        
        trend = MarketAnalysis.detect_trend(downtrend_prices, 20)
        assert trend == 'downtrend'
        
        # 횡보
        sideways_prices = [100] * 20
        trend = MarketAnalysis.detect_trend(sideways_prices, 20)
        assert trend == 'sideways'
    
    def test_volatility_calculation(self):
        """변동성 계산 테스트"""
        # 높은 변동성 데이터
        volatile_prices = [100, 120, 90, 110, 80, 130, 70, 140, 60, 150]
        volatility = MarketAnalysis.calculate_volatility(volatile_prices, 10)
        
        assert volatility is not None
        assert volatility > 0
        
        # 낮은 변동성 데이터
        stable_prices = [100, 101, 99, 100.5, 99.5, 100.2, 99.8, 100.1, 99.9, 100]
        low_volatility = MarketAnalysis.calculate_volatility(stable_prices, 10)
        
        assert low_volatility is not None
        assert low_volatility < volatility
    
    def test_support_resistance(self):
        """지지선/저항선 탐지 테스트"""
        # 명확한 고점/저점이 있는 데이터
        prices = [100, 95, 90, 95, 100, 105, 110, 105, 100, 95, 90, 95, 100]
        
        support_levels, resistance_levels = MarketAnalysis.find_support_resistance(prices, 2)
        
        assert isinstance(support_levels, list)
        assert isinstance(resistance_levels, list)


class TestStrategyManager:
    """전략 매니저 테스트"""
    
    @pytest.fixture
    def strategy_manager(self):
        """테스트용 전략 매니저"""
        # 테스트용 임시 디렉토리 사용
        manager = StrategyManager("app/strategies/plugins")
        yield manager
        manager.cleanup()
    
    def test_manager_initialization(self, strategy_manager):
        """매니저 초기화 테스트"""
        assert isinstance(strategy_manager, StrategyManager)
        assert len(strategy_manager.loaded_strategies) == 0
        assert len(strategy_manager.active_strategies) == 0
    
    def test_strategy_discovery(self, strategy_manager):
        """전략 스캔 테스트"""
        strategies = strategy_manager.discover_strategies()
        assert isinstance(strategies, list)
    
    def test_manager_status(self, strategy_manager):
        """매니저 상태 조회 테스트"""
        status = strategy_manager.get_manager_status()
        
        assert 'loaded_strategies' in status
        assert 'active_strategies' in status
        assert 'strategy_list' in status
        assert 'timestamp' in status


def test_integration_strategy_system():
    """전략 시스템 통합 테스트"""
    # 전략 설정 생성
    config = StrategyConfig(
        name="integration_test",
        version="1.0.0",
        description="통합 테스트용 전략",
        parameters={
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'sma_short': 10,
            'sma_long': 20,
            'risk_percentage': 0.02,
            'min_confidence': 0.6
        },
        risk_level=0.5,
        min_balance=100.0,
        supported_exchanges=["okx"]
    )
    
    # 시장 데이터 생성
    market_data = MarketData(
        symbol="BTC/USDT",
        timestamp=datetime.now(),
        open=49500,
        high=50500,
        low=49000,
        close=50000,
        volume=1000.0
    )
    
    # 기술적 지표 계산
    prices = [49000, 49200, 49500, 49800, 50000]
    indicators = calculate_all_indicators(prices)
    
    assert indicators is not None
    
    # 위험 관리 계산
    position_size = RiskManagement.calculate_position_size(
        account_balance=10000,
        risk_percentage=0.02,
        entry_price=50000,
        stop_loss_price=48500
    )
    
    assert position_size > 0


if __name__ == "__main__":
    # 개별 테스트 실행
    pytest.main([__file__, "-v"])