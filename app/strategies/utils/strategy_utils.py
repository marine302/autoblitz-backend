# 파일: app/strategies/utils/strategy_utils.py
# 경로: /workspaces/autoblitz-backend/app/strategies/utils/strategy_utils.py

"""
오토블리츠 전략 유틸리티 모듈

전략 구현에 필요한 공통 유틸리티 함수들을 제공합니다.
기술적 지표, 위험 관리, 포지션 계산 등의 기능을 포함합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """기술적 지표 결과"""
    sma: Optional[float] = None          # 단순이동평균
    ema: Optional[float] = None          # 지수이동평균
    rsi: Optional[float] = None          # RSI
    macd: Optional[float] = None         # MACD
    macd_signal: Optional[float] = None  # MACD 시그널
    bollinger_upper: Optional[float] = None  # 볼린저 밴드 상단
    bollinger_lower: Optional[float] = None  # 볼린저 밴드 하단
    volume_sma: Optional[float] = None   # 거래량 이동평균


class TechnicalAnalysis:
    """기술적 분석 유틸리티"""
    
    @staticmethod
    def simple_moving_average(prices: List[float], period: int) -> Optional[float]:
        """
        단순이동평균 계산
        
        Args:
            prices: 가격 리스트
            period: 기간
            
        Returns:
            float: SMA 값
        """
        if len(prices) < period:
            return None
        
        return sum(prices[-period:]) / period
    
    @staticmethod
    def exponential_moving_average(
        prices: List[float], 
        period: int, 
        prev_ema: Optional[float] = None
    ) -> Optional[float]:
        """
        지수이동평균 계산
        
        Args:
            prices: 가격 리스트
            period: 기간
            prev_ema: 이전 EMA 값
            
        Returns:
            float: EMA 값
        """
        if len(prices) == 0:
            return None
        
        current_price = prices[-1]
        multiplier = 2 / (period + 1)
        
        if prev_ema is None:
            if len(prices) < period:
                return None
            # 첫 번째 EMA는 SMA로 계산
            prev_ema = TechnicalAnalysis.simple_moving_average(prices[:-1], period)
            if prev_ema is None:
                return None
        
        return (current_price * multiplier) + (prev_ema * (1 - multiplier))
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """
        RSI (Relative Strength Index) 계산
        
        Args:
            prices: 가격 리스트
            period: 계산 기간
            
        Returns:
            float: RSI 값 (0-100)
        """
        if len(prices) < period + 1:
            return None
        
        # 가격 변화 계산
        price_changes = []
        for i in range(1, len(prices)):
            price_changes.append(prices[i] - prices[i-1])
        
        if len(price_changes) < period:
            return None
        
        # 상승/하락 분리
        gains = [change if change > 0 else 0 for change in price_changes[-period:]]
        losses = [-change if change < 0 else 0 for change in price_changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(
        prices: List[float], 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        MACD (Moving Average Convergence Divergence) 계산
        
        Args:
            prices: 가격 리스트
            fast_period: 빠른 EMA 기간
            slow_period: 느린 EMA 기간
            signal_period: 시그널 라인 기간
            
        Returns:
            Tuple[float, float]: (MACD, Signal)
        """
        if len(prices) < slow_period:
            return None, None
        
        # EMA 계산을 위한 간단한 구현
        fast_ema = TechnicalAnalysis.exponential_moving_average(prices, fast_period)
        slow_ema = TechnicalAnalysis.exponential_moving_average(prices, slow_period)
        
        if fast_ema is None or slow_ema is None:
            return None, None
        
        macd_line = fast_ema - slow_ema
        
        # 시그널 라인은 MACD의 EMA (간단화)
        signal_line = macd_line  # 실제로는 MACD 값들의 EMA를 계산해야 함
        
        return macd_line, signal_line
    
    @staticmethod
    def bollinger_bands(
        prices: List[float], 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        볼린저 밴드 계산
        
        Args:
            prices: 가격 리스트
            period: 이동평균 기간
            std_dev: 표준편차 배수
            
        Returns:
            Tuple[float, float, float]: (상단, 중간, 하단)
        """
        if len(prices) < period:
            return None, None, None
        
        recent_prices = prices[-period:]
        middle = sum(recent_prices) / period
        
        # 표준편차 계산
        variance = sum((price - middle) ** 2 for price in recent_prices) / period
        std = variance ** 0.5
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower


class RiskManagement:
    """위험 관리 유틸리티"""
    
    @staticmethod
    def calculate_position_size(
        account_balance: float,
        risk_percentage: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        포지션 크기 계산 (리스크 기반)
        
        Args:
            account_balance: 계좌 잔고
            risk_percentage: 위험 비율 (0.01 = 1%)
            entry_price: 진입 가격
            stop_loss_price: 손절 가격
            
        Returns:
            float: 포지션 크기
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
        
        risk_amount = account_balance * risk_percentage
        price_difference = abs(entry_price - stop_loss_price)
        
        if price_difference == 0:
            return 0
        
        position_size = risk_amount / price_difference
        
        # 최대 포지션 크기 제한 (계좌의 50%)
        max_position_value = account_balance * 0.5
        max_position_size = max_position_value / entry_price
        
        return min(position_size, max_position_size)
    
    @staticmethod
    def calculate_kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        켈리 공식으로 최적 베팅 비율 계산
        
        Args:
            win_rate: 승률 (0.0 ~ 1.0)
            avg_win: 평균 수익
            avg_loss: 평균 손실
            
        Returns:
            float: 최적 베팅 비율
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0
        
        # 켈리 공식: f = (bp - q) / b
        # b = avg_win / avg_loss (승률비)
        # p = win_rate (승률)
        # q = 1 - p (패율)
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_ratio = (b * p - q) / b
        
        # 켈리 비율을 25%로 제한 (안전을 위해)
        return min(max(kelly_ratio, 0), 0.25)
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.02
    ) -> Optional[float]:
        """
        샤프 비율 계산
        
        Args:
            returns: 수익률 리스트
            risk_free_rate: 무위험 수익률
            
        Returns:
            float: 샤프 비율
        """
        if len(returns) < 2:
            return None
        
        avg_return = sum(returns) / len(returns)
        
        # 표준편차 계산
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return None
        
        excess_return = avg_return - risk_free_rate
        sharpe_ratio = excess_return / std_dev
        
        return sharpe_ratio
    
    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> Tuple[float, int, int]:
        """
        최대 낙폭 계산
        
        Args:
            prices: 가격 리스트
            
        Returns:
            Tuple[float, int, int]: (최대낙폭, 고점인덱스, 저점인덱스)
        """
        if len(prices) < 2:
            return 0.0, 0, 0
        
        peak = prices[0]
        peak_idx = 0
        max_drawdown = 0.0
        max_dd_start = 0
        max_dd_end = 0
        
        for i, price in enumerate(prices):
            if price > peak:
                peak = price
                peak_idx = i
            
            drawdown = (peak - price) / peak if peak > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_dd_start = peak_idx
                max_dd_end = i
        
        return max_drawdown, max_dd_start, max_dd_end


class MarketAnalysis:
    """시장 분석 유틸리티"""
    
    @staticmethod
    def detect_trend(prices: List[float], period: int = 20) -> str:
        """
        트렌드 감지
        
        Args:
            prices: 가격 리스트
            period: 분석 기간
            
        Returns:
            str: 'uptrend', 'downtrend', 'sideways'
        """
        if len(prices) < period:
            return 'sideways'
        
        recent_prices = prices[-period:]
        first_half = recent_prices[:period//2]
        second_half = recent_prices[period//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percentage = (second_avg - first_avg) / first_avg
        
        if change_percentage > 0.02:  # 2% 이상 상승
            return 'uptrend'
        elif change_percentage < -0.02:  # 2% 이상 하락
            return 'downtrend'
        else:
            return 'sideways'
    
    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 20) -> Optional[float]:
        """
        변동성 계산 (표준편차 기반)
        
        Args:
            prices: 가격 리스트
            period: 계산 기간
            
        Returns:
            float: 변동성
        """
        if len(prices) < period:
            return None
        
        recent_prices = prices[-period:]
        
        # 수익률 계산
        returns = []
        for i in range(1, len(recent_prices)):
            ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            returns.append(ret)
        
        if len(returns) < 2:
            return None
        
        # 표준편차 계산
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        return volatility
    
    @staticmethod
    def find_support_resistance(
        prices: List[float], 
        window: int = 5
    ) -> Tuple[List[float], List[float]]:
        """
        지지선과 저항선 찾기
        
        Args:
            prices: 가격 리스트
            window: 탐지 윈도우 크기
            
        Returns:
            Tuple[List[float], List[float]]: (지지선 목록, 저항선 목록)
        """
        if len(prices) < window * 2 + 1:
            return [], []
        
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(prices) - window):
            # 지지선 (저점)
            is_support = all(
                prices[i] <= prices[i + j] for j in range(-window, window + 1) if j != 0
            )
            
            # 저항선 (고점)
            is_resistance = all(
                prices[i] >= prices[i + j] for j in range(-window, window + 1) if j != 0
            )
            
            if is_support:
                support_levels.append(prices[i])
            
            if is_resistance:
                resistance_levels.append(prices[i])
        
        return support_levels, resistance_levels


def calculate_all_indicators(
    prices: List[float],
    volumes: Optional[List[float]] = None
) -> TechnicalIndicators:
    """
    모든 기술적 지표를 한 번에 계산
    
    Args:
        prices: 가격 리스트
        volumes: 거래량 리스트 (선택사항)
        
    Returns:
        TechnicalIndicators: 계산된 지표들
    """
    indicators = TechnicalIndicators()
    
    try:
        # 이동평균
        indicators.sma = TechnicalAnalysis.simple_moving_average(prices, 20)
        indicators.ema = TechnicalAnalysis.exponential_moving_average(prices, 20)
        
        # RSI
        indicators.rsi = TechnicalAnalysis.rsi(prices, 14)
        
        # MACD
        macd, signal = TechnicalAnalysis.macd(prices)
        indicators.macd = macd
        indicators.macd_signal = signal
        
        # 볼린저 밴드
        upper, middle, lower = TechnicalAnalysis.bollinger_bands(prices)
        indicators.bollinger_upper = upper
        indicators.bollinger_lower = lower
        
        # 거래량 지표
        if volumes:
            indicators.volume_sma = TechnicalAnalysis.simple_moving_average(volumes, 20)
        
    except Exception as e:
        logger.error(f"지표 계산 중 오류: {e}")
    
    return indicators