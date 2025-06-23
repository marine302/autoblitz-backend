# 파일명: app/strategies/utils/technical_analysis.py
"""
기술적 분석 모듈
"""

import numpy as np
from typing import List, Tuple

class TechnicalAnalysis:
    """기본 기술적 분석 클래스"""
    
    def calculate_sma(self, prices: np.ndarray, period: int = 20) -> float:
        """단순 이동평균 계산"""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0
        return np.mean(prices[-period:])
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, 
                                 std_dev: float = 2) -> Tuple[float, float, float]:
        """볼린저 밴드 계산"""
        if len(prices) < period:
            current_price = prices[-1] if len(prices) > 0 else 0
            return current_price, current_price, current_price
        
        sma = self.calculate_sma(prices, period)
        std = np.std(prices[-period:])
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band