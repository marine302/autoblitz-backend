# 파일명: app/strategies/utils/risk_management.py
"""
리스크 관리 모듈
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class RiskManager:
    """리스크 관리 클래스"""
    
    def __init__(self):
        self.max_risk_per_trade = 0.02  # 거래당 최대 리스크 2%
        self.max_daily_loss = 0.05      # 일일 최대 손실 5%
        self.max_drawdown = 0.10        # 최대 낙폭 10%
    
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                              stop_loss_price: float, risk_percentage: float = 0.02) -> float:
        """포지션 크기 계산"""
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0.0
        
        risk_amount = account_balance * risk_percentage
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0.0
        
        position_size = risk_amount / price_risk
        return min(position_size, account_balance * 0.1)  # 최대 10% 제한
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """켈리 공식 계산"""
        if avg_loss == 0 or win_rate <= 0:
            return 0.0
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return max(0, min(kelly, 0.25))  # 최대 25% 제한
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # 일일 무위험 수익률
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self, equity_curve: List[float]) -> Tuple[float, float]:
        """최대 낙폭 계산"""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0, 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        max_dd_percent = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown = peak - value
            drawdown_percent = drawdown / peak if peak > 0 else 0
            
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_percent = drawdown_percent
        
        return max_dd, max_dd_percent
    
    def check_risk_limits(self, current_positions: int, daily_pnl: float, 
                         account_balance: float) -> Dict[str, bool]:
        """리스크 한도 확인"""
        
        daily_loss_percent = abs(daily_pnl) / account_balance if account_balance > 0 else 0
        
        return {
            'max_positions_ok': current_positions <= 10,  # 최대 10개 포지션
            'daily_loss_ok': daily_loss_percent <= self.max_daily_loss,
            'account_positive': account_balance > 0,
            'can_trade': daily_loss_percent <= self.max_daily_loss and current_positions <= 10
        }