"""
모델 패키지 초기화
모든 데이터베이스 모델을 여기서 임포트
"""

# 사용자 관련 모델
from app.models.user import User, UserSession

# 봇 관련 모델
from app.models.bot import Bot, Trade, BotStatus, BotType

# 전략 관련 모델
from app.models.strategy import Strategy, Backtest, StrategyReview

# 모든 모델을 한 번에 임포트할 수 있도록 __all__ 정의
__all__ = [
    # User models
    "User",
    "UserSession",
    
    # Bot models
    "Bot",
    "Trade", 
    "BotStatus",
    "BotType",
    
    # Strategy models
    "Strategy",
    "Backtest", 
    "StrategyReview"
]