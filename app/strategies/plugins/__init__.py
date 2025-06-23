# 전략 플러그인

# 기존 전략들
from .simple_momentum import *

# 단타로 전략 임포트
from .scalping_strategy import ScalpingStrategy, ScalpingConfig, create_scalping_strategy

__all__ = [
    'ScalpingStrategy', 
    'ScalpingConfig', 
    'create_scalping_strategy'
]