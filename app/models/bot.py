"""
봇 모델 정의
자동매매 봇 설정 및 상태 관리
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Text, ForeignKey, Numeric, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from app.core.database import Base


class BotStatus(PyEnum):
    """봇 상태"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class BotType(PyEnum):
    """봇 타입"""
    SPOT = "spot"        # 현물 거래
    FUTURES = "futures"  # 선물 거래
    HYBRID = "hybrid"    # 현물+선물


class Bot(Base):
    """자동매매 봇 테이블"""
    __tablename__ = "bots"
    
    # 기본 정보
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # 소유자 정보
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # 봇 설정
    bot_type = Column(Enum(BotType), default=BotType.SPOT)
    exchange = Column(String(20), nullable=False)  # okx, upbit
    trading_pair = Column(String(20), nullable=False)  # BTC/USDT
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    
    # 자금 관리
    allocated_amount = Column(Numeric(20, 8), default=0)  # 할당된 자금
    min_order_amount = Column(Numeric(20, 8), default=10)  # 최소 주문 금액
    max_position_size = Column(Numeric(20, 8), default=1000)  # 최대 포지션 크기
    
    # 리스크 관리
    stop_loss_percent = Column(Numeric(5, 2), default=5.0)  # 손절 비율
    take_profit_percent = Column(Numeric(5, 2), default=10.0)  # 익절 비율
    max_daily_loss = Column(Numeric(20, 8), default=100)  # 일일 최대 손실
    
    # 봇 상태
    status = Column(Enum(BotStatus), default=BotStatus.STOPPED)
    is_active = Column(Boolean, default=True)
    is_paper_trading = Column(Boolean, default=True)  # 모의투자 여부
    
    # 성과 지표
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(Numeric(20, 8), default=0)  # 총 손익
    daily_pnl = Column(Numeric(20, 8), default=0)  # 일일 손익
    
    # 설정 정보
    bot_config = Column(JSON, default={
        "check_interval": 30,  # 체크 간격 (초)
        "order_timeout": 300,  # 주문 타임아웃 (초)
        "retry_count": 3,      # 재시도 횟수
        "log_level": "INFO"
    })
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    stopped_at = Column(DateTime(timezone=True))
    
    # 관계 설정
    owner = relationship("User", back_populates="bots")
    strategy = relationship("Strategy", back_populates="bots")
    trades = relationship("Trade", back_populates="bot", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Bot(id={self.id}, name={self.name}, status={self.status})>"


class Trade(Base):
    """거래 기록 테이블"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False)
    
    # 거래 정보
    exchange_order_id = Column(String(100))  # 거래소 주문 ID
    trading_pair = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # buy, sell
    order_type = Column(String(20), default="market")  # market, limit
    
    # 가격 및 수량
    price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    filled_quantity = Column(Numeric(20, 8), default=0)
    
    # 수수료 및 손익
    fee = Column(Numeric(20, 8), default=0)
    pnl = Column(Numeric(20, 8), default=0)  # 실현 손익
    
    # 상태
    status = Column(String(20), default="pending")  # pending, filled, cancelled
    
    # 전략 정보
    strategy_signal = Column(JSON)  # 전략 신호 정보
    reason = Column(String(100))  # 거래 이유
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    executed_at = Column(DateTime(timezone=True))
    
    # 관계 설정
    bot = relationship("Bot", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(id={self.id}, bot_id={self.bot_id}, side={self.side})>"