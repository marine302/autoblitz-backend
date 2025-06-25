"""
전략 모델 정의
매매 전략 설정 및 관리
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Text, ForeignKey, Numeric
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base


class Strategy(Base):
    """매매 전략 테이블"""
    __tablename__ = "strategies"
    
    # 기본 정보
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    version = Column(String(20), default="1.0.0")
    
    # 작성자 정보
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_public = Column(Boolean, default=False)  # 공개 전략 여부
    is_verified = Column(Boolean, default=False)  # 검증된 전략 여부
    
    # 전략 타입
    strategy_type = Column(String(50), default="dantaro")  # dantaro, scalping, swing 등
    time_frame = Column(String(10), default="1m")  # 1m, 5m, 15m, 1h, 4h, 1d
    
    # 전략 설정
    parameters = Column(JSON, default={
        "entry_conditions": [],
        "exit_conditions": [],
        "risk_management": {},
        "indicators": {}
    })
    
    # 성능 지표
    total_backtests = Column(Integer, default=0)
    success_rate = Column(Numeric(5, 2), default=0)  # 성공률 (%)
    avg_profit = Column(Numeric(10, 4), default=0)   # 평균 수익률
    max_drawdown = Column(Numeric(10, 4), default=0) # 최대 낙폭
    sharpe_ratio = Column(Numeric(10, 4), default=0) # 샤프 비율
    
    # 사용 통계
    active_bots = Column(Integer, default=0)  # 현재 사용 중인 봇 수
    total_users = Column(Integer, default=0)  # 총 사용자 수
    
    # 평가 정보
    rating = Column(Numeric(3, 2), default=0)  # 평점 (0-5)
    review_count = Column(Integer, default=0)
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_backtest = Column(DateTime(timezone=True))
    
    # 관계 설정
    creator = relationship("User", back_populates="strategies")
    bots = relationship("Bot", back_populates="strategy")
    backtests = relationship("Backtest", back_populates="strategy", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Strategy(id={self.id}, name={self.name}, type={self.strategy_type})>"


class Backtest(Base):
    """백테스트 결과 테이블"""
    __tablename__ = "backtests"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # 백테스트 설정
    trading_pair = Column(String(20), nullable=False)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    initial_capital = Column(Numeric(20, 8), default=10000)
    
    # 백테스트 결과
    final_capital = Column(Numeric(20, 8))
    total_return = Column(Numeric(10, 4))  # 총 수익률
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    
    # 리스크 메트릭
    max_drawdown = Column(Numeric(10, 4))
    sharpe_ratio = Column(Numeric(10, 4))
    sortino_ratio = Column(Numeric(10, 4))
    volatility = Column(Numeric(10, 4))
    
    # 세부 결과
    trade_details = Column(JSON)  # 개별 거래 상세 정보
    equity_curve = Column(JSON)   # 자산 곡선 데이터
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # 관계 설정
    strategy = relationship("Strategy", back_populates="backtests")
    
    def __repr__(self):
        return f"<Backtest(id={self.id}, strategy_id={self.strategy_id})>"


class StrategyReview(Base):
    """전략 리뷰 테이블"""
    __tablename__ = "strategy_reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # 리뷰 내용
    rating = Column(Integer, nullable=False)  # 1-5 점
    title = Column(String(200))
    content = Column(Text)
    
    # 사용 기간
    usage_duration = Column(Integer)  # 사용 기간 (일)
    total_trades = Column(Integer)    # 총 거래 수
    profit_loss = Column(Numeric(20, 8))  # 실제 손익
    
    # 상태
    is_verified = Column(Boolean, default=False)  # 검증된 리뷰
    is_hidden = Column(Boolean, default=False)
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<StrategyReview(id={self.id}, rating={self.rating})>"