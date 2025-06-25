"""
사용자 모델 정의
오토블리츠 사용자 정보 및 인증 관리
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base


class User(Base):
    """사용자 테이블"""
    __tablename__ = "users"
    
    # 기본 정보
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # 사용자 상태
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)
    
    # 프로필 정보
    full_name = Column(String(100))
    phone = Column(String(20))
    timezone = Column(String(50), default="Asia/Seoul")
    
    # 거래소 연결 정보 (암호화 저장)
    exchange_credentials = Column(JSON)  # 암호화된 API 키들
    
    # 알림 설정
    notification_settings = Column(JSON, default={
        "email_alerts": True,
        "sms_alerts": False,
        "discord_webhook": None,
        "telegram_chat_id": None
    })
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # 관계 설정
    bots = relationship("Bot", back_populates="owner", cascade="all, delete-orphan")
    strategies = relationship("Strategy", back_populates="creator")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class UserSession(Base):
    """사용자 세션 관리"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_token = Column(String(255), unique=True, nullable=False)
    refresh_token = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # 세션 정보
    ip_address = Column(String(45))  # IPv6 지원
    user_agent = Column(Text)
    device_info = Column(JSON)
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id})>"