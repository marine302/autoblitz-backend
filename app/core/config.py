# 파일: app/core/config.py (업데이트된 버전)
# 경로: /workspaces/autoblitz-backend/app/core/config.py
"""
AutoBlitz 설정 관리
Pydantic Settings 기반 환경 설정
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 기본 설정
    app_name: str = "AutoBlitz Backend"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # JWT 설정
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    jwt_secret_key: str = Field(default="your-secret-key-change-in-production", env="JWT_SECRET_KEY")  # 기존 코드 호환성
    jwt_algorithm: str = "HS256"  # 기존 코드 호환성
    algorithm: str = "HS256"  # 새 코드용
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7  # 기존 코드에서 사용
    
    # 데이터베이스 설정
    database_host: str = Field(default="localhost", env="DB_HOST")
    database_port: int = Field(default=3306, env="DB_PORT")
    database_user: str = Field(default="autoblitz", env="DB_USER")
    database_password: str = Field(default="password", env="DB_PASSWORD")
    database_name: str = Field(default="autoblitz_db", env="DB_NAME")
    
    # Redis 설정
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # AWS 설정
    aws_region: str = Field(default="ap-northeast-2", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # 모니터링 설정
    cloudwatch_namespace: str = Field(default="AutoBlitz/Development", env="CLOUDWATCH_NAMESPACE")
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    
    # Rate Limiting 설정
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # 초
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 글로벌 설정 인스턴스
_settings = None

def get_settings() -> Settings:
    """설정 싱글톤 인스턴스 반환"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# 기존 코드 호환성을 위한 settings 객체
settings = get_settings()

# 환경별 설정
def is_development() -> bool:
    """개발 환경 여부 확인"""
    return get_settings().environment == "development"

def is_production() -> bool:
    """운영 환경 여부 확인"""
    return get_settings().environment == "production"