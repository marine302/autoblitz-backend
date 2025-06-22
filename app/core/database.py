# 파일: app/core/database.py (수정된 버전)
# 경로: /workspaces/autoblitz-backend/app/core/database.py
"""
AutoBlitz 데이터베이스 연결 관리
SQLAlchemy 2.0 비동기 지원 + init_db 함수
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
import logging
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class Base(DeclarativeBase):
    """SQLAlchemy 베이스 클래스"""
    pass

# 데이터베이스 엔진 및 세션 설정
engine = None
async_session_maker = None

def get_database_url() -> str:
    """데이터베이스 URL 생성"""
    if settings.environment == "development":
        # SQLite (개발용)
        return "sqlite+aiosqlite:///./autoblitz.db"
    else:
        # MySQL (운영용)
        return f"mysql+aiomysql://{settings.database_user}:{settings.database_password}@{settings.database_host}:{settings.database_port}/{settings.database_name}"

def create_engine():
    """데이터베이스 엔진 생성"""
    global engine, async_session_maker
    
    database_url = get_database_url()
    
    # SQLite용 설정
    if database_url.startswith("sqlite"):
        engine = create_async_engine(
            database_url,
            echo=settings.environment == "development",
            connect_args={"check_same_thread": False}
        )
    else:
        # MySQL용 설정
        engine = create_async_engine(
            database_url,
            echo=settings.environment == "development",
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600
        )
    
    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    logger.info(f"데이터베이스 엔진 생성 완료: {database_url.split('://')[0]}")

async def init_db():
    """데이터베이스 초기화"""
    global engine
    
    try:
        # 엔진이 없으면 생성
        if engine is None:
            create_engine()
        
        # 기존 개별 모델들 import
        try:
            from ..models import user, bot, strategy  # 개별 모델 파일들
            # 모든 모델의 Base 메타데이터 수집
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("데이터베이스 테이블 생성 완료")
            
        except ImportError as e:
            logger.warning(f"모델 import 실패: {e}")
            # Base만으로 기본 테이블 생성
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("기본 데이터베이스 구조 생성 완료")
        
        logger.info("데이터베이스 초기화 완료")
        
    except Exception as e:
        logger.warning(f"데이터베이스 초기화 중 오류: {e}")
        # 개발 환경에서는 에러를 무시하고 계속 진행
        logger.info("개발 환경에서 데이터베이스 없이 계속 진행합니다")

async def get_db() -> AsyncSession:
    """데이터베이스 세션 의존성"""
    if async_session_maker is None:
        create_engine()
    
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def close_db():
    """데이터베이스 연결 종료"""
    global engine
    if engine:
        await engine.dispose()
        logger.info("데이터베이스 연결 종료")

# 편의 함수들
async def execute_query(query: str, params: dict = None):
    """원시 SQL 쿼리 실행"""
    if engine is None:
        create_engine()
    
    async with async_session_maker() as session:
        try:
            result = await session.execute(query, params or {})
            await session.commit()
            return result
        except Exception as e:
            await session.rollback()
            logger.error(f"쿼리 실행 실패: {e}")
            raise

async def test_connection():
    """데이터베이스 연결 테스트"""
    try:
        if engine is None:
            create_engine()
        
        async with async_session_maker() as session:
            await session.execute("SELECT 1")
            logger.info("데이터베이스 연결 테스트 성공")
            return True
            
    except Exception as e:
        logger.error(f"데이터베이스 연결 테스트 실패: {e}")
        return False