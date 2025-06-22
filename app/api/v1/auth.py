# 파일: app/api/v1/auth.py
# 경로: /workspaces/autoblitz-backend/app/api/v1/auth.py

"""
오토블리츠 인증 API 라우터 (실제 JWT 구현)
실제 JWT 토큰 기반 보안 인증 시스템
"""
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from typing import Optional
import logging

from app.core.cache import cache_manager
from app.core.security import security_manager

logger = logging.getLogger(__name__)
security = HTTPBearer()

# 라우터 인스턴스
router = APIRouter(prefix="/auth", tags=["인증"])


# Pydantic 모델들
class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRegister(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        # 비밀번호 강도 검증
        validation = security_manager.validate_password_strength(v)
        if not validation["is_valid"]:
            raise ValueError(f"비밀번호가 보안 요구사항을 충족하지 않습니다: {', '.join(validation['errors'])}")
        return v
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError("사용자명은 최소 3자 이상이어야 합니다")
        if not v.isalnum():
            raise ValueError("사용자명은 영문자와 숫자만 사용 가능합니다")
        return v.lower()


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30분


class UserProfile(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    is_premium: bool


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        validation = security_manager.validate_password_strength(v)
        if not validation["is_valid"]:
            raise ValueError(f"새 비밀번호가 보안 요구사항을 충족하지 않습니다: {', '.join(validation['errors'])}")
        return v


# JWT 토큰 검증 의존성
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """실제 JWT 토큰 검증"""
    token = credentials.credentials
    
    # JWT 토큰 검증
    payload = security_manager.verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않거나 만료된 토큰입니다",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 사용자 정보 캐시에서 조회
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰에서 사용자 정보를 찾을 수 없습니다"
        )
    
    # 캐시에서 사용자 정보 조회
    user_key = f"user:{user_id}"
    user_data = await cache_manager.get(user_key)
    
    if not user_data:
        # 임시 사용자 데이터 (실제로는 DB에서 조회)
        if user_id == "1":  # 테스트 사용자
            user_data = {
                "id": 1,
                "email": "test@autoblitz.com",
                "username": "testuser",
                "full_name": "테스트 사용자",
                "is_active": True,
                "is_verified": True,
                "is_premium": False
            }
            # 캐시에 저장
            await cache_manager.set(user_key, user_data, expire=3600)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="사용자를 찾을 수 없습니다"
            )
    
    # 사용자가 비활성화된 경우
    if not user_data.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="비활성화된 계정입니다"
        )
    
    return user_data


# API 엔드포인트들
@router.post("/register", response_model=TokenResponse, summary="사용자 회원가입")
async def register(user_data: UserRegister):
    """
    새 사용자 회원가입 (실제 JWT 토큰 발급)
    
    - **email**: 사용자 이메일 (고유해야 함)
    - **username**: 사용자명 (3자 이상, 영문+숫자)
    - **password**: 비밀번호 (보안 요구사항 충족 필요)
    - **full_name**: 실명 (선택사항)
    """
    logger.info(f"회원가입 시도: {user_data.email}")
    
    # 이메일 중복 확인 (임시 구현)
    existing_user_key = f"email:{user_data.email}"
    existing_user = await cache_manager.get(existing_user_key)
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 등록된 이메일입니다"
        )
    
    # 사용자명 중복 확인
    existing_username_key = f"username:{user_data.username}"
    existing_username = await cache_manager.get(existing_username_key)
    
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 사용 중인 사용자명입니다"
        )
    
    # 비밀번호 해시화
    hashed_password = security_manager.hash_password(user_data.password)
    
    # 새 사용자 ID 생성
    new_user_id = str(hash(user_data.email) % 100000)
    
    # 사용자 정보 생성
    user_info = {
        "id": int(new_user_id),
        "email": user_data.email,
        "username": user_data.username,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
        "is_active": True,
        "is_verified": False,
        "is_premium": False,
        "created_at": "2025-06-22T01:00:00Z"
    }
    
    # 캐시에 저장
    await cache_manager.set(f"user:{new_user_id}", user_info, expire=86400)  # 24시간
    await cache_manager.set(existing_user_key, new_user_id, expire=86400)
    await cache_manager.set(existing_username_key, new_user_id, expire=86400)
    
    # JWT 토큰 생성
    token_data = {"sub": new_user_id, "email": user_data.email, "username": user_data.username}
    access_token = security_manager.create_access_token(token_data)
    refresh_token = security_manager.create_refresh_token(token_data)
    
    logger.info(f"✅ 회원가입 완료: {user_data.email}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800
    )


@router.post("/login", response_model=TokenResponse, summary="사용자 로그인")
async def login(user_data: UserLogin):
    """
    사용자 로그인 (실제 JWT 토큰 발급)
    
    - **email**: 등록된 이메일
    - **password**: 비밀번호
    """
    logger.info(f"로그인 시도: {user_data.email}")
    
    # 사용자 조회
    user_key_lookup = f"email:{user_data.email}"
    user_id = await cache_manager.get(user_key_lookup)
    
    if not user_id:
        # 테스트 계정 허용
        if user_data.email == "test@autoblitz.com" and user_data.password == "Test123!":
            user_id = "1"
            # 임시 사용자 정보 생성
            test_user = {
                "id": 1,
                "email": "test@autoblitz.com",
                "username": "testuser",
                "full_name": "테스트 사용자",
                "hashed_password": security_manager.hash_password("Test123!"),
                "is_active": True,
                "is_verified": True,
                "is_premium": False
            }
            await cache_manager.set(f"user:{user_id}", test_user, expire=86400)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="이메일 또는 비밀번호가 올바르지 않습니다"
            )
    
    # 사용자 정보 조회
    user_info = await cache_manager.get(f"user:{user_id}")
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자 정보를 찾을 수 없습니다"
        )
    
    # 비밀번호 검증
    if not security_manager.verify_password(user_data.password, user_info["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다"
        )
    
    # 계정 활성화 확인
    if not user_info.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="비활성화된 계정입니다"
        )
    
    # JWT 토큰 생성
    token_data = {
        "sub": str(user_info["id"]),
        "email": user_info["email"],
        "username": user_info["username"]
    }
    access_token = security_manager.create_access_token(token_data)
    refresh_token = security_manager.create_refresh_token(token_data)
    
    # 로그인 시간 업데이트
    user_info["last_login"] = "2025-06-22T01:00:00Z"
    await cache_manager.set(f"user:{user_id}", user_info, expire=86400)
    
    logger.info(f"✅ 로그인 성공: {user_data.email}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800
    )


@router.get("/profile", response_model=UserProfile, summary="내 프로필 조회")
async def get_profile(current_user: dict = Depends(verify_token)):
    """현재 로그인한 사용자의 프로필 정보 조회"""
    return UserProfile(**current_user)


@router.post("/logout", summary="로그아웃")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """사용자 로그아웃 (토큰 블랙리스트 등록)"""
    token = credentials.credentials
    
    # 토큰을 블랙리스트에 추가 (토큰 만료시까지 유지)
    blacklist_key = f"blacklist:{token}"
    await cache_manager.set(blacklist_key, "blacklisted", expire=1800)  # 30분
    
    logger.info("✅ 로그아웃 완료")
    
    return {
        "message": "로그아웃이 완료되었습니다",
        "timestamp": "2025-06-22T01:00:00Z"
    }


@router.post("/change-password", summary="비밀번호 변경")
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: dict = Depends(verify_token)
):
    """비밀번호 변경"""
    user_id = current_user["id"]
    
    # 현재 비밀번호 확인
    user_info = await cache_manager.get(f"user:{user_id}")
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="사용자 정보를 찾을 수 없습니다"
        )
    
    # 현재 비밀번호 검증
    if not security_manager.verify_password(password_data.current_password, user_info["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="현재 비밀번호가 올바르지 않습니다"
        )
    
    # 새 비밀번호 해시화
    new_hashed_password = security_manager.hash_password(password_data.new_password)
    
    # 사용자 정보 업데이트
    user_info["hashed_password"] = new_hashed_password
    user_info["password_changed_at"] = "2025-06-22T01:00:00Z"
    
    await cache_manager.set(f"user:{user_id}", user_info, expire=86400)
    
    logger.info(f"✅ 비밀번호 변경: 사용자 {user_id}")
    
    return {
        "message": "비밀번호가 성공적으로 변경되었습니다",
        "timestamp": "2025-06-22T01:00:00Z"
    }


@router.get("/security-info", summary="보안 정보 조회")
async def get_security_info():
    """인증 시스템 보안 정보"""
    return {
        "jwt_algorithm": "HS256",
        "token_expire_minutes": 30,
        "refresh_token_expire_days": 7,
        "password_requirements": {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_special_chars": True
        },
        "security_features": [
            "JWT 토큰 인증",
            "비밀번호 해시화 (bcrypt)",
            "토큰 블랙리스트",
            "비밀번호 강도 검증",
            "자동 토큰 만료"
        ]
    }