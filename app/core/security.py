# 파일: app/core/security.py
# 경로: /workspaces/autoblitz-backend/app/core/security.py

"""
오토블리츠 보안 모듈
JWT, 암호화, 보안 유틸리티
"""
from datetime import datetime, timedelta
from typing import Optional, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets
import hashlib
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# 암호화 컨텍스트
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT 설정
ALGORITHM = settings.jwt_algorithm
SECRET_KEY = settings.jwt_secret_key


class SecurityManager:
    """보안 관리자 클래스"""
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        JWT 액세스 토큰 생성
        
        Args:
            data: 토큰에 포함할 데이터
            expires_delta: 만료 시간 (기본 30분)
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        
        try:
            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            logger.debug(f"✅ JWT 토큰 생성: {data.get('sub', 'unknown')}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"❌ JWT 토큰 생성 실패: {e}")
            raise
    
    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """
        JWT 리프레시 토큰 생성 (7일 만료)
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        try:
            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            logger.debug(f"✅ 리프레시 토큰 생성: {data.get('sub', 'unknown')}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"❌ 리프레시 토큰 생성 실패: {e}")
            raise
    
    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        """
        JWT 토큰 검증 및 페이로드 반환
        
        Args:
            token: JWT 토큰 문자열
            
        Returns:
            토큰 페이로드 또는 None
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # 만료 시간 확인
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                logger.warning("⚠️ 만료된 토큰")
                return None
            
            logger.debug(f"✅ 토큰 검증 성공: {payload.get('sub', 'unknown')}")
            return payload
            
        except JWTError as e:
            logger.warning(f"⚠️ JWT 토큰 검증 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ 토큰 검증 오류: {e}")
            return None
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        비밀번호 해시화
        
        Args:
            password: 평문 비밀번호
            
        Returns:
            해시된 비밀번호
        """
        try:
            hashed = pwd_context.hash(password)
            logger.debug("✅ 비밀번호 해시화 완료")
            return hashed
        except Exception as e:
            logger.error(f"❌ 비밀번호 해시화 실패: {e}")
            raise
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        비밀번호 검증
        
        Args:
            plain_password: 평문 비밀번호
            hashed_password: 해시된 비밀번호
            
        Returns:
            검증 성공 여부
        """
        try:
            result = pwd_context.verify(plain_password, hashed_password)
            if result:
                logger.debug("✅ 비밀번호 검증 성공")
            else:
                logger.debug("⚠️ 비밀번호 검증 실패")
            return result
        except Exception as e:
            logger.error(f"❌ 비밀번호 검증 오류: {e}")
            return False
    
    @staticmethod
    def generate_api_key() -> str:
        """
        안전한 API 키 생성
        
        Returns:
            32바이트 헥스 API 키
        """
        api_key = secrets.token_hex(32)
        logger.debug("✅ API 키 생성 완료")
        return api_key
    
    @staticmethod
    def encrypt_api_credentials(api_key: str, secret_key: str, passphrase: Optional[str] = None) -> dict:
        """
        거래소 API 자격증명 암호화 (간단한 해시 방식)
        실제 프로덕션에서는 AES 암호화 사용 권장
        
        Args:
            api_key: API 키
            secret_key: 시크릿 키
            passphrase: 패스프레이즈 (선택사항)
            
        Returns:
            암호화된 자격증명
        """
        try:
            # 단순 해시 암호화 (데모용)
            salt = secrets.token_hex(16)
            
            encrypted_api_key = hashlib.sha256((api_key + salt).encode()).hexdigest()
            encrypted_secret = hashlib.sha256((secret_key + salt).encode()).hexdigest()
            encrypted_passphrase = None
            
            if passphrase:
                encrypted_passphrase = hashlib.sha256((passphrase + salt).encode()).hexdigest()
            
            logger.debug("✅ API 자격증명 암호화 완료")
            
            return {
                "encrypted_api_key": encrypted_api_key,
                "encrypted_secret": encrypted_secret,
                "encrypted_passphrase": encrypted_passphrase,
                "salt": salt,
                "encryption_method": "sha256_demo"  # 실제로는 AES 사용
            }
            
        except Exception as e:
            logger.error(f"❌ API 자격증명 암호화 실패: {e}")
            raise
    
    @staticmethod
    def mask_sensitive_data(data: str, show_length: int = 4) -> str:
        """
        민감한 데이터 마스킹
        
        Args:
            data: 마스킹할 데이터
            show_length: 앞뒤로 보여줄 길이
            
        Returns:
            마스킹된 데이터
        """
        if not data or len(data) <= show_length * 2:
            return "***"
        
        return f"{data[:show_length]}{'*' * (len(data) - show_length * 2)}{data[-show_length:]}"
    
    @staticmethod
    def validate_password_strength(password: str) -> dict:
        """
        비밀번호 강도 검증
        
        Args:
            password: 검증할 비밀번호
            
        Returns:
            검증 결과
        """
        result = {
            "is_valid": True,
            "score": 0,
            "errors": []
        }
        
        # 길이 검사
        if len(password) < 8:
            result["errors"].append("비밀번호는 최소 8자 이상이어야 합니다")
            result["is_valid"] = False
        else:
            result["score"] += 1
        
        # 대소문자 검사
        if not any(c.isupper() for c in password):
            result["errors"].append("대문자를 포함해야 합니다")
            result["is_valid"] = False
        else:
            result["score"] += 1
        
        if not any(c.islower() for c in password):
            result["errors"].append("소문자를 포함해야 합니다")
            result["is_valid"] = False
        else:
            result["score"] += 1
        
        # 숫자 검사
        if not any(c.isdigit() for c in password):
            result["errors"].append("숫자를 포함해야 합니다")
            result["is_valid"] = False
        else:
            result["score"] += 1
        
        # 특수문자 검사
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            result["errors"].append("특수문자를 포함해야 합니다")
            result["is_valid"] = False
        else:
            result["score"] += 1
        
        # 강도 레벨 설정
        if result["score"] >= 4:
            result["strength"] = "강함"
        elif result["score"] >= 3:
            result["strength"] = "보통"
        else:
            result["strength"] = "약함"
        
        return result


# 전역 보안 매니저 인스턴스
security_manager = SecurityManager()