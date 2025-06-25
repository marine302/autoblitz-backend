# app/core/auth.py
"""
간단한 인증 시스템 - 개발용
"""

from fastapi import HTTPException, Header
from typing import Optional, Dict

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict:
    """
    현재 사용자 정보 조회 (개발용 간단 버전)
    실제 운영시에는 JWT 토큰 검증 등이 필요
    """
    # 개발 모드에서는 기본 사용자 반환
    return {
        "user_id": "demo_user",
        "username": "demo",
        "email": "demo@autoblitz.com",
        "is_active": True
    }

def verify_api_key(api_key: Optional[str] = Header(None)) -> bool:
    """
    API 키 검증 (개발용)
    """
    # 개발 모드에서는 항상 True 반환
    return True