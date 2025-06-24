# 오토블리츠(Autoblitz) - AI 암호화폐 자동매매 플랫폼

## 소개
오토블리츠는 5대 글로벌 거래소를 통합 분석하고, AI 기반 코인 추천 및 7단계 물타기 자동매매 전략을 제공하는 SaaS 백엔드 플랫폼입니다.

## 주요 기능
- **자동매매 봇**: 7단계 물타기 전략 기반 자동거래
- **AI 코인 추천**: 5개 거래소 통합 분석 기반 안전한 코인 추천
- **실시간 모니터링**: 100개 코인 실시간 분석 및 위험 감지
- **리스크 관리**: 위험 패턴 자동 감지 및 실시간 알림
- **RESTful API**: FastAPI 기반의 확장성 높은 API 제공

## 지원 거래소
- OKX (SPOT, SWAP, BTC 마켓)
- 업비트 (KRW, USDT 마켓)
- 바이낸스 (USDT 마켓)
<!-- - 코인베이스 (USD 마켓) -->
<!-- - 크라켄 (USD 마켓) -->

## 설치 및 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
python app/main.py --env dev

# 프로덕션 서버 실행
python app/main.py --env prod
```

## API 문서
- [Swagger UI](/docs)
- [ReDoc](/redoc)

### 주요 엔드포인트
- `/api/v1/trading` : 자동매매 봇 관리 및 거래 실행
- `/api/v1/recommendations` : AI 기반 코인 추천 시스템
- `/api/v1/analysis` : 시장 분석 및 신호 생성

## 폴더 구조
- `app/` : 메인 백엔드 애플리케이션
- `app/api/` : API 라우터 모듈
- `app/analysis/` : AI 추천 및 분석 엔진
- `app/strategies/` : 자동매매 전략 및 봇
- `app/core/` : 공통 유틸리티, 인증, DB 등

## 기여 방법
1. 이슈 등록 또는 PR 요청
2. 코드 스타일 및 문서화 준수
3. 주요 변경사항은 CHANGELOG에 기록

## 라이선스
MIT License
