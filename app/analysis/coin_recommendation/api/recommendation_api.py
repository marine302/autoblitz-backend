# app/analysis/coin_recommendation/api/recommendation_api.py
"""
코인 추천 API 엔드포인트
- 실시간 추천 조회
- 위험 알림
- 시장 분석 결과
"""

import asyncio
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field

from ..core.data_collector import MultiExchangeDataCollector, CoinData
from ..core.analysis_engine import VolatilityAnalysisEngine
from ..core.scoring_system import AIRecommendationEngine, RecommendationScore

logger = logging.getLogger(__name__)

# 전역 인스턴스 (싱글톤 패턴)
data_collector: Optional[MultiExchangeDataCollector] = None
analysis_engine: Optional[VolatilityAnalysisEngine] = None
recommendation_engine: Optional[AIRecommendationEngine] = None

# 시스템 상태
system_status = {
    'is_running': False,
    'last_update': None,
    'total_coins': 0,
    'data_collection_errors': 0
}

# API 라우터
router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])

# 응답 모델들
class CoinRecommendationResponse(BaseModel):
    symbol: str
    market: str
    final_score: float
    grade: str
    is_recommended: bool
    confidence: float
    current_price: float
    volume_24h_usdt: float
    change_24h: float
    recommendation_reason: str
    risk_warnings: List[str]
    strength_points: List[str]
    last_updated: datetime
    
    # 세부 점수
    volume_score: float
    volatility_score: float
    technical_score: float
    market_condition_score: float
    liquidity_score: float

class MarketSummaryResponse(BaseModel):
    total_analyzed: int
    recommended_count: int
    grade_distribution: Dict[str, int]
    average_score: float
    market_trend: str
    volatility_index: float
    last_update: datetime

class RiskAlertResponse(BaseModel):
    symbol: str
    market: str
    severity: str
    risk_flags: List[str]
    description: str
    safety_score: float
    detected_at: datetime

class SystemStatusResponse(BaseModel):
    status: str
    is_collecting_data: bool
    total_coins_tracked: int
    total_markets: int
    last_update: Optional[datetime]
    data_collection_errors: int
    uptime_seconds: int

# 시작/종료 이벤트
@router.on_event("startup")
async def startup_event():
    """API 시작시 초기화"""
    global data_collector, analysis_engine, recommendation_engine
    
    try:
        logger.info("코인 추천 시스템 초기화 시작")
        
        # 컴포넌트 초기화
        data_collector = MultiExchangeDataCollector()
        analysis_engine = VolatilityAnalysisEngine()
        recommendation_engine = AIRecommendationEngine()
        
        # 저장된 모델 로드
        recommendation_engine.load_model()
        
        # 데이터 수집 시작
        await data_collector.__aenter__()
        
        # 백그라운드 태스크 시작
        asyncio.create_task(background_data_collection())
        asyncio.create_task(background_recommendation_update())
        
        system_status['is_running'] = True
        logger.info("코인 추천 시스템 초기화 완료")
        
    except Exception as e:
        logger.error(f"시스템 초기화 실패: {e}")
        raise

@router.on_event("shutdown")
async def shutdown_event():
    """API 종료시 정리"""
    global data_collector, system_status
    
    try:
        system_status['is_running'] = False
        
        if data_collector:
            data_collector.stop_collection()
            await data_collector.__aexit__(None, None, None)
        
        # 모델 저장
        if recommendation_engine:
            recommendation_engine.save_model()
            
        logger.info("코인 추천 시스템 정상 종료")
        
    except Exception as e:
        logger.error(f"시스템 종료 중 오류: {e}")

# 백그라운드 태스크들
async def background_data_collection():
    """백그라운드 데이터 수집"""
    global data_collector, analysis_engine, system_status
    
    while system_status['is_running']:
        try:
            # 모든 거래소 데이터 수집
            tasks = []
            for exchange in ['okx', 'upbit', 'binance', 'coinbase', 'kraken']:
                task = data_collector._collect_exchange_data(exchange)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_coins = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    system_status['data_collection_errors'] += 1
                    logger.error(f"데이터 수집 오류: {result}")
                else:
                    total_coins += len(result)
                    # 분석 엔진에 데이터 추가
                    for coin_data in result:
                        analysis_engine.add_coin_data(coin_data)
            
            system_status['total_coins'] = total_coins
            system_status['last_update'] = datetime.now()
            
            logger.info(f"데이터 수집 완료: {total_coins}개 코인")
            
        except Exception as e:
            logger.error(f"백그라운드 데이터 수집 오류: {e}")
            system_status['data_collection_errors'] += 1
        
        # 1분 대기
        await asyncio.sleep(60)

async def background_recommendation_update():
    """백그라운드 추천 업데이트"""
    global data_collector, recommendation_engine, system_status
    
    while system_status['is_running']:
        try:
            # 5분마다 추천 업데이트
            await asyncio.sleep(300)
            
            if not data_collector:
                continue
            
            # 현재 코인 데이터 가져오기
            all_coins = data_collector.get_all_coins()
            
            if all_coins:
                # 배치 추천 생성
                await recommendation_engine.generate_batch_recommendations(all_coins)
                logger.info(f"추천 업데이트 완료: {len(all_coins)}개 코인")
            
        except Exception as e:
            logger.error(f"백그라운드 추천 업데이트 오류: {e}")

# API 엔드포인트들

@router.get("/", response_model=List[CoinRecommendationResponse])
async def get_recommendations(
    market: Optional[str] = Query(None, description="마켓 필터 (예: OKX_USDT_SPOT)"),
    grade: Optional[str] = Query("B", description="최소 등급 (A+, A, B, C, D)"),
    limit: int = Query(50, ge=1, le=200, description="결과 수 제한"),
    only_recommended: bool = Query(True, description="추천 코인만 표시")
):
    """추천 코인 목록 조회"""
    
    if not recommendation_engine:
        raise HTTPException(status_code=503, detail="추천 엔진이 초기화되지 않음")
    
    try:
        # 추천 코인 조회
        recommendations = recommendation_engine.get_top_recommendations(
            limit=limit,
            market_filter=market,
            min_grade=grade
        )
        
        if only_recommended:
            recommendations = [r for r in recommendations if r.is_recommended]
        
        # 응답 데이터 변환
        response_data = []
        for rec in recommendations:
            # 현재 가격 데이터 가져오기
            coin_data = data_collector.get_coin_data(rec.symbol, rec.market) if data_collector else None
            
            response_data.append(CoinRecommendationResponse(
                symbol=rec.symbol,
                market=rec.market,
                final_score=rec.final_score,
                grade=rec.grade,
                is_recommended=rec.is_recommended,
                confidence=rec.confidence,
                current_price=coin_data.price if coin_data else 0.0,
                volume_24h_usdt=coin_data.volume_usdt if coin_data else 0.0,
                change_24h=coin_data.change_24h if coin_data else 0.0,
                recommendation_reason=rec.recommendation_reason,
                risk_warnings=rec.risk_warnings,
                strength_points=rec.strength_points,
                last_updated=rec.last_updated,
                volume_score=rec.volume_score,
                volatility_score=rec.volatility_score,
                technical_score=rec.technical_score,
                market_condition_score=rec.market_condition_score,
                liquidity_score=rec.liquidity_score
            ))
        
        return response_data
        
    except Exception as e:
        logger.error(f"추천 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary", response_model=MarketSummaryResponse)
async def get_market_summary():
    """시장 요약 정보"""
    
    if not recommendation_engine:
        raise HTTPException(status_code=503, detail="추천 엔진이 초기화되지 않음")
    
    try:
        summary = recommendation_engine.get_market_summary()
        
        return MarketSummaryResponse(
            total_analyzed=summary['total_analyzed'],
            recommended_count=summary['recommended_count'],
            grade_distribution=summary['grade_distribution'],
            average_score=summary['average_score'],
            market_trend=summary['market_context']['trend'],
            volatility_index=summary['market_context']['volatility_index'],
            last_update=summary['last_update']
        )
        
    except Exception as e:
        logger.error(f"시장 요약 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts", response_model=List[RiskAlertResponse])
async def get_risk_alerts(
    severity: Optional[str] = Query(None, description="위험도 필터 (HIGH, CRITICAL)"),
    limit: int = Query(20, ge=1, le=100, description="결과 수 제한")
):
    """위험 알림 목록"""
    
    if not analysis_engine:
        raise HTTPException(status_code=503, detail="분석 엔진이 초기화되지 않음")
    
    try:
        alerts = analysis_engine.get_risk_alerts(severity_filter=severity)
        
        response_data = []
        for alert in alerts[:limit]:
            response_data.append(RiskAlertResponse(
                symbol=alert['symbol'],
                market=alert['market'],
                severity=alert['severity'],
                risk_flags=alert['risk_flags'],
                description=alert['description'],
                safety_score=alert['safety_score'],
                detected_at=alert['detected_at']
            ))
        
        return response_data
        
    except Exception as e:
        logger.error(f"위험 알림 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coin/{symbol}", response_model=CoinRecommendationResponse)
async def get_coin_recommendation(
    symbol: str,
    market: Optional[str] = Query(None, description="특정 마켓 (없으면 모든 마켓 검색)")
):
    """특정 코인 추천 정보"""
    
    if not recommendation_engine or not data_collector:
        raise HTTPException(status_code=503, detail="시스템이 초기화되지 않음")
    
    try:
        # 코인 데이터 검색
        coin_data = None
        if market:
            coin_data = data_collector.get_coin_data(symbol, market)
        else:
            # 모든 마켓에서 검색
            all_coins = data_collector.get_all_coins()
            for coin in all_coins:
                if coin.symbol.upper() == symbol.upper():
                    coin_data = coin
                    break
        
        if not coin_data:
            raise HTTPException(status_code=404, detail=f"코인 {symbol} 데이터를 찾을 수 없음")
        
        # 추천 정보 검색
        key = f"{coin_data.symbol}:{coin_data.market}"
        recommendation = recommendation_engine.current_recommendations.get(key)
        
        if not recommendation:
            # 즉시 분석 수행
            recommendation = await recommendation_engine.generate_recommendation(coin_data)
        
        return CoinRecommendationResponse(
            symbol=recommendation.symbol,
            market=recommendation.market,
            final_score=recommendation.final_score,
            grade=recommendation.grade,
            is_recommended=recommendation.is_recommended,
            confidence=recommendation.confidence,
            current_price=coin_data.price,
            volume_24h_usdt=coin_data.volume_usdt,
            change_24h=coin_data.change_24h,
            recommendation_reason=recommendation.recommendation_reason,
            risk_warnings=recommendation.risk_warnings,
            strength_points=recommendation.strength_points,
            last_updated=recommendation.last_updated,
            volume_score=recommendation.volume_score,
            volatility_score=recommendation.volatility_score,
            technical_score=recommendation.technical_score,
            market_condition_score=recommendation.market_condition_score,
            liquidity_score=recommendation.liquidity_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"코인 추천 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/markets", response_model=Dict[str, int])
async def get_available_markets():
    """사용 가능한 마켓 목록"""
    
    if not data_collector:
        raise HTTPException(status_code=503, detail="데이터 수집기가 초기화되지 않음")
    
    try:
        markets = data_collector.get_market_summary()
        return markets
        
    except Exception as e:
        logger.error(f"마켓 정보 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """시스템 상태 확인"""
    
    try:
        uptime = 0
        if system_status['last_update']:
            uptime = int((datetime.now() - system_status['last_update']).total_seconds())
        
        # 시스템 상태 판단
        status = "healthy"
        if not system_status['is_running']:
            status = "stopped"
        elif system_status['data_collection_errors'] > 10:
            status = "degraded"
        elif not system_status['last_update'] or \
             (datetime.now() - system_status['last_update']).total_seconds() > 300:
            status = "stale"
        
        return SystemStatusResponse(
            status=status,
            is_collecting_data=system_status['is_running'],
            total_coins_tracked=system_status['total_coins'],
            total_markets=len(data_collector.get_market_summary()) if data_collector else 0,
            last_update=system_status['last_update'],
            data_collection_errors=system_status['data_collection_errors'],
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"시스템 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/{symbol}")
async def force_analyze_coin(
    symbol: str,
    market: Optional[str] = Query(None, description="특정 마켓"),
    background_tasks: BackgroundTasks = None
):
    """특정 코인 강제 분석"""
    
    if not recommendation_engine or not data_collector:
        raise HTTPException(status_code=503, detail="시스템이 초기화되지 않음")
    
    try:
        # 코인 데이터 검색
        coin_data = None
        if market:
            coin_data = data_collector.get_coin_data(symbol, market)
        else:
            all_coins = data_collector.get_all_coins()
            for coin in all_coins:
                if coin.symbol.upper() == symbol.upper():
                    coin_data = coin
                    break
        
        if not coin_data:
            raise HTTPException(status_code=404, detail=f"코인 {symbol} 데이터를 찾을 수 없음")
        
        # 백그라운드에서 분석 수행
        if background_tasks:
            background_tasks.add_task(
                recommendation_engine.generate_recommendation, 
                coin_data
            )
            return {"message": f"{symbol} 분석이 백그라운드에서 시작됨"}
        else:
            # 즉시 분석
            recommendation = await recommendation_engine.generate_recommendation(coin_data)
            return {
                "message": f"{symbol} 분석 완료",
                "score": recommendation.final_score,
                "grade": recommendation.grade
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"강제 분석 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refresh")
async def refresh_all_data(background_tasks: BackgroundTasks):
    """모든 데이터 강제 새로고침"""
    
    if not system_status['is_running']:
        raise HTTPException(status_code=503, detail="시스템이 실행 중이 아님")
    
    try:
        # 백그라운드에서 전체 데이터 수집 및 분석
        background_tasks.add_task(force_refresh_data)
        
        return {
            "message": "전체 데이터 새로고침이 백그라운드에서 시작됨",
            "estimated_completion": "약 2-3분 소요 예상"
        }
        
    except Exception as e:
        logger.error(f"데이터 새로고침 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def force_refresh_data():
    """강제 데이터 새로고침 (백그라운드 태스크)"""
    global data_collector, recommendation_engine
    
    try:
        logger.info("강제 데이터 새로고침 시작")
        
        # 1. 모든 거래소 데이터 수집
        tasks = []
        for exchange in ['okx', 'upbit', 'binance', 'coinbase', 'kraken']:
            task = data_collector._collect_exchange_data(exchange)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_coins = []
        for result in results:
            if not isinstance(result, Exception):
                all_coins.extend(result)
        
        # 2. 분석 엔진에 데이터 추가
        for coin_data in all_coins:
            analysis_engine.add_coin_data(coin_data)
        
        # 3. 배치 추천 생성
        if all_coins:
            await recommendation_engine.generate_batch_recommendations(all_coins)
        
        logger.info(f"강제 데이터 새로고침 완료: {len(all_coins)}개 코인")
        
    except Exception as e:
        logger.error(f"강제 데이터 새로고침 실패: {e}")

# 헬스체크 엔드포인트
@router.get("/health")
async def health_check():
    """간단한 헬스체크"""
    
    health_status = {
        "status": "healthy" if system_status['is_running'] else "unhealthy",
        "timestamp": datetime.now(),
        "components": {
            "data_collector": data_collector is not None,
            "analysis_engine": analysis_engine is not None,
            "recommendation_engine": recommendation_engine is not None
        }
    }
    
    if not system_status['is_running']:
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

# 테스트 엔드포인트
@router.get("/test")
async def test_recommendation():
    """테스트용 추천 생성"""
    
    if not recommendation_engine:
        raise HTTPException(status_code=503, detail="추천 엔진이 초기화되지 않음")
    
    try:
        # 테스트 데이터 생성
        test_coin = CoinData(
            symbol='BTC',
            market='TEST_MARKET',
            price=43250.0,
            volume_24h=50000,
            volume_usdt=2150000000,
            change_1h=0.5,
            change_24h=2.3,
            high_24h=43800,
            low_24h=42900,
            timestamp=datetime.now()
        )
        
        # 추천 생성
        recommendation = await recommendation_engine.generate_recommendation(test_coin)
        
        return {
            "message": "테스트 추천 생성 성공",
            "symbol": recommendation.symbol,
            "score": recommendation.final_score,
            "grade": recommendation.grade,
            "recommended": recommendation.is_recommended,
            "reason": recommendation.recommendation_reason
        }
        
    except Exception as e:
        logger.error(f"테스트 추천 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))