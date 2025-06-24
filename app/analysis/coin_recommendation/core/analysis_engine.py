# app/analysis/coin_recommendation/core/analysis_engine.py
"""
거래량/변동성 분석 엔진
- 실시간 거래량 분석 및 비정상 패턴 감지
- 변동성 기반 리스크 평가
- 안전성 스코어 계산
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from collections import deque
import statistics

from .data_collector import CoinData

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """분석 결과 구조체"""
    symbol: str
    market: str
    
    # 거래량 분석
    volume_score: float  # 0-100
    volume_ratio: float  # 평균 대비 거래량 비율
    volume_trend: str    # 'increasing', 'decreasing', 'stable'
    
    # 변동성 분석
    volatility_score: float  # 0-100 (높을수록 안정)
    price_volatility_1h: float
    price_volatility_24h: float
    atr_ratio: float
    
    # 위험 요소
    risk_flags: List[str]
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    
    # 종합 점수
    safety_score: float  # 0-100
    recommendation_grade: str  # 'A+', 'A', 'B', 'C', 'D'
    
    timestamp: datetime

class VolatilityAnalysisEngine:
    """변동성 분석 엔진"""
    
    def __init__(self, history_window: int = 1440):  # 24시간 = 1440분
        self.history_window = history_window
        
        # 코인별 가격 히스토리 (최근 24시간)
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        
        # 분석 결과 캐시
        self.analysis_cache: Dict[str, AnalysisResult] = {}
        
        # 시장 전체 통계
        self.market_stats = {
            'total_volume': 0,
            'average_volatility': 0,
            'active_coins': 0,
            'last_update': None
        }

    def add_coin_data(self, coin_data: CoinData):
        """코인 데이터 추가 및 히스토리 관리"""
        key = f"{coin_data.symbol}:{coin_data.market}"
        
        # 가격 히스토리 관리
        if key not in self.price_history:
            self.price_history[key] = deque(maxlen=self.history_window)
            self.volume_history[key] = deque(maxlen=self.history_window)
        
        # 새 데이터 추가
        self.price_history[key].append({
            'price': coin_data.price,
            'timestamp': coin_data.timestamp,
            'high': coin_data.high_24h,
            'low': coin_data.low_24h
        })
        
        self.volume_history[key].append({
            'volume': coin_data.volume_usdt,
            'timestamp': coin_data.timestamp
        })

    def analyze_coin(self, coin_data: CoinData) -> AnalysisResult:
        """개별 코인 종합 분석"""
        key = f"{coin_data.symbol}:{coin_data.market}"
        
        # 히스토리 데이터 확인
        if key not in self.price_history or len(self.price_history[key]) < 10:
            # 데이터 부족시 기본 분석
            return self._basic_analysis(coin_data)
        
        # 상세 분석 수행
        volume_analysis = self._analyze_volume(key, coin_data)
        volatility_analysis = self._analyze_volatility(key, coin_data)
        risk_analysis = self._analyze_risk_factors(key, coin_data)
        
        # 종합 점수 계산
        safety_score = self._calculate_safety_score(
            volume_analysis, volatility_analysis, risk_analysis
        )
        
        # 추천 등급 결정
        grade = self._determine_grade(safety_score, risk_analysis['risk_level'])
        
        result = AnalysisResult(
            symbol=coin_data.symbol,
            market=coin_data.market,
            volume_score=volume_analysis['score'],
            volume_ratio=volume_analysis['ratio'],
            volume_trend=volume_analysis['trend'],
            volatility_score=volatility_analysis['score'],
            price_volatility_1h=volatility_analysis['volatility_1h'],
            price_volatility_24h=volatility_analysis['volatility_24h'],
            atr_ratio=volatility_analysis['atr_ratio'],
            risk_flags=risk_analysis['flags'],
            risk_level=risk_analysis['risk_level'],
            safety_score=safety_score,
            recommendation_grade=grade,
            timestamp=datetime.now()
        )
        
        # 결과 캐시
        self.analysis_cache[key] = result
        
        return result

    def _analyze_volume(self, key: str, coin_data: CoinData) -> Dict:
        """거래량 분석"""
        volume_data = list(self.volume_history[key])
        
        if len(volume_data) < 10:
            return {
                'score': 50.0,
                'ratio': 1.0,
                'trend': 'unknown'
            }
        
        # 현재 거래량
        current_volume = coin_data.volume_usdt
        
        # 최근 24시간 평균 거래량
        recent_volumes = [v['volume'] for v in volume_data[-24:]]
        avg_volume = statistics.mean(recent_volumes) if recent_volumes else 1
        
        # 거래량 비율 계산
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # 거래량 트렌드 분석
        if len(recent_volumes) >= 12:
            first_half = statistics.mean(recent_volumes[:12])
            second_half = statistics.mean(recent_volumes[12:])
            
            if second_half > first_half * 1.2:
                trend = 'increasing'
            elif second_half < first_half * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        # 거래량 점수 계산 (적정 거래량 = 높은 점수)
        if volume_ratio >= 0.5 and volume_ratio <= 3.0:
            # 적정 범위: 평균의 50%~300%
            base_score = 90
            if volume_ratio > 2.0:
                base_score -= (volume_ratio - 2.0) * 20  # 너무 많으면 감점
        elif volume_ratio >= 0.2:
            # 낮은 거래량
            base_score = 60 - (0.5 - volume_ratio) * 100
        else:
            # 매우 낮은 거래량
            base_score = 20
        
        # 최소값 보장
        volume_score = max(0, min(100, base_score))
        
        return {
            'score': volume_score,
            'ratio': volume_ratio,
            'trend': trend
        }

    def _analyze_volatility(self, key: str, coin_data: CoinData) -> Dict:
        """변동성 분석"""
        price_data = list(self.price_history[key])
        
        if len(price_data) < 10:
            return {
                'score': 50.0,
                'volatility_1h': 0.0,
                'volatility_24h': abs(coin_data.change_24h),
                'atr_ratio': 1.0
            }
        
        # 최근 가격들
        prices = [p['price'] for p in price_data]
        
        # 1시간 변동성 계산 (최근 60분)
        if len(prices) >= 60:
            recent_1h = prices[-60:]
            volatility_1h = (max(recent_1h) - min(recent_1h)) / statistics.mean(recent_1h) * 100
        else:
            volatility_1h = abs(coin_data.change_24h / 24)  # 근사치
        
        # 24시간 변동성
        volatility_24h = abs(coin_data.change_24h)
        
        # ATR (Average True Range) 계산
        atr_values = []
        for i in range(1, min(len(price_data), 15)):  # 최근 14개 데이터
            current = price_data[-i]
            previous = price_data[-i-1]
            
            tr = max(
                current['price'] - previous['price'],
                abs(current['price'] - previous['price']),
                abs(previous['price'] - current['price'])
            )
            atr_values.append(tr)
        
        atr = statistics.mean(atr_values) if atr_values else 0
        atr_ratio = atr / coin_data.price if coin_data.price > 0 else 0
        
        # 변동성 점수 계산 (낮은 변동성 = 높은 점수)
        volatility_score = 100
        
        # 1시간 변동성 페널티
        if volatility_1h > 10:
            volatility_score -= (volatility_1h - 10) * 5
        
        # 24시간 변동성 페널티
        if volatility_24h > 20:
            volatility_score -= (volatility_24h - 20) * 2
        
        # ATR 페널티
        if atr_ratio > 0.05:  # 5% 이상
            volatility_score -= (atr_ratio - 0.05) * 500
        
        # 최소값 보장
        volatility_score = max(0, min(100, volatility_score))
        
        return {
            'score': volatility_score,
            'volatility_1h': volatility_1h,
            'volatility_24h': volatility_24h,
            'atr_ratio': atr_ratio
        }

    def _analyze_risk_factors(self, key: str, coin_data: CoinData) -> Dict:
        """위험 요소 분석"""
        risk_flags = []
        risk_level = 'LOW'
        
        # 급격한 가격 변동 감지
        if abs(coin_data.change_24h) > 30:
            risk_flags.append('HIGH_VOLATILITY')
            risk_level = 'HIGH'
        elif abs(coin_data.change_24h) > 20:
            risk_flags.append('MEDIUM_VOLATILITY')
            if risk_level == 'LOW':
                risk_level = 'MEDIUM'
        
        # 거래량 이상 패턴 감지
        volume_data = list(self.volume_history[key])
        if len(volume_data) >= 24:
            recent_volumes = [v['volume'] for v in volume_data[-24:]]
            avg_volume = statistics.mean(recent_volumes)
            current_volume = coin_data.volume_usdt
            
            # 거래량 급증
            if current_volume > avg_volume * 10:
                risk_flags.append('VOLUME_SPIKE')
                risk_level = 'HIGH'
            elif current_volume > avg_volume * 5:
                risk_flags.append('HIGH_VOLUME')
                if risk_level == 'LOW':
                    risk_level = 'MEDIUM'
            
            # 거래량 급감
            elif current_volume < avg_volume * 0.1:
                risk_flags.append('LOW_LIQUIDITY')
                if risk_level == 'LOW':
                    risk_level = 'MEDIUM'
        
        # Pump & Dump 패턴 감지
        price_data = list(self.price_history[key])
        if len(price_data) >= 60:  # 최근 1시간 데이터
            recent_prices = [p['price'] for p in price_data[-60:]]
            price_changes = []
            
            for i in range(1, len(recent_prices)):
                change = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] * 100
                price_changes.append(change)
            
            # 급등 후 급락 패턴
            max_increase = max(price_changes) if price_changes else 0
            min_decrease = min(price_changes) if price_changes else 0
            
            if max_increase > 15 and min_decrease < -10:
                risk_flags.append('PUMP_DUMP_PATTERN')
                risk_level = 'CRITICAL'
        
        # 가격 대비 거래량 비정상
        if coin_data.volume_usdt > 0 and coin_data.price > 0:
            volume_price_ratio = coin_data.volume_usdt / coin_data.price
            
            # 이 비율이 비정상적으로 높으면 조작 의심
            if volume_price_ratio > 1000000:  # 임계값 (조정 필요)
                risk_flags.append('MANIPULATION_SUSPECTED')
                risk_level = 'HIGH'
        
        # 신규 상장 코인 (데이터 부족)
        if len(price_data) < 72:  # 3시간 미만 데이터
            risk_flags.append('INSUFFICIENT_DATA')
            if risk_level == 'LOW':
                risk_level = 'MEDIUM'
        
        return {
            'flags': risk_flags,
            'risk_level': risk_level
        }

    def _calculate_safety_score(self, volume_analysis: Dict, 
                               volatility_analysis: Dict, 
                               risk_analysis: Dict) -> float:
        """종합 안전성 점수 계산"""
        
        # 기본 점수 (가중 평균)
        base_score = (
            volume_analysis['score'] * 0.4 +      # 거래량 40%
            volatility_analysis['score'] * 0.6    # 변동성 60%
        )
        
        # 위험 요소 페널티
        risk_penalty = 0
        for flag in risk_analysis['flags']:
            if flag == 'PUMP_DUMP_PATTERN':
                risk_penalty += 30
            elif flag == 'HIGH_VOLATILITY':
                risk_penalty += 20
            elif flag == 'VOLUME_SPIKE':
                risk_penalty += 15
            elif flag == 'MANIPULATION_SUSPECTED':
                risk_penalty += 25
            elif flag == 'LOW_LIQUIDITY':
                risk_penalty += 10
            elif flag in ['MEDIUM_VOLATILITY', 'HIGH_VOLUME']:
                risk_penalty += 5
            elif flag == 'INSUFFICIENT_DATA':
                risk_penalty += 3
        
        # 최종 점수
        final_score = max(0, min(100, base_score - risk_penalty))
        
        return final_score

    def _determine_grade(self, safety_score: float, risk_level: str) -> str:
        """추천 등급 결정"""
        
        # 위험 레벨이 CRITICAL이면 무조건 D
        if risk_level == 'CRITICAL':
            return 'D'
        
        # 점수 기반 등급
        if safety_score >= 90 and risk_level == 'LOW':
            return 'A+'
        elif safety_score >= 80 and risk_level in ['LOW', 'MEDIUM']:
            return 'A'
        elif safety_score >= 70:
            return 'B'
        elif safety_score >= 60:
            return 'C'
        else:
            return 'D'

    def _basic_analysis(self, coin_data: CoinData) -> AnalysisResult:
        """데이터 부족시 기본 분석"""
        
        # 기본적인 24시간 변동성만으로 판단
        volatility_24h = abs(coin_data.change_24h)
        
        if volatility_24h > 30:
            risk_level = 'HIGH'
            safety_score = 30
            grade = 'D'
        elif volatility_24h > 20:
            risk_level = 'MEDIUM'
            safety_score = 50
            grade = 'C'
        elif volatility_24h > 10:
            risk_level = 'MEDIUM'
            safety_score = 70
            grade = 'B'
        else:
            risk_level = 'LOW'
            safety_score = 80
            grade = 'A'
        
        return AnalysisResult(
            symbol=coin_data.symbol,
            market=coin_data.market,
            volume_score=50.0,  # 기본값
            volume_ratio=1.0,
            volume_trend='unknown',
            volatility_score=max(0, 100 - volatility_24h * 3),
            price_volatility_1h=volatility_24h / 24,
            price_volatility_24h=volatility_24h,
            atr_ratio=0.0,
            risk_flags=['INSUFFICIENT_DATA'],
            risk_level=risk_level,
            safety_score=safety_score,
            recommendation_grade=grade,
            timestamp=datetime.now()
        )

    def get_market_overview(self) -> Dict:
        """전체 시장 개요"""
        if not self.analysis_cache:
            return {
                'total_coins': 0,
                'recommended_coins': 0,
                'average_safety_score': 0,
                'grade_distribution': {},
                'risk_distribution': {}
            }
        
        results = list(self.analysis_cache.values())
        
        # 등급별 분포
        grade_dist = {}
        for result in results:
            grade = result.recommendation_grade
            grade_dist[grade] = grade_dist.get(grade, 0) + 1
        
        # 위험도별 분포
        risk_dist = {}
        for result in results:
            risk = result.risk_level
            risk_dist[risk] = risk_dist.get(risk, 0) + 1
        
        # 추천 가능한 코인 (B등급 이상)
        recommended = len([r for r in results if r.recommendation_grade in ['A+', 'A', 'B']])
        
        return {
            'total_coins': len(results),
            'recommended_coins': recommended,
            'average_safety_score': statistics.mean([r.safety_score for r in results]),
            'grade_distribution': grade_dist,
            'risk_distribution': risk_dist,
            'last_update': max([r.timestamp for r in results]) if results else None
        }

    def get_recommended_coins(self, min_grade: str = 'B', 
                            market_filter: str = None, 
                            limit: int = 50) -> List[AnalysisResult]:
        """추천 코인 목록 조회"""
        
        grade_order = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
        min_grade_value = grade_order.get(min_grade, 3)
        
        results = []
        for result in self.analysis_cache.values():
            # 등급 필터
            if grade_order.get(result.recommendation_grade, 0) < min_grade_value:
                continue
            
            # 마켓 필터
            if market_filter and result.market != market_filter:
                continue
            
            results.append(result)
        
        # 안전성 점수 기준 정렬
        results.sort(key=lambda x: x.safety_score, reverse=True)
        
        return results[:limit]

    def get_risk_alerts(self, severity_filter: str = None) -> List[Dict]:
        """위험 알림 목록"""
        alerts = []
        
        for result in self.analysis_cache.values():
            if result.risk_level in ['HIGH', 'CRITICAL']:
                severity = 'HIGH' if result.risk_level == 'HIGH' else 'CRITICAL'
                
                if severity_filter and severity != severity_filter:
                    continue
                
                # 주요 위험 요소 설명
                descriptions = {
                    'HIGH_VOLATILITY': f"24시간 변동률 {result.price_volatility_24h:.1f}% 위험 수준",
                    'PUMP_DUMP_PATTERN': "급등 후 급락 패턴 감지",
                    'VOLUME_SPIKE': f"비정상적 거래량 급증 (평균 대비 {result.volume_ratio:.1f}배)",
                    'MANIPULATION_SUSPECTED': "시장 조작 의심 패턴 감지",
                    'LOW_LIQUIDITY': "유동성 부족으로 인한 위험"
                }
                
                main_risk = result.risk_flags[0] if result.risk_flags else 'UNKNOWN'
                description = descriptions.get(main_risk, "위험 요소 감지")
                
                alerts.append({
                    'symbol': result.symbol,
                    'market': result.market,
                    'severity': severity,
                    'risk_flags': result.risk_flags,
                    'description': description,
                    'safety_score': result.safety_score,
                    'detected_at': result.timestamp
                })
        
        # 위험도 기준 정렬
        alerts.sort(key=lambda x: (x['severity'] == 'CRITICAL', -x['safety_score']))
        
        return alerts


# 사용 예시 및 테스트
async def test_analysis_engine():
    """분석 엔진 테스트"""
    from .data_collector import MultiExchangeDataCollector
    
    # 분석 엔진 초기화
    engine = VolatilityAnalysisEngine()
    
    # 테스트 데이터 생성
    test_coins = [
        CoinData(
            symbol='BTC',
            market='OKX_USDT_SPOT', 
            price=43250.0,
            volume_24h=50000,
            volume_usdt=2150000000,
            change_1h=0.5,
            change_24h=2.3,
            high_24h=43800,
            low_24h=42900,
            timestamp=datetime.now()
        ),
        CoinData(
            symbol='ETH',
            market='OKX_USDT_SPOT',
            price=2650.0, 
            volume_24h=80000,
            volume_usdt=2120000000,
            change_1h=-0.2,
            change_24h=1.8,
            high_24h=2680,
            low_24h=2620,
            timestamp=datetime.now()
        ),
        CoinData(
            symbol='SHIB',
            market='OKX_USDT_SPOT',
            price=0.000025,
            volume_24h=1000000000,
            volume_usdt=25000000,
            change_1h=15.2,
            change_24h=45.7,  # 높은 변동성
            high_24h=0.000030,
            low_24h=0.000018,
            timestamp=datetime.now()
        )
    ]
    
    # 분석 실행
    for coin in test_coins:
        # 히스토리 데이터 시뮬레이션
        for i in range(100):
            engine.add_coin_data(coin)
        
        # 분석 수행
        result = engine.analyze_coin(coin)
        
        print(f"\n=== {coin.symbol} 분석 결과 ===")
        print(f"추천 등급: {result.recommendation_grade}")
        print(f"안전성 점수: {result.safety_score:.1f}")
        print(f"위험 레벨: {result.risk_level}")
        print(f"거래량 점수: {result.volume_score:.1f}")
        print(f"변동성 점수: {result.volatility_score:.1f}")
        print(f"위험 요소: {', '.join(result.risk_flags)}")
    
    # 시장 개요
    overview = engine.get_market_overview()
    print(f"\n=== 시장 개요 ===")
    print(f"분석된 코인 수: {overview['total_coins']}")
    print(f"추천 코인 수: {overview['recommended_coins']}")
    print(f"평균 안전성 점수: {overview['average_safety_score']:.1f}")
    print(f"등급별 분포: {overview['grade_distribution']}")
    
    # 추천 코인 목록
    recommended = engine.get_recommended_coins(min_grade='B', limit=10)
    print(f"\n=== 추천 코인 (B등급 이상) ===")
    for coin in recommended:
        print(f"{coin.symbol} ({coin.market}): {coin.recommendation_grade} - {coin.safety_score:.1f}점")
    
    # 위험 알림
    alerts = engine.get_risk_alerts()
    if alerts:
        print(f"\n=== 위험 알림 ===")
        for alert in alerts:
            print(f"{alert['symbol']}: {alert['severity']} - {alert['description']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_analysis_engine())