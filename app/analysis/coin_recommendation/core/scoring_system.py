# app/analysis/coin_recommendation/core/scoring_system.py
"""
AI 기반 코인 추천 스코어링 시스템
- 머신러닝 기반 종합 점수 계산
- 다중 요소 가중치 최적화
- 실시간 추천 생성 및 관리
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import json
from collections import defaultdict
import pickle
import os

from .data_collector import CoinData
from .analysis_engine import AnalysisResult, VolatilityAnalysisEngine

logger = logging.getLogger(__name__)

@dataclass
class RecommendationScore:
    """추천 점수 상세 구조체"""
    symbol: str
    market: str
    
    # 개별 점수
    volume_score: float
    volatility_score: float
    technical_score: float
    market_condition_score: float
    liquidity_score: float
    
    # 가중치 적용 점수
    weighted_total: float
    final_score: float  # 0-100
    
    # 등급 및 추천
    grade: str  # A+, A, B, C, D
    is_recommended: bool
    confidence: float  # 추천 신뢰도 0-1
    
    # 설명 및 이유
    recommendation_reason: str
    risk_warnings: List[str]
    strength_points: List[str]
    
    # 메타데이터
    last_updated: datetime
    data_quality: str  # HIGH, MEDIUM, LOW
    
class AIRecommendationEngine:
    """AI 기반 추천 엔진"""
    
    def __init__(self, model_path: str = "models/recommendation_model.pkl"):
        self.model_path = model_path
        self.analysis_engine = VolatilityAnalysisEngine()
        
        # 가중치 설정 (학습을 통해 최적화 가능)
        self.weights = {
            'volume': 0.25,
            'volatility': 0.30,
            'technical': 0.20,
            'market_condition': 0.15,
            'liquidity': 0.10
        }
        
        # 시장 조건별 가중치 조정
        self.market_conditions = {
            'bullish': {'volatility': 0.25, 'technical': 0.25},
            'bearish': {'volatility': 0.35, 'liquidity': 0.15},
            'sideways': {'volume': 0.30, 'technical': 0.25}
        }
        
        # 추천 이력 및 성과 추적
        self.recommendation_history: Dict[str, List[RecommendationScore]] = defaultdict(list)
        self.performance_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'total_recommendations': 0,
            'successful_recommendations': 0
        }
        
        # 실시간 데이터
        self.current_recommendations: Dict[str, RecommendationScore] = {}
        self.market_context = {
            'trend': 'sideways',
            'volatility_index': 0.5,
            'volume_index': 0.5,
            'fear_greed_index': 50
        }

    async def generate_recommendation(self, coin_data: CoinData) -> RecommendationScore:
        """개별 코인 추천 생성"""
        
        # 1단계: 기본 분석
        analysis_result = self.analysis_engine.analyze_coin(coin_data)
        
        # 2단계: 개별 점수 계산
        scores = await self._calculate_individual_scores(coin_data, analysis_result)
        
        # 3단계: 시장 조건 반영
        market_adjusted_scores = self._adjust_for_market_conditions(scores)
        
        # 4단계: 가중치 적용 및 최종 점수
        final_score = self._calculate_weighted_score(market_adjusted_scores)
        
        # 5단계: 등급 및 추천 결정
        grade, is_recommended, confidence = self._determine_recommendation(
            final_score, analysis_result, scores
        )
        
        # 6단계: 설명 생성
        reason, warnings, strengths = self._generate_explanations(
            scores, analysis_result, grade
        )
        
        # 7단계: 데이터 품질 평가
        data_quality = self._assess_data_quality(coin_data, analysis_result)
        
        recommendation = RecommendationScore(
            symbol=coin_data.symbol,
            market=coin_data.market,
            volume_score=scores['volume'],
            volatility_score=scores['volatility'],
            technical_score=scores['technical'],
            market_condition_score=scores['market_condition'],
            liquidity_score=scores['liquidity'],
            weighted_total=sum(score * self.weights[key] for key, score in market_adjusted_scores.items()),
            final_score=final_score,
            grade=grade,
            is_recommended=is_recommended,
            confidence=confidence,
            recommendation_reason=reason,
            risk_warnings=warnings,
            strength_points=strengths,
            last_updated=datetime.now(),
            data_quality=data_quality
        )
        
        # 캐시 저장
        key = f"{coin_data.symbol}:{coin_data.market}"
        self.current_recommendations[key] = recommendation
        
        # 이력 저장
        self.recommendation_history[key].append(recommendation)
        
        return recommendation

    async def _calculate_individual_scores(self, coin_data: CoinData, 
                                         analysis_result: AnalysisResult) -> Dict[str, float]:
        """개별 점수 계산"""
        
        scores = {}
        
        # 1. 거래량 점수 (이미 분석 엔진에서 계산됨)
        scores['volume'] = analysis_result.volume_score
        
        # 2. 변동성 점수 (이미 분석 엔진에서 계산됨)
        scores['volatility'] = analysis_result.volatility_score
        
        # 3. 기술적 분석 점수
        scores['technical'] = await self._calculate_technical_score(coin_data, analysis_result)
        
        # 4. 시장 조건 점수
        scores['market_condition'] = self._calculate_market_condition_score(coin_data)
        
        # 5. 유동성 점수
        scores['liquidity'] = self._calculate_liquidity_score(coin_data, analysis_result)
        
        return scores

    async def _calculate_technical_score(self, coin_data: CoinData, 
                                       analysis_result: AnalysisResult) -> float:
        """기술적 분석 점수"""
        
        technical_score = 50.0  # 기본 점수
        
        # 가격 위치 분석 (고점 대비 현재 위치)
        if coin_data.high_24h > 0:
            price_position = coin_data.price / coin_data.high_24h
            if price_position > 0.95:  # 고점 근처
                technical_score += 10
            elif price_position > 0.85:
                technical_score += 20
            elif price_position > 0.70:
                technical_score += 30  # 최적 위치
            elif price_position > 0.50:
                technical_score += 20
            else:  # 저점 근처
                technical_score += 5
        
        # 24시간 변동성 기반 모멘텀 분석
        change_24h = coin_data.change_24h
        if 0 < change_24h <= 5:  # 적정 상승
            technical_score += 20
        elif 5 < change_24h <= 10:  # 좋은 상승
            technical_score += 15
        elif 10 < change_24h <= 20:  # 높은 상승 (주의)
            technical_score += 5
        elif change_24h > 20:  # 과도한 상승
            technical_score -= 10
        elif -5 <= change_24h < 0:  # 소폭 하락
            technical_score += 10
        elif -15 <= change_24h < -5:  # 조정
            technical_score += 25  # 매수 기회
        else:  # 큰 하락
            technical_score -= 15
        
        # 거래량-가격 일치도
        if analysis_result.volume_ratio > 1.5 and change_24h > 0:  # 상승과 함께 거래량 증가
            technical_score += 15
        elif analysis_result.volume_ratio < 0.5 and change_24h > 5:  # 거래량 없는 상승
            technical_score -= 20
        
        return max(0, min(100, technical_score))

    def _calculate_market_condition_score(self, coin_data: CoinData) -> float:
        """시장 조건 점수"""
        
        base_score = 60.0
        
        # 현재 시장 트렌드에 따른 조정
        market_trend = self.market_context['trend']
        
        if market_trend == 'bullish':
            # 상승장에서는 모멘텀 중시
            if coin_data.change_24h > 0:
                base_score += 20
            else:
                base_score -= 10
                
        elif market_trend == 'bearish':
            # 하락장에서는 안정성 중시
            if abs(coin_data.change_24h) < 5:  # 안정적
                base_score += 30
            elif coin_data.change_24h < -10:  # 과도한 하락
                base_score -= 25
                
        else:  # sideways
            # 횡보장에서는 거래량 중시
            if 0.8 <= coin_data.volume_usdt / 1000000 <= 5.0:  # 적정 거래량
                base_score += 20
        
        # 전체 시장 변동성 지수 반영
        market_vol = self.market_context['volatility_index']
        if market_vol > 0.8:  # 고변동성 시장
            base_score -= 15
        elif market_vol < 0.3:  # 저변동성 시장
            base_score += 10
        
        return max(0, min(100, base_score))

    def _calculate_liquidity_score(self, coin_data: CoinData, 
                                 analysis_result: AnalysisResult) -> float:
        """유동성 점수"""
        
        # 거래량 기반 유동성 평가
        volume_usdt = coin_data.volume_usdt
        
        if volume_usdt >= 10000000:  # 1000만 USDT 이상
            liquidity_score = 95
        elif volume_usdt >= 5000000:  # 500만 USDT 이상
            liquidity_score = 85
        elif volume_usdt >= 1000000:  # 100만 USDT 이상
            liquidity_score = 75
        elif volume_usdt >= 500000:  # 50만 USDT 이상
            liquidity_score = 60
        elif volume_usdt >= 100000:  # 10만 USDT 이상
            liquidity_score = 40
        else:
            liquidity_score = 20
        
        # 거래량 안정성 보너스
        if analysis_result.volume_trend == 'stable':
            liquidity_score += 10
        elif analysis_result.volume_trend == 'increasing':
            liquidity_score += 5
        
        # 가격 안정성 보너스
        if analysis_result.price_volatility_24h < 10:
            liquidity_score += 5
        
        return max(0, min(100, liquidity_score))

    def _adjust_for_market_conditions(self, scores: Dict[str, float]) -> Dict[str, float]:
        """시장 조건에 따른 점수 조정"""
        
        adjusted_scores = scores.copy()
        market_trend = self.market_context['trend']
        
        # 시장 조건별 가중치 조정 적용
        if market_trend in self.market_conditions:
            adjustments = self.market_conditions[market_trend]
            
            for score_type, boost in adjustments.items():
                if score_type in adjusted_scores:
                    # 가중치 증가는 점수에 소폭 반영
                    adjusted_scores[score_type] = min(100, adjusted_scores[score_type] * (1 + boost))
        
        return adjusted_scores

    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """가중치 적용 최종 점수 계산"""
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score_type, score in scores.items():
            if score_type in self.weights:
                weight = self.weights[score_type]
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight

    def _determine_recommendation(self, final_score: float, 
                                analysis_result: AnalysisResult,
                                scores: Dict[str, float]) -> Tuple[str, bool, float]:
        """등급 및 추천 여부 결정"""
        
        # 위험 레벨 고려
        risk_level = analysis_result.risk_level
        
        # 기본 등급 결정
        if final_score >= 90:
            base_grade = 'A+'
        elif final_score >= 80:
            base_grade = 'A'
        elif final_score >= 70:
            base_grade = 'B'
        elif final_score >= 60:
            base_grade = 'C'
        else:
            base_grade = 'D'
        
        # 위험 레벨에 따른 등급 하향 조정
        if risk_level == 'CRITICAL':
            final_grade = 'D'
        elif risk_level == 'HIGH':
            final_grade = max('C', base_grade) if base_grade > 'C' else base_grade
        elif risk_level == 'MEDIUM':
            final_grade = max('B', base_grade) if base_grade > 'B' else base_grade
        else:
            final_grade = base_grade
        
        # 추천 여부 결정
        is_recommended = (
            final_grade in ['A+', 'A', 'B'] and
            risk_level != 'CRITICAL' and
            final_score >= 70
        )
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(final_score, analysis_result, scores)
        
        return final_grade, is_recommended, confidence

    def _calculate_confidence(self, final_score: float, 
                            analysis_result: AnalysisResult,
                            scores: Dict[str, float]) -> float:
        """추천 신뢰도 계산"""
        
        base_confidence = final_score / 100.0
        
        # 데이터 품질에 따른 조정
        if len(analysis_result.risk_flags) == 0:
            base_confidence += 0.1
        elif 'INSUFFICIENT_DATA' in analysis_result.risk_flags:
            base_confidence -= 0.2
        
        # 점수 일관성 검토
        score_values = list(scores.values())
        if len(score_values) > 1:
            score_std = np.std(score_values)
            if score_std < 15:  # 점수들이 일관된 경우
                base_confidence += 0.1
            elif score_std > 30:  # 점수들이 상충하는 경우
                base_confidence -= 0.15
        
        # 거래량 신뢰성
        if analysis_result.volume_ratio >= 0.5:
            base_confidence += 0.05
        
        return max(0.0, min(1.0, base_confidence))

    def _generate_explanations(self, scores: Dict[str, float],
                             analysis_result: AnalysisResult,
                             grade: str) -> Tuple[str, List[str], List[str]]:
        """추천 이유 및 경고사항 생성"""
        
        # 추천 이유 생성
        reasons = []
        warnings = []
        strengths = []
        
        # 점수 기반 강점 분석
        if scores['volume'] >= 80:
            strengths.append("안정적이고 충분한 거래량")
        if scores['volatility'] >= 80:
            strengths.append("낮은 변동성으로 안전한 거래환경")
        if scores['technical'] >= 80:
            strengths.append("기술적으로 양호한 차트 패턴")
        if scores['liquidity'] >= 80:
            strengths.append("우수한 유동성 확보")
        
        # 등급별 기본 추천 이유
        if grade == 'A+':
            reasons.append("모든 지표가 우수하여 강력 추천")
        elif grade == 'A':
            reasons.append("대부분 지표가 양호하여 추천")
        elif grade == 'B':
            reasons.append("적정 수준의 리스크로 신중한 거래 권장")
        elif grade == 'C':
            reasons.append("높은 리스크, 소액 거래만 권장")
        else:
            reasons.append("고위험 상태로 거래 비추천")
        
        # 위험 요소 기반 경고
        for risk_flag in analysis_result.risk_flags:
            if risk_flag == 'HIGH_VOLATILITY':
                warnings.append("높은 변동성 주의 - 손실 위험 증가")
            elif risk_flag == 'PUMP_DUMP_PATTERN':
                warnings.append("급등급락 패턴 감지 - 즉시 거래 중단 권고")
            elif risk_flag == 'VOLUME_SPIKE':
                warnings.append("비정상적 거래량 급증 - 시장 조작 가능성")
            elif risk_flag == 'LOW_LIQUIDITY':
                warnings.append("유동성 부족 - 거래 체결 어려움 가능")
            elif risk_flag == 'MANIPULATION_SUSPECTED':
                warnings.append("시장 조작 의심 - 거래 금지 권고")
        
        # 추가 세부 분석
        if analysis_result.price_volatility_24h > 20:
            warnings.append(f"24시간 변동률 {analysis_result.price_volatility_24h:.1f}% - 고위험")
        
        if analysis_result.volume_ratio < 0.3:
            warnings.append("거래량 부족으로 유동성 위험")
        elif analysis_result.volume_ratio > 5.0:
            warnings.append("비정상적 거래량 - 주의 필요")
        
        # 최종 추천 이유 조합
        main_reason = " | ".join(reasons)
        if strengths:
            main_reason += f" (강점: {', '.join(strengths)})"
        
        return main_reason, warnings, strengths

    def _assess_data_quality(self, coin_data: CoinData, 
                           analysis_result: AnalysisResult) -> str:
        """데이터 품질 평가"""
        
        quality_score = 100
        
        # 기본 데이터 완정성 체크
        if coin_data.volume_usdt <= 0:
            quality_score -= 30
        if coin_data.price <= 0:
            quality_score -= 40
        if coin_data.high_24h <= coin_data.low_24h:
            quality_score -= 20
        
        # 분석 데이터 품질
        if 'INSUFFICIENT_DATA' in analysis_result.risk_flags:
            quality_score -= 25
        
        # 거래량 신뢰성
        if analysis_result.volume_ratio < 0.1 or analysis_result.volume_ratio > 20:
            quality_score -= 15
        
        # 변동성 데이터 신뢰성
        if analysis_result.price_volatility_24h > 100:  # 비현실적 변동성
            quality_score -= 20
        
        if quality_score >= 80:
            return 'HIGH'
        elif quality_score >= 60:
            return 'MEDIUM'
        else:
            return 'LOW'

    async def generate_batch_recommendations(self, coin_list: List[CoinData]) -> List[RecommendationScore]:
        """배치 추천 생성"""
        
        logger.info(f"배치 추천 시작: {len(coin_list)}개 코인")
        
        recommendations = []
        
        # 시장 컨텍스트 업데이트
        await self._update_market_context(coin_list)
        
        # 병렬 처리
        tasks = []
        for coin_data in coin_list:
            task = self.generate_recommendation(coin_data)
            tasks.append(task)
        
        # 최대 동시 처리 수 제한
        semaphore = asyncio.Semaphore(20)
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await task
        
        # 배치 실행
        try:
            results = await asyncio.gather(
                *[process_with_semaphore(task) for task in tasks],
                return_exceptions=True
            )
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"코인 {coin_list[i].symbol} 추천 생성 실패: {result}")
                else:
                    recommendations.append(result)
                    
        except Exception as e:
            logger.error(f"배치 추천 생성 중 오류: {e}")
        
        logger.info(f"배치 추천 완료: {len(recommendations)}개 성공")
        
        return recommendations

    async def _update_market_context(self, coin_list: List[CoinData]):
        """시장 컨텍스트 업데이트"""
        
        if not coin_list:
            return
        
        # 전체 시장 통계 계산
        total_volume = sum(coin.volume_usdt for coin in coin_list)
        price_changes = [coin.change_24h for coin in coin_list if coin.change_24h is not None]
        
        if price_changes:
            avg_change = np.mean(price_changes)
            volatility_index = np.std(price_changes) / 20.0  # 정규화
            
            # 트렌드 결정
            if avg_change > 2.0:
                trend = 'bullish'
            elif avg_change < -2.0:
                trend = 'bearish'
            else:
                trend = 'sideways'
            
            # 공포/탐욕 지수 계산 (간단한 모델)
            positive_coins = len([c for c in price_changes if c > 0])
            fear_greed = (positive_coins / len(price_changes)) * 100
            
            self.market_context.update({
                'trend': trend,
                'volatility_index': min(1.0, volatility_index),
                'volume_index': min(1.0, total_volume / 1000000000),  # 정규화
                'fear_greed_index': fear_greed
            })
            
            logger.info(f"시장 컨텍스트 업데이트: {self.market_context}")

    def get_top_recommendations(self, limit: int = 20, 
                              market_filter: str = None,
                              min_grade: str = 'B') -> List[RecommendationScore]:
        """상위 추천 코인 조회"""
        
        grade_values = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
        min_grade_value = grade_values.get(min_grade, 3)
        
        filtered_recommendations = []
        
        for recommendation in self.current_recommendations.values():
            # 등급 필터
            if grade_values.get(recommendation.grade, 0) < min_grade_value:
                continue
                
            # 마켓 필터
            if market_filter and recommendation.market != market_filter:
                continue
                
            # 추천 여부 필터
            if not recommendation.is_recommended:
                continue
                
            filtered_recommendations.append(recommendation)
        
        # 최종 점수 기준 정렬
        filtered_recommendations.sort(
            key=lambda x: (x.final_score, x.confidence), 
            reverse=True
        )
        
        return filtered_recommendations[:limit]

    def get_market_summary(self) -> Dict:
        """시장 요약 정보"""
        
        if not self.current_recommendations:
            return {
                'total_analyzed': 0,
                'recommended_count': 0,
                'grade_distribution': {},
                'average_score': 0,
                'market_context': self.market_context
            }
        
        recommendations = list(self.current_recommendations.values())
        
        # 등급별 분포
        grade_dist = {}
        for rec in recommendations:
            grade = rec.grade
            grade_dist[grade] = grade_dist.get(grade, 0) + 1
        
        # 추천 가능한 코인 수
        recommended_count = len([r for r in recommendations if r.is_recommended])
        
        # 평균 점수
        avg_score = np.mean([r.final_score for r in recommendations])
        
        return {
            'total_analyzed': len(recommendations),
            'recommended_count': recommended_count,
            'grade_distribution': grade_dist,
            'average_score': round(avg_score, 2),
            'market_context': self.market_context,
            'last_update': max([r.last_updated for r in recommendations])
        }

    def save_model(self):
        """모델 및 설정 저장"""
        
        model_data = {
            'weights': self.weights,
            'market_conditions': self.market_conditions,
            'performance_metrics': self.performance_metrics,
            'market_context': self.market_context
        }
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"모델 저장 완료: {self.model_path}")

    def load_model(self):
        """모델 및 설정 로드"""
        
        if not os.path.exists(self.model_path):
            logger.warning(f"모델 파일 없음: {self.model_path}")
            return
            
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.weights = model_data.get('weights', self.weights)
            self.market_conditions = model_data.get('market_conditions', self.market_conditions)
            self.performance_metrics = model_data.get('performance_metrics', self.performance_metrics)
            self.market_context = model_data.get('market_context', self.market_context)
            
            logger.info(f"모델 로드 완료: {self.model_path}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")


# 테스트 및 사용 예시
async def test_recommendation_system():
    """추천 시스템 테스트"""
    
    # 엔진 초기화
    engine = AIRecommendationEngine()
    
    # 테스트 데이터
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
            symbol='DOGE',
            market='OKX_USDT_SPOT',
            price=0.085,
            volume_24h=5000000,
            volume_usdt=425000,
            change_1h=2.1,
            change_24h=8.5,
            high_24h=0.092,
            low_24h=0.078,
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
    
    # 배치 추천 생성
    recommendations = await engine.generate_batch_recommendations(test_coins)
    
    print("\n=== AI 추천 시스템 결과 ===")
    
    for rec in recommendations:
        print(f"\n{rec.symbol} ({rec.market})")
        print(f"  최종 점수: {rec.final_score:.1f}")
        print(f"  등급: {rec.grade}")
        print(f"  추천 여부: {'✅' if rec.is_recommended else '❌'}")
        print(f"  신뢰도: {rec.confidence:.2f}")
        print(f"  이유: {rec.recommendation_reason}")
        
        if rec.risk_warnings:
            print(f"  ⚠️ 경고: {', '.join(rec.risk_warnings)}")
        
        if rec.strength_points:
            print(f"  💪 강점: {', '.join(rec.strength_points)}")
        
        print(f"  세부 점수:")
        print(f"    거래량: {rec.volume_score:.1f}")
        print(f"    변동성: {rec.volatility_score:.1f}")
        print(f"    기술적: {rec.technical_score:.1f}")
        print(f"    시장조건: {rec.market_condition_score:.1f}")
        print(f"    유동성: {rec.liquidity_score:.1f}")
    
    # 상위 추천 조회
    top_recommendations = engine.get_top_recommendations(limit=5)
    print(f"\n=== 상위 추천 코인 ===")
    for i, rec in enumerate(top_recommendations, 1):
        print(f"{i}. {rec.symbol}: {rec.grade} - {rec.final_score:.1f}점")
    
    # 시장 요약
    summary = engine.get_market_summary()
    print(f"\n=== 시장 요약 ===")
    print(f"분석된 코인: {summary['total_analyzed']}개")
    print(f"추천 코인: {summary['recommended_count']}개")
    print(f"평균 점수: {summary['average_score']}")
    print(f"등급 분포: {summary['grade_distribution']}")
    print(f"시장 트렌드: {summary['market_context']['trend']}")
    
    # 모델 저장
    engine.save_model()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_recommendation_system())