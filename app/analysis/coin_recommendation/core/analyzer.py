# app/analysis/coin_recommendation/core/analyzer.py
# 작업: AI 기반 코인 분석 및 추천 엔진

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from .exchange_manager import ExchangeManager
from ..models.recommendation_model import (
    CoinRecommendation, CoinAnalysis, TechnicalIndicators, 
    MarketSentiment, ExchangeData, RecommendationSignal, RiskLevel
)

logger = logging.getLogger(__name__)

@dataclass
class AnalysisWeights:
    volume_weight: float = 0.25
    price_stability: float = 0.20
    liquidity_weight: float = 0.20
    technical_weight: float = 0.15
    sentiment_weight: float = 0.10
    arbitrage_weight: float = 0.10

class CoinAnalyzer:
    def __init__(self):
        self.exchange_manager = ExchangeManager()
        self.weights = AnalysisWeights()
        self.fear_greed_index = 50  # Mock data
        
    async def initialize(self):
        """분석기 초기화"""
        await self.exchange_manager.initialize_clients()
        logger.info("✅ 코인 분석 엔진 초기화 완료")
    
    async def analyze_coin(self, symbol: str) -> Optional[CoinRecommendation]:
        """개별 코인 분석"""
        try:
            # 모든 거래소 데이터 수집
            aggregated_data = await self.exchange_manager.get_aggregated_data(symbol)
            
            if not aggregated_data["exchanges"]:
                logger.warning(f"{symbol}: 거래소 데이터 없음")
                return None
            
            # 분석 컴포넌트 계산
            technical_indicators = await self._calculate_technical_indicators(symbol, aggregated_data)
            market_sentiment = await self._analyze_market_sentiment(symbol)
            risk_factors = self._calculate_risk_factors(aggregated_data)
            
            # 코인 분석 결과 생성
            analysis = CoinAnalysis(
                symbol=symbol,
                exchanges_data=[
                    ExchangeData(
                        exchange=data["exchange"],
                        symbol=symbol,
                        price=data["price"],
                        volume_24h=data["volume_24h"],
                        volume_change_24h=data.get("volume_change_24h", 0),
                        price_change_24h=data["price_change_24h"],
                        liquidity_score=aggregated_data["liquidity_score"]
                    ) for data in aggregated_data["exchanges"]
                ],
                technical_indicators=technical_indicators,
                market_sentiment=market_sentiment,
                risk_factors=risk_factors,
                delisting_risk=self._calculate_delisting_risk(aggregated_data),
                analysis_timestamp=datetime.now()
            )
            
            # 최종 추천 점수 계산
            recommendation = self._generate_recommendation(symbol, analysis, aggregated_data)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"{symbol} 분석 중 오류: {e}")
            return None
    
    async def analyze_top_coins(self, limit: int = 50) -> List[CoinRecommendation]:
        """상위 코인들 일괄 분석"""
        # 공통 심볼 목록 가져오기
        symbols = await self.exchange_manager._get_common_symbols()
        
        # 병렬 분석 실행
        tasks = []
        for symbol in symbols[:limit]:
            task = self.analyze_coin(symbol)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 성공한 분석 결과만 필터링
        recommendations = []
        for result in results:
            if isinstance(result, CoinRecommendation):
                recommendations.append(result)
            elif isinstance(result, Exception):
                logger.error(f"코인 분석 실패: {result}")
        
        # 점수 기준 정렬
        return sorted(recommendations, key=lambda x: x.score, reverse=True)
    
    async def _calculate_technical_indicators(self, symbol: str, aggregated_data: Dict) -> TechnicalIndicators:
        """기술적 지표 계산"""
        # 실제 구현에서는 각 거래소에서 OHLCV 데이터를 가져와 계산
        # 현재는 간단한 Mock 계산
        
        avg_price = aggregated_data["avg_price"]
        price_variance = aggregated_data["price_variance"]
        total_volume = aggregated_data["total_volume"]
        
        # 가격 기반 간단한 지표 계산
        rsi = 50 + (np.random.random() - 0.5) * 60  # 20-80 범위
        
        return TechnicalIndicators(
            rsi=rsi,
            ma_20=avg_price * (0.95 + np.random.random() * 0.1),
            ma_50=avg_price * (0.90 + np.random.random() * 0.2),
            bollinger_upper=avg_price * 1.05,
            bollinger_lower=avg_price * 0.95,
            macd=np.random.random() - 0.5,
            volume_ma=total_volume * (0.8 + np.random.random() * 0.4),
            volatility=price_variance
        )
    
    async def _analyze_market_sentiment(self, symbol: str) -> MarketSentiment:
        """시장 심리 분석"""
        # 실제 구현에서는 외부 API에서 데이터 수집
        return MarketSentiment(
            fear_greed_index=self.fear_greed_index,
            social_sentiment=0.5 + (np.random.random() - 0.5) * 0.6,
            news_sentiment=0.4 + (np.random.random() - 0.5) * 0.4,
            whale_activity=np.random.random(),
            funding_rate=0.0001 + (np.random.random() - 0.5) * 0.0002
        )
    
    def _calculate_risk_factors(self, aggregated_data: Dict) -> Dict[str, float]:
        """리스크 요인 계산"""
        exchange_count = len(aggregated_data["exchanges"])
        price_variance = aggregated_data["price_variance"]
        liquidity_score = aggregated_data["liquidity_score"]
        
        return {
            "exchange_concentration_risk": max(0, 1 - exchange_count / 5.0),
            "price_volatility_risk": min(price_variance / 10.0, 1.0),
            "liquidity_risk": max(0, 1 - liquidity_score / 100.0),
            "market_cap_risk": 0.3,  # Mock value
            "regulatory_risk": 0.2   # Mock value
        }
    
    def _calculate_delisting_risk(self, aggregated_data: Dict) -> float:
        """상장폐지 위험도 계산"""
        exchange_count = len(aggregated_data["exchanges"])
        total_volume = aggregated_data["total_volume"]
        
        # 거래소 수와 거래량 기반 위험도
        exchange_risk = max(0, 1 - exchange_count / 5.0)
        volume_risk = max(0, 1 - total_volume / 1000000)  # 100만 달러 기준
        
        return (exchange_risk + volume_risk) / 2
    
    def _generate_recommendation(self, symbol: str, analysis: CoinAnalysis, aggregated_data: Dict) -> CoinRecommendation:
        """최종 추천 생성"""
        # 각 요소별 점수 계산
        volume_score = self._calculate_volume_score(aggregated_data["total_volume"])
        liquidity_score = aggregated_data["liquidity_score"]
        stability_score = self._calculate_stability_score(aggregated_data["price_variance"])
        technical_score = self._calculate_technical_score(analysis.technical_indicators)
        sentiment_score = self._calculate_sentiment_score(analysis.market_sentiment)
        arbitrage_score = self._calculate_arbitrage_score(aggregated_data.get("arbitrage_opportunities", []))
        
        # 가중 평균 계산
        final_score = (
            volume_score * self.weights.volume_weight +
            stability_score * self.weights.price_stability +
            liquidity_score * self.weights.liquidity_weight +
            technical_score * self.weights.technical_weight +
            sentiment_score * self.weights.sentiment_weight +
            arbitrage_score * self.weights.arbitrage_weight
        )
        
        # 리스크 조정
        risk_penalty = sum(analysis.risk_factors.values()) * 20
        final_score = max(0, final_score - risk_penalty)
        
        # 상장폐지 위험 반영
        delisting_penalty = analysis.delisting_risk * 30
        final_score = max(0, final_score - delisting_penalty)
        
        # 신호 및 리스크 레벨 결정
        signal, risk_level = self._determine_signal_and_risk(final_score, analysis)
        
        # 추천 이유 생성
        reasons = self._generate_reasons(final_score, analysis, aggregated_data)
        
        return CoinRecommendation(
            symbol=symbol,
            score=int(final_score),
            signal=signal,
            risk_level=risk_level,
            confidence=min(final_score / 100, 1.0),
            reasons=reasons,
            target_allocation=self._calculate_target_allocation(final_score, risk_level),
            analysis=analysis
        )
    
    def _calculate_volume_score(self, volume: float) -> float:
        """거래량 점수 (0-100)"""
        return min(volume / 10000000 * 100, 100)  # 1천만 달러 기준 100점
    
    def _calculate_stability_score(self, variance: float) -> float:
        """가격 안정성 점수 (0-100)"""
        return max(0, 100 - variance * 10)  # 변동성이 낮을수록 높은 점수
    
    def _calculate_technical_score(self, indicators: TechnicalIndicators) -> float:
        """기술적 분석 점수 (0-100)"""
        rsi_score = 100 - abs(indicators.rsi - 50) * 2  # RSI 50 근처가 좋음
        macd_score = 50 + indicators.macd * 100  # MACD 양수면 좋음
        return (rsi_score + macd_score) / 2
    
    def _calculate_sentiment_score(self, sentiment: MarketSentiment) -> float:
        """시장 심리 점수 (0-100)"""
        return (
            sentiment.fear_greed_index * 0.4 +
            sentiment.social_sentiment * 100 * 0.3 +
            sentiment.news_sentiment * 100 * 0.3
        )
    
    def _calculate_arbitrage_score(self, opportunities: List[Dict]) -> float:
        """차익거래 기회 점수 (0-100)"""
        if not opportunities:
            return 50
        
        max_profit = max(opp["profit_percent"] for opp in opportunities)
        return min(max_profit * 20, 100)  # 5% 차익이면 100점
    
    def _determine_signal_and_risk(self, score: float, analysis: CoinAnalysis) -> Tuple[RecommendationSignal, RiskLevel]:
        """매매 신호 및 리스크 레벨 결정"""
        # 리스크 레벨 결정
        avg_risk = sum(analysis.risk_factors.values()) / len(analysis.risk_factors)
        
        if avg_risk > 0.7:
            risk_level = RiskLevel.EXTREME
        elif avg_risk > 0.5:
            risk_level = RiskLevel.HIGH
        elif avg_risk > 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # 매매 신호 결정
        if score >= 80:
            signal = RecommendationSignal.STRONG_BUY
        elif score >= 65:
            signal = RecommendationSignal.BUY
        elif score >= 35:
            signal = RecommendationSignal.HOLD
        elif score >= 20:
            signal = RecommendationSignal.SELL
        else:
            signal = RecommendationSignal.STRONG_SELL
        
        return signal, risk_level
    
    def _generate_reasons(self, score: float, analysis: CoinAnalysis, aggregated_data: Dict) -> List[str]:
        """추천 이유 생성"""
        reasons = []
        
        # 점수 기반 이유
        if score >= 70:
            reasons.append("높은 종합 분석 점수")
        
        # 거래량 기반
        if aggregated_data["total_volume"] > 10000000:
            reasons.append("충분한 거래량 확보")
        
        # 거래소 수 기반
        exchange_count = len(aggregated_data["exchanges"])
        if exchange_count >= 4:
            reasons.append(f"{exchange_count}개 주요 거래소 상장")
        
        # 유동성 기반
        if aggregated_data["liquidity_score"] > 70:
            reasons.append("우수한 유동성")
        
        # 가격 안정성
        if aggregated_data["price_variance"] < 2:
            reasons.append("안정적인 가격 움직임")
        
        # 차익거래 기회
        arbitrage_opps = aggregated_data.get("arbitrage_opportunities", [])
        if arbitrage_opps:
            max_profit = max(opp["profit_percent"] for opp in arbitrage_opps)
            if max_profit > 1:
                reasons.append(f"차익거래 기회 존재 ({max_profit:.1f}%)")
        
        # 기술적 지표
        if analysis.technical_indicators.rsi < 30:
            reasons.append("과매도 구간 (RSI)")
        elif analysis.technical_indicators.rsi > 70:
            reasons.append("과매수 구간 (RSI)")
        
        return reasons[:5]  # 최대 5개 이유
    
    def _calculate_target_allocation(self, score: float, risk_level: RiskLevel) -> float:
        """목표 할당 비중 계산"""
        base_allocation = score / 100 * 0.1  # 최대 10%
        
        # 리스크 레벨에 따른 조정
        risk_multiplier = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.4,
            RiskLevel.EXTREME: 0.1
        }
        
        return base_allocation * risk_multiplier[risk_level]
    
    async def close(self):
        """분석기 종료"""
        await self.exchange_manager.close_all_clients()
        logger.info("✅ 코인 분석 엔진 종료 완료")