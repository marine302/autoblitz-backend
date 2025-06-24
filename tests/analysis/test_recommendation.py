# tests/analysis/test_recommendation.py
# 작업: 코인 추천 시스템 테스트

import pytest
import asyncio
from datetime import datetime

from app.analysis.coin_recommendation.core.analyzer import CoinAnalyzer
from app.analysis.coin_recommendation.core.exchange_manager import ExchangeManager
from app.analysis.coin_recommendation.models.recommendation_model import RecommendationSignal, RiskLevel

class TestCoinRecommendation:
    
    @pytest.fixture
    async def analyzer(self):
        """테스트용 분석기 인스턴스"""
        analyzer = CoinAnalyzer()
        await analyzer.initialize()
        yield analyzer
        await analyzer.close()
    
    @pytest.fixture
    def exchange_manager(self):
        """테스트용 거래소 매니저"""
        return ExchangeManager()
    
    @pytest.mark.asyncio
    async def test_exchange_manager_initialization(self, exchange_manager):
        """거래소 매니저 초기화 테스트"""
        await exchange_manager.initialize_clients()
        
        # 최소 OKX와 업비트는 초기화되어야 함
        assert "okx" in exchange_manager.clients
        assert "upbit" in exchange_manager.clients
        
        await exchange_manager.close_all_clients()
    
    @pytest.mark.asyncio
    async def test_btc_analysis(self, analyzer):
        """BTC 분석 테스트"""
        recommendation = await analyzer.analyze_coin("BTC")
        
        assert recommendation is not None
        assert recommendation.symbol == "BTC"
        assert 0 <= recommendation.score <= 100
        assert recommendation.signal in RecommendationSignal
        assert recommendation.risk_level in RiskLevel
        assert len(recommendation.reasons) > 0
        assert recommendation.analysis is not None
    
    @pytest.mark.asyncio
    async def test_multiple_coins_analysis(self, analyzer):
        """다중 코인 분석 테스트"""
        symbols = ["BTC", "ETH", "BNB"]
        tasks = [analyzer.analyze_coin(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        # 모든 코인 분석 성공 확인
        for i, result in enumerate(results):
            assert result is not None, f"{symbols[i]} 분석 실패"
            assert result.symbol == symbols[i]
    
    @pytest.mark.asyncio
    async def test_top_recommendations(self, analyzer):
        """상위 추천 테스트"""
        recommendations = await analyzer.analyze_top_coins(limit=5)
        
        assert len(recommendations) > 0
        assert len(recommendations) <= 5
        
        # 점수 기준 내림차순 정렬 확인
        for i in range(len(recommendations) - 1):
            assert recommendations[i].score >= recommendations[i + 1].score
    
    @pytest.mark.asyncio
    async def test_aggregated_data(self, exchange_manager):
        """통합 데이터 수집 테스트"""
        await exchange_manager.initialize_clients()
        
        aggregated = await exchange_manager.get_aggregated_data("BTC")
        
        assert aggregated["symbol"] == "BTC"
        assert aggregated["avg_price"] > 0
        assert aggregated["total_volume"] > 0
        assert len(aggregated["exchanges"]) > 0
        
        await exchange_manager.close_all_clients()
    
    @pytest.mark.asyncio
    async def test_arbitrage_detection(self, exchange_manager):
        """차익거래 기회 탐지 테스트"""
        await exchange_manager.initialize_clients()
        
        # ETH로 테스트 (거래소간 가격 차이가 있을 가능성)
        aggregated = await exchange_manager.get_aggregated_data("ETH")
        
        if len(aggregated["exchanges"]) >= 2:
            # 차익거래 기회가 탐지될 수 있음
            arbitrage_opps = aggregated.get("arbitrage_opportunities", [])
            # 기회가 있다면 유효성 검증
            for opp in arbitrage_opps:
                assert "buy_exchange" in opp
                assert "sell_exchange" in opp
                assert "profit_percent" in opp
                assert opp["profit_percent"] > 0
        
        await exchange_manager.close_all_clients()
    
    def test_recommendation_model_validation(self):
        """추천 모델 유효성 검증"""
        from app.analysis.coin_recommendation.models.recommendation_model import (
            CoinRecommendation, CoinAnalysis, TechnicalIndicators, MarketSentiment, ExchangeData
        )
        
        # 기본 데이터 생성
        exchange_data = ExchangeData(
            exchange="test",
            symbol="BTC",
            price=50000.0,
            volume_24h=1000000.0,
            volume_change_24h=10.0,
            price_change_24h=5.0,
            liquidity_score=80.0,
            timestamp=datetime.now()
        )
        
        technical = TechnicalIndicators(
            rsi=45.0,
            ma_20=49000.0,
            ma_50=48000.0,
            bollinger_upper=52000.0,
            bollinger_lower=46000.0,
            macd=100.0,
            volume_ma=900000.0,
            volatility=2.5
        )
        
        sentiment = MarketSentiment(
            fear_greed_index=60,
            social_sentiment=0.7,
            news_sentiment=0.6,
            whale_activity=0.3,
            funding_rate=0.0001
        )
        
        analysis = CoinAnalysis(
            symbol="BTC",
            exchanges_data=[exchange_data],
            technical_indicators=technical,
            market_sentiment=sentiment,
            risk_factors={"volatility": 0.3, "liquidity": 0.2},
            delisting_risk=0.1,
            analysis_timestamp=datetime.now()
        )
        
        recommendation = CoinRecommendation(
            symbol="BTC",
            score=75,
            signal=RecommendationSignal.BUY,
            risk_level=RiskLevel.MEDIUM,
            confidence=0.8,
            reasons=["기술적 분석 양호", "거래량 충분"],
            target_allocation=0.05,
            analysis=analysis
        )
        
        # 모델 유효성 확인
        assert recommendation.symbol == "BTC"
        assert 0 <= recommendation.score <= 100
        assert 0.0 <= recommendation.confidence <= 1.0
        assert len(recommendation.reasons) > 0

if __name__ == "__main__":
    # 간단한 실행 테스트
    async def quick_test():
        analyzer = CoinAnalyzer()
        await analyzer.initialize()
        
        print("🧪 BTC 분석 테스트...")
        btc_rec = await analyzer.analyze_coin("BTC")
        if btc_rec:
            print(f"✅ BTC 점수: {btc_rec.score}, 신호: {btc_rec.signal.value}")
        else:
            print("❌ BTC 분석 실패")
        
        await analyzer.close()
    
    asyncio.run(quick_test())