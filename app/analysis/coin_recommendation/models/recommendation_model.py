# app/analysis/coin_recommendation/models/recommendation_model.py
# 작업: 코인 추천 시스템 데이터 모델 정의

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class RecommendationSignal(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class ExchangeData(BaseModel):
    exchange: str
    symbol: str
    price: float
    volume_24h: float
    volume_change_24h: float
    price_change_24h: float
    market_cap: Optional[float] = None
    liquidity_score: float
    timestamp: datetime

class TechnicalIndicators(BaseModel):
    rsi: float
    ma_20: float
    ma_50: float
    bollinger_upper: float
    bollinger_lower: float
    macd: float
    volume_ma: float
    volatility: float

class MarketSentiment(BaseModel):
    fear_greed_index: int
    social_sentiment: float
    news_sentiment: float
    whale_activity: float
    funding_rate: float

class CoinAnalysis(BaseModel):
    symbol: str
    exchanges_data: List[ExchangeData]
    technical_indicators: TechnicalIndicators
    market_sentiment: MarketSentiment
    risk_factors: Dict[str, float]
    delisting_risk: float
    analysis_timestamp: datetime

class CoinRecommendation(BaseModel):
    symbol: str
    score: int = Field(..., ge=0, le=100)
    signal: RecommendationSignal
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str]
    target_allocation: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    analysis: CoinAnalysis
    created_at: datetime = Field(default_factory=datetime.now)

class RecommendationBatch(BaseModel):
    recommendations: List[CoinRecommendation]
    total_analyzed: int
    batch_id: str
    analysis_duration: float
    market_conditions: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)