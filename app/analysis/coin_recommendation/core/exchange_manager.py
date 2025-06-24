# app/analysis/coin_recommendation/core/exchange_manager.py
# 작업: 5개 거래소 통합 데이터 수집 관리자

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass

from app.strategies.utils.okx_client import OKXClient
from app.strategies.utils.upbit_client import UpbitClient
from app.exchanges.binance.client import BinanceClient
from app.exchanges.coinbase.client import CoinbaseClient
from app.exchanges.kraken.client import KrakenClient

logger = logging.getLogger(__name__)

@dataclass
class ExchangeConfig:
    name: str
    client_class: Any
    api_key: str = None
    api_secret: str = None
    passphrase: str = None
    weight: float = 1.0
    enabled: bool = True

class ExchangeManager:
    def __init__(self):
        self.exchanges = {
            "okx": ExchangeConfig("okx", OKXClient, weight=1.2),
            "upbit": ExchangeConfig("upbit", UpbitClient, weight=1.0),
            "binance": ExchangeConfig("binance", BinanceClient, weight=1.1),
            "coinbase": ExchangeConfig("coinbase", CoinbaseClient, weight=0.9),
            "kraken": ExchangeConfig("kraken", KrakenClient, weight=0.8)
        }
        self.clients = {}
        
    async def initialize_clients(self):
        """모든 거래소 클라이언트 초기화"""
        for exchange_id, config in self.exchanges.items():
            if not config.enabled:
                continue
                
            try:
                if exchange_id == "okx":
                    client = config.client_class(
                        api_key=config.api_key,
                        secret_key=config.api_secret,
                        passphrase=config.passphrase
                    )
                else:
                    client = config.client_class(
                        api_key=config.api_key,
                        api_secret=config.api_secret
                    )
                
                self.clients[exchange_id] = client
                logger.info(f"✅ {exchange_id.upper()} 클라이언트 초기화 완료")
                
            except Exception as e:
                logger.error(f"❌ {exchange_id.upper()} 클라이언트 초기화 실패: {e}")
                
    async def collect_all_exchange_data(self, symbols: List[str] = None) -> Dict[str, List[Dict]]:
        """모든 거래소에서 데이터 수집"""
        if not symbols:
            symbols = await self._get_common_symbols()
            
        results = {}
        
        # 각 거래소별 병렬 데이터 수집
        tasks = []
        for exchange_id, client in self.clients.items():
            task = self._collect_exchange_data(exchange_id, client, symbols)
            tasks.append(task)
            
        exchange_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 정리
        for i, (exchange_id, _) in enumerate(self.clients.items()):
            if isinstance(exchange_results[i], dict):
                results[exchange_id] = exchange_results[i]
            else:
                logger.error(f"{exchange_id} 데이터 수집 실패: {exchange_results[i]}")
                results[exchange_id] = {}
                
        return results
    
    async def _collect_exchange_data(self, exchange_id: str, client: Any, symbols: List[str]) -> Dict[str, Dict]:
        """개별 거래소 데이터 수집"""
        try:
            if exchange_id == "okx":
                return await self._collect_okx_data(client, symbols)
            elif exchange_id == "upbit":
                return await self._collect_upbit_data(client, symbols)
            elif exchange_id == "binance":
                return await self._collect_binance_data(client, symbols)
            elif exchange_id == "coinbase":
                return await self._collect_coinbase_data(client, symbols)
            elif exchange_id == "kraken":
                return await self._collect_kraken_data(client, symbols)
        except Exception as e:
            logger.error(f"{exchange_id} 데이터 수집 중 오류: {e}")
            return {}
    
    async def _collect_okx_data(self, client: OKXClient, symbols: List[str]) -> Dict[str, Dict]:
        """OKX 데이터 수집"""
        data = {}
        for symbol in symbols:
            try:
                # OKX 포맷으로 변환 (예: BTC -> BTC-USDT)
                okx_symbol = f"{symbol}-USDT"
                ticker = await client.get_ticker(okx_symbol)
                if ticker:
                    data[symbol] = {
                        "exchange": "okx",
                        "symbol": symbol,
                        "price": float(ticker['last']),
                        "volume_24h": float(ticker['volCcy24h']),
                        "price_change_24h": float(ticker['change24h']),
                        "timestamp": datetime.now()
                    }
            except Exception as e:
                logger.warning(f"OKX {symbol} 데이터 수집 실패: {e}")
        return data
    
    async def _collect_upbit_data(self, client: UpbitClient, symbols: List[str]) -> Dict[str, Dict]:
        """업비트 데이터 수집"""
        data = {}
        for symbol in symbols:
            try:
                # 업비트 포맷으로 변환 (예: BTC -> KRW-BTC)
                upbit_symbol = f"KRW-{symbol}"
                ticker = await client.get_ticker(upbit_symbol)
                if ticker:
                    data[symbol] = {
                        "exchange": "upbit",
                        "symbol": symbol,
                        "price": float(ticker['trade_price']),
                        "volume_24h": float(ticker['acc_trade_volume_24h']),
                        "price_change_24h": float(ticker['change_rate']) * 100,
                        "timestamp": datetime.now()
                    }
            except Exception as e:
                logger.warning(f"업비트 {symbol} 데이터 수집 실패: {e}")
        return data
    
    async def _collect_binance_data(self, client: BinanceClient, symbols: List[str]) -> Dict[str, Dict]:
        """바이낸스 데이터 수집"""
        data = {}
        async with client:
            for symbol in symbols:
                try:
                    # 바이낸스 포맷으로 변환 (예: BTC -> BTCUSDT)
                    binance_symbol = f"{symbol}USDT"
                    stats = await client.get_coin_stats(binance_symbol)
                    if stats:
                        data[symbol] = stats
                except Exception as e:
                    logger.warning(f"바이낸스 {symbol} 데이터 수집 실패: {e}")
        return data
    
    async def _collect_coinbase_data(self, client: CoinbaseClient, symbols: List[str]) -> Dict[str, Dict]:
        """코인베이스 데이터 수집"""
        data = {}
        async with client:
            for symbol in symbols:
                try:
                    # 코인베이스 포맷으로 변환 (예: BTC -> BTC-USD)
                    coinbase_symbol = f"{symbol}-USD"
                    stats = await client.get_coin_stats(coinbase_symbol)
                    if stats:
                        data[symbol] = stats
                except Exception as e:
                    logger.warning(f"코인베이스 {symbol} 데이터 수집 실패: {e}")
        return data
    
    async def _collect_kraken_data(self, client: KrakenClient, symbols: List[str]) -> Dict[str, Dict]:
        """크라켄 데이터 수집"""
        data = {}
        async with client:
            for symbol in symbols:
                try:
                    # 크라켄 포맷 매핑 (복잡함)
                    kraken_mapping = {
                        "BTC": "XXBTZUSD",
                        "ETH": "XETHZUSD",
                        "ADA": "ADAUSD",
                        "DOT": "DOTUSD"
                    }
                    
                    kraken_symbol = kraken_mapping.get(symbol)
                    if kraken_symbol:
                        stats = await client.get_coin_stats(kraken_symbol)
                        if stats:
                            data[symbol] = stats
                except Exception as e:
                    logger.warning(f"크라켄 {symbol} 데이터 수집 실패: {e}")
        return data
    
    async def _get_common_symbols(self) -> List[str]:
        """모든 거래소에서 공통으로 지원하는 주요 코인 목록"""
        return [
            "BTC", "ETH", "BNB", "ADA", "SOL", "XRP", "DOT", "MATIC", 
            "AVAX", "LINK", "UNI", "LTC", "BCH", "ATOM", "ICP",
            "NEAR", "APT", "OP", "ARB", "SUI", "DOGE", "SHIB"
        ]
    
    async def get_aggregated_data(self, symbol: str) -> Dict:
        """특정 코인의 모든 거래소 데이터 집계"""
        all_data = await self.collect_all_exchange_data([symbol])
        
        aggregated = {
            "symbol": symbol,
            "exchanges": [],
            "avg_price": 0,
            "total_volume": 0,
            "price_variance": 0,
            "liquidity_score": 0,
            "arbitrage_opportunities": [],
            "timestamp": datetime.now()
        }
        
        prices = []
        total_volume = 0
        
        for exchange_id, exchange_data in all_data.items():
            if symbol in exchange_data:
                coin_data = exchange_data[symbol]
                aggregated["exchanges"].append(coin_data)
                prices.append(coin_data["price"])
                total_volume += coin_data["volume_24h"]
        
        if prices:
            aggregated["avg_price"] = sum(prices) / len(prices)
            aggregated["total_volume"] = total_volume
            aggregated["price_variance"] = self._calculate_variance(prices)
            aggregated["liquidity_score"] = self._calculate_liquidity_score(aggregated["exchanges"])
            aggregated["arbitrage_opportunities"] = self._find_arbitrage_opportunities(aggregated["exchanges"])
        
        return aggregated
    
    def _calculate_variance(self, prices: List[float]) -> float:
        """가격 분산 계산"""
        if len(prices) < 2:
            return 0
        
        avg = sum(prices) / len(prices)
        variance = sum((p - avg) ** 2 for p in prices) / len(prices)
        return (variance ** 0.5) / avg * 100  # 변동계수 %
    
    def _calculate_liquidity_score(self, exchanges_data: List[Dict]) -> float:
        """유동성 점수 계산 (0-100)"""
        if not exchanges_data:
            return 0
        
        total_volume = sum(data["volume_24h"] for data in exchanges_data)
        exchange_count = len(exchanges_data)
        
        # 거래량과 거래소 수를 고려한 유동성 점수
        volume_score = min(total_volume / 10000000, 100)  # 1천만 달러 기준 100점
        exchange_score = exchange_count * 10  # 거래소당 10점
        
        return min(volume_score + exchange_score, 100)
    
    def _find_arbitrage_opportunities(self, exchanges_data: List[Dict]) -> List[Dict]:
        """차익거래 기회 탐지"""
        if len(exchanges_data) < 2:
            return []
        
        opportunities = []
        
        for i, data1 in enumerate(exchanges_data):
            for j, data2 in enumerate(exchanges_data[i+1:], i+1):
                price_diff = abs(data1["price"] - data2["price"])
                avg_price = (data1["price"] + data2["price"]) / 2
                diff_percent = (price_diff / avg_price) * 100
                
                if diff_percent > 0.5:  # 0.5% 이상 차이
                    buy_exchange = data1 if data1["price"] < data2["price"] else data2
                    sell_exchange = data2 if data1["price"] < data2["price"] else data1
                    
                    opportunities.append({
                        "buy_exchange": buy_exchange["exchange"],
                        "sell_exchange": sell_exchange["exchange"],
                        "buy_price": buy_exchange["price"],
                        "sell_price": sell_exchange["price"],
                        "profit_percent": diff_percent,
                        "potential_profit": diff_percent
                    })
        
        return sorted(opportunities, key=lambda x: x["profit_percent"], reverse=True)
    
    async def close_all_clients(self):
        """모든 클라이언트 연결 종료"""
        for exchange_id, client in self.clients.items():
            try:
                if hasattr(client, 'close'):
                    await client.close()
                logger.info(f"✅ {exchange_id.upper()} 클라이언트 종료")
            except Exception as e:
                logger.error(f"❌ {exchange_id.upper()} 클라이언트 종료 실패: {e}")