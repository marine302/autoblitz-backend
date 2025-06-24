# app/exchanges/binance/client.py
# 작업: Binance API 클라이언트 구현

import aiohttp
import asyncio
import time
import hashlib
import hmac
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    def _generate_signature(self, params: Dict) -> str:
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    async def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
        
        if params is None:
            params = {}
            
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
            
        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Binance API error: {response.status} - {await response.text()}")
                    return {}
        except Exception as e:
            logger.error(f"Binance request failed: {e}")
            return {}
    
    async def get_24hr_ticker(self, symbol: str = None) -> List[Dict]:
        endpoint = "/api/v3/ticker/24hr"
        params = {"symbol": symbol} if symbol else {}
        return await self._request("GET", endpoint, params)
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        endpoint = "/api/v3/depth"
        params = {"symbol": symbol, "limit": limit}
        return await self._request("GET", endpoint, params)
    
    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> List[List]:
        endpoint = "/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return await self._request("GET", endpoint, params)
    
    async def get_exchange_info(self) -> Dict:
        endpoint = "/api/v3/exchangeInfo"
        return await self._request("GET", endpoint)
    
    async def get_top_coins_by_volume(self, limit: int = 100) -> List[Dict]:
        tickers = await self.get_24hr_ticker()
        if not tickers:
            return []
            
        # USDT 페어만 필터링하고 거래량 기준 정렬
        usdt_pairs = [
            ticker for ticker in tickers 
            if ticker['symbol'].endswith('USDT') and float(ticker['quoteVolume']) > 1000000
        ]
        
        sorted_pairs = sorted(
            usdt_pairs, 
            key=lambda x: float(x['quoteVolume']), 
            reverse=True
        )
        
        return sorted_pairs[:limit]
    
    async def get_coin_stats(self, symbol: str) -> Dict:
        ticker_data = await self.get_24hr_ticker(symbol)
        if not ticker_data:
            return {}
            
        return {
            "exchange": "binance",
            "symbol": symbol,
            "price": float(ticker_data[0]['lastPrice']),
            "volume_24h": float(ticker_data[0]['volume']),
            "volume_change_24h": float(ticker_data[0]['count']),
            "price_change_24h": float(ticker_data[0]['priceChangePercent']),
            "high_24h": float(ticker_data[0]['highPrice']),
            "low_24h": float(ticker_data[0]['lowPrice']),
            "timestamp": datetime.now()
        }