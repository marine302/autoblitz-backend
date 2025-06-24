# app/exchanges/coinbase/client.py
# 작업: Coinbase Advanced API 클라이언트 구현

import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CoinbaseClient:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.exchange.coinbase.com"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def _request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        url = f"{self.base_url}{endpoint}"
        headers = {"Accept": "application/json"}
        
        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Coinbase API error: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Coinbase request failed: {e}")
            return {}
    
    async def get_products(self) -> List[Dict]:
        endpoint = "/products"
        return await self._request("GET", endpoint)
    
    async def get_24hr_stats(self, product_id: str) -> Dict:
        endpoint = f"/products/{product_id}/stats"
        return await self._request("GET", endpoint)
    
    async def get_ticker(self, product_id: str) -> Dict:
        endpoint = f"/products/{product_id}/ticker"
        return await self._request("GET", endpoint)
    
    async def get_orderbook(self, product_id: str, level: int = 2) -> Dict:
        endpoint = f"/products/{product_id}/book"
        params = {"level": level}
        return await self._request("GET", endpoint, params)
    
    async def get_top_coins_by_volume(self, limit: int = 100) -> List[Dict]:
        products = await self.get_products()
        if not products:
            return []
            
        # USD 페어만 필터링
        usd_pairs = [p for p in products if p['quote_currency'] == 'USD' and p['status'] == 'online']
        
        # 각 제품의 24시간 통계 가져오기
        stats_tasks = []
        for product in usd_pairs[:limit]:
            stats_tasks.append(self.get_24hr_stats(product['id']))
            
        stats_results = await asyncio.gather(*stats_tasks, return_exceptions=True)
        
        # 거래량이 있는 제품만 필터링하고 정렬
        valid_products = []
        for i, stats in enumerate(stats_results):
            if isinstance(stats, dict) and 'volume' in stats:
                product_data = usd_pairs[i].copy()
                product_data.update(stats)
                valid_products.append(product_data)
        
        return sorted(valid_products, key=lambda x: float(x.get('volume', 0)), reverse=True)[:limit]
    
    async def get_coin_stats(self, product_id: str) -> Dict:
        ticker = await self.get_ticker(product_id)
        stats = await self.get_24hr_stats(product_id)
        
        if not ticker or not stats:
            return {}
            
        return {
            "exchange": "coinbase",
            "symbol": product_id,
            "price": float(ticker.get('price', 0)),
            "volume_24h": float(stats.get('volume', 0)),
            "volume_change_24h": 0,  # Coinbase doesn't provide volume change
            "price_change_24h": 0,   # Calculate from open price
            "high_24h": float(stats.get('high', 0)),
            "low_24h": float(stats.get('low', 0)),
            "timestamp": datetime.now()
        }