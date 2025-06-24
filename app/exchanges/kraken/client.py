# app/exchanges/kraken/client.py
# 작업: Kraken API 클라이언트 구현

import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class KrakenClient:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.kraken.com"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def _request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('error'):
                            logger.error(f"Kraken API error: {data['error']}")
                            return {}
                        return data.get('result', {})
                    else:
                        logger.error(f"Kraken API error: {response.status}")
                        return {}
            else:
                async with self.session.post(url, data=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('error'):
                            logger.error(f"Kraken API error: {data['error']}")
                            return {}
                        return data.get('result', {})
                    else:
                        logger.error(f"Kraken API error: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Kraken request failed: {e}")
            return {}
    
    async def get_asset_pairs(self) -> Dict:
        endpoint = "/0/public/AssetPairs"
        return await self._request("GET", endpoint)
    
    async def get_ticker(self, pairs: str = None) -> Dict:
        endpoint = "/0/public/Ticker"
        params = {"pair": pairs} if pairs else {}
        return await self._request("GET", endpoint, params)
    
    async def get_orderbook(self, pair: str, count: int = 100) -> Dict:
        endpoint = "/0/public/Depth"
        params = {"pair": pair, "count": count}
        return await self._request("GET", endpoint, params)
    
    async def get_ohlc(self, pair: str, interval: int = 60) -> Dict:
        endpoint = "/0/public/OHLC"
        params = {"pair": pair, "interval": interval}
        return await self._request("GET", endpoint, params)
    
    async def get_top_coins_by_volume(self, limit: int = 100) -> List[Dict]:
        # 모든 거래쌍 정보 가져오기
        asset_pairs = await self.get_asset_pairs()
        if not asset_pairs:
            return []
            
        # USD 페어만 필터링
        usd_pairs = [
            pair_id for pair_id, pair_info in asset_pairs.items() 
            if pair_info.get('quote') in ['ZUSD', 'USD'] and pair_info.get('status') == 'online'
        ]
        
        # 티커 정보 가져오기 (배치로)
        if not usd_pairs:
            return []
            
        # Kraken은 한 번에 많은 페어를 요청할 수 있음
        pairs_str = ",".join(usd_pairs[:limit])
        ticker_data = await self.get_ticker(pairs_str)
        
        if not ticker_data:
            return []
            
        # 거래량 기준으로 정렬
        sorted_pairs = []
        for pair_id, ticker in ticker_data.items():
            volume = float(ticker['v'][1])  # 24h volume
            if volume > 10000:  # 최소 거래량 필터
                sorted_pairs.append({
                    'pair': pair_id,
                    'volume': volume,
                    'ticker': ticker
                })
        
        return sorted(sorted_pairs, key=lambda x: x['volume'], reverse=True)[:limit]
    
    async def get_coin_stats(self, pair: str) -> Dict:
        ticker_data = await self.get_ticker(pair)
        if not ticker_data or pair not in ticker_data:
            return {}
            
        ticker = ticker_data[pair]
        
        return {
            "exchange": "kraken",
            "symbol": pair,
            "price": float(ticker['c'][0]),  # last trade price
            "volume_24h": float(ticker['v'][1]),  # 24h volume
            "volume_change_24h": 0,  # Not directly available
            "price_change_24h": 0,   # Calculate from open price
            "high_24h": float(ticker['h'][1]),  # 24h high
            "low_24h": float(ticker['l'][1]),   # 24h low
            "timestamp": datetime.now()
        }