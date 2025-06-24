# app/analysis/coin_recommendation/core/data_collector.py
"""
5개 거래소 통합 데이터 수집기
- OKX: USDT-SPOT, USDT-SWAP, BTC-SPOT
- 업비트: KRW-SPOT, USDT-SPOT  
- 실시간 거래량, 가격, 변동성 데이터 수집
"""

import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class CoinData:
    """코인 데이터 구조체"""
    symbol: str
    market: str
    price: float
    volume_24h: float
    volume_usdt: float  # USDT 환산 거래량
    change_1h: float
    change_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime

class MultiExchangeDataCollector:
    """다중 거래소 데이터 수집기"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.collect_interval = 60  # 1분마다 수집
        self.is_running = False
        
        # 거래소별 설정
        self.exchanges = {
            'okx': {
                'base_url': 'https://www.okx.com/api/v5',
                'markets': ['USDT-SPOT', 'USDT-SWAP', 'BTC-SPOT'],
                'rate_limit': 20  # requests per second
            },
            'upbit': {
                'base_url': 'https://api.upbit.com/v1',
                'markets': ['KRW-SPOT', 'USDT-SPOT'],
                'rate_limit': 10
            },
            'binance': {
                'base_url': 'https://api.binance.com/api/v3',
                'markets': ['USDT-SPOT'],
                'rate_limit': 20
            },
            'coinbase': {
                'base_url': 'https://api.exchange.coinbase.com',
                'markets': ['USD-SPOT'],
                'rate_limit': 10
            },
            'kraken': {
                'base_url': 'https://api.kraken.com/0/public',
                'markets': ['USD-SPOT'],
                'rate_limit': 15
            }
        }
        
        # 최근 데이터 캐시
        self.coin_cache: Dict[str, CoinData] = {}
        self.last_update: Dict[str, datetime] = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def start_collection(self):
        """데이터 수집 시작"""
        self.is_running = True
        logger.info("5개 거래소 데이터 수집 시작")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # 병렬로 모든 거래소 데이터 수집
                tasks = []
                for exchange in self.exchanges.keys():
                    tasks.append(self._collect_exchange_data(exchange))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 결과 처리
                total_coins = 0
                for i, result in enumerate(results):
                    exchange = list(self.exchanges.keys())[i]
                    if isinstance(result, Exception):
                        logger.error(f"{exchange} 데이터 수집 실패: {result}")
                    else:
                        total_coins += len(result)
                        logger.info(f"{exchange}: {len(result)}개 코인 수집 완료")
                
                # 수집 완료 로그
                elapsed = time.time() - start_time
                logger.info(f"총 {total_coins}개 코인 데이터 수집 완료 ({elapsed:.2f}초)")
                
                # 다음 수집까지 대기
                await asyncio.sleep(max(0, self.collect_interval - elapsed))
                
            except Exception as e:
                logger.error(f"데이터 수집 중 오류: {e}")
                await asyncio.sleep(5)

    async def _collect_exchange_data(self, exchange: str) -> List[CoinData]:
        """거래소별 데이터 수집"""
        if exchange == 'okx':
            return await self._collect_okx_data()
        elif exchange == 'upbit':
            return await self._collect_upbit_data()
        elif exchange == 'binance':
            return await self._collect_binance_data()
        elif exchange == 'coinbase':
            return await self._collect_coinbase_data()
        elif exchange == 'kraken':
            return await self._collect_kraken_data()
        else:
            return []

    async def _collect_okx_data(self) -> List[CoinData]:
        """OKX 데이터 수집"""
        coin_data = []
        
        try:
            # SPOT 마켓 데이터
            spot_url = f"{self.exchanges['okx']['base_url']}/market/tickers?instType=SPOT"
            async with self.session.get(spot_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for ticker in data.get('data', []):
                        if not ticker['instId'].endswith('-USDT'):
                            continue
                            
                        symbol = ticker['instId'].replace('-USDT', '')
                        
                        coin_data.append(CoinData(
                            symbol=symbol,
                            market='OKX_USDT_SPOT',
                            price=float(ticker['last']),
                            volume_24h=float(ticker['vol24h']),
                            volume_usdt=float(ticker['volCcy24h']),
                            change_1h=0.0,  # OKX는 1시간 데이터 별도 요청 필요
                            change_24h=float(ticker['changePercent']) * 100,
                            high_24h=float(ticker['high24h']),
                            low_24h=float(ticker['low24h']),
                            timestamp=datetime.now()
                        ))
            
            # SWAP (선물) 마켓 데이터
            swap_url = f"{self.exchanges['okx']['base_url']}/market/tickers?instType=SWAP"
            async with self.session.get(swap_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for ticker in data.get('data', []):
                        if not ticker['instId'].endswith('-USDT-SWAP'):
                            continue
                            
                        symbol = ticker['instId'].replace('-USDT-SWAP', '')
                        
                        coin_data.append(CoinData(
                            symbol=symbol,
                            market='OKX_USDT_SWAP',
                            price=float(ticker['last']),
                            volume_24h=float(ticker['vol24h']),
                            volume_usdt=float(ticker['volCcy24h']),
                            change_1h=0.0,
                            change_24h=float(ticker['changePercent']) * 100,
                            high_24h=float(ticker['high24h']),
                            low_24h=float(ticker['low24h']),
                            timestamp=datetime.now()
                        ))
                        
        except Exception as e:
            logger.error(f"OKX 데이터 수집 실패: {e}")
            
        return coin_data

    async def _collect_upbit_data(self) -> List[CoinData]:
        """업비트 데이터 수집"""
        coin_data = []
        
        try:
            # 업비트 마켓 코드 조회
            markets_url = f"{self.exchanges['upbit']['base_url']}/market/all"
            async with self.session.get(markets_url) as response:
                if response.status == 200:
                    markets = await response.json()
                    
                    # KRW 마켓만 필터링
                    krw_markets = [m['market'] for m in markets if m['market'].startswith('KRW-')]
                    
                    # 티커 정보 조회 (한 번에 최대 100개씩)
                    for i in range(0, len(krw_markets), 100):
                        market_batch = krw_markets[i:i+100]
                        markets_param = ','.join(market_batch)
                        
                        ticker_url = f"{self.exchanges['upbit']['base_url']}/ticker?markets={markets_param}"
                        async with self.session.get(ticker_url) as ticker_response:
                            if ticker_response.status == 200:
                                tickers = await ticker_response.json()
                                
                                for ticker in tickers:
                                    symbol = ticker['market'].replace('KRW-', '')
                                    
                                    coin_data.append(CoinData(
                                        symbol=symbol,
                                        market='UPBIT_KRW_SPOT',
                                        price=float(ticker['trade_price']),
                                        volume_24h=float(ticker['acc_trade_volume_24h']),
                                        volume_usdt=float(ticker['acc_trade_price_24h']) / 1300,  # 대략적인 KRW->USD 환산
                                        change_1h=0.0,  # 업비트는 1시간 데이터 없음
                                        change_24h=float(ticker['change_rate']) * 100,
                                        high_24h=float(ticker['high_price']),
                                        low_24h=float(ticker['low_price']),
                                        timestamp=datetime.now()
                                    ))
                        
                        # 레이트 리미트 준수
                        await asyncio.sleep(0.1)
                        
        except Exception as e:
            logger.error(f"업비트 데이터 수집 실패: {e}")
            
        return coin_data

    async def _collect_binance_data(self) -> List[CoinData]:
        """바이낸스 데이터 수집"""
        coin_data = []
        
        try:
            ticker_url = f"{self.exchanges['binance']['base_url']}/ticker/24hr"
            async with self.session.get(ticker_url) as response:
                if response.status == 200:
                    tickers = await response.json()
                    
                    for ticker in tickers:
                        if not ticker['symbol'].endswith('USDT'):
                            continue
                            
                        symbol = ticker['symbol'].replace('USDT', '')
                        
                        coin_data.append(CoinData(
                            symbol=symbol,
                            market='BINANCE_USDT_SPOT',
                            price=float(ticker['lastPrice']),
                            volume_24h=float(ticker['volume']),
                            volume_usdt=float(ticker['quoteVolume']),
                            change_1h=0.0,  # 별도 API 호출 필요
                            change_24h=float(ticker['priceChangePercent']),
                            high_24h=float(ticker['highPrice']),
                            low_24h=float(ticker['lowPrice']),
                            timestamp=datetime.now()
                        ))
                        
        except Exception as e:
            logger.error(f"바이낸스 데이터 수집 실패: {e}")
            
        return coin_data

    async def _collect_coinbase_data(self) -> List[CoinData]:
        """코인베이스 데이터 수집"""
        coin_data = []
        
        try:
            # 코인베이스는 개별 상품 조회 필요
            products_url = f"{self.exchanges['coinbase']['base_url']}/products"
            async with self.session.get(products_url) as response:
                if response.status == 200:
                    products = await response.json()
                    
                    # USD 마켓만 필터링
                    usd_products = [p for p in products if p['quote_currency'] == 'USD' and p['status'] == 'online']
                    
                    # 각 상품의 24시간 통계 수집
                    for product in usd_products[:50]:  # API 제한으로 상위 50개만
                        product_id = product['id']
                        stats_url = f"{self.exchanges['coinbase']['base_url']}/products/{product_id}/stats"
                        
                        async with self.session.get(stats_url) as stats_response:
                            if stats_response.status == 200:
                                stats = await stats_response.json()
                                
                                symbol = product['base_currency']
                                
                                coin_data.append(CoinData(
                                    symbol=symbol,
                                    market='COINBASE_USD_SPOT',
                                    price=float(stats['last']),
                                    volume_24h=float(stats['volume']),
                                    volume_usdt=float(stats['volume']) * float(stats['last']),
                                    change_1h=0.0,
                                    change_24h=((float(stats['last']) - float(stats['open'])) / float(stats['open'])) * 100,
                                    high_24h=float(stats['high']),
                                    low_24h=float(stats['low']),
                                    timestamp=datetime.now()
                                ))
                        
                        await asyncio.sleep(0.1)  # 레이트 리미트
                        
        except Exception as e:
            logger.error(f"코인베이스 데이터 수집 실패: {e}")
            
        return coin_data

    async def _collect_kraken_data(self) -> List[CoinData]:
        """크라켄 데이터 수집"""
        coin_data = []
        
        try:
            ticker_url = f"{self.exchanges['kraken']['base_url']}/Ticker"
            async with self.session.get(ticker_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for pair_name, ticker_data in data.get('result', {}).items():
                        if not pair_name.endswith('USD'):
                            continue
                            
                        symbol = pair_name.replace('USD', '').replace('X', '').replace('Z', '')
                        
                        coin_data.append(CoinData(
                            symbol=symbol,
                            market='KRAKEN_USD_SPOT',
                            price=float(ticker_data['c'][0]),  # 마지막 거래 가격
                            volume_24h=float(ticker_data['v'][1]),  # 24시간 거래량
                            volume_usdt=float(ticker_data['v'][1]) * float(ticker_data['c'][0]),
                            change_1h=0.0,
                            change_24h=((float(ticker_data['c'][0]) - float(ticker_data['o'])) / float(ticker_data['o'])) * 100,
                            high_24h=float(ticker_data['h'][1]),
                            low_24h=float(ticker_data['l'][1]),
                            timestamp=datetime.now()
                        ))
                        
        except Exception as e:
            logger.error(f"크라켄 데이터 수집 실패: {e}")
            
        return coin_data

    def get_coin_data(self, symbol: str, market: str = None) -> Optional[CoinData]:
        """특정 코인 데이터 조회"""
        key = f"{symbol}:{market}" if market else symbol
        return self.coin_cache.get(key)

    def get_all_coins(self, market_filter: str = None) -> List[CoinData]:
        """모든 코인 데이터 조회"""
        if market_filter:
            return [data for data in self.coin_cache.values() if data.market == market_filter]
        return list(self.coin_cache.values())

    def get_market_summary(self) -> Dict[str, int]:
        """마켓별 코인 수 요약"""
        summary = {}
        for coin_data in self.coin_cache.values():
            market = coin_data.market
            summary[market] = summary.get(market, 0) + 1
        return summary

    def stop_collection(self):
        """데이터 수집 중지"""
        self.is_running = False
        logger.info("데이터 수집 중지됨")


# 사용 예시
async def main():
    """테스트 실행"""
    async with MultiExchangeDataCollector() as collector:
        # 데이터 수집 시작 (백그라운드)
        collection_task = asyncio.create_task(collector.start_collection())
        
        # 10분 후 수집 중지
        await asyncio.sleep(600)
        collector.stop_collection()
        await collection_task

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())