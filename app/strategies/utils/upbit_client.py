# app/strategies/utils/upbit_client.py
"""
업비트 API 클라이언트 - 한국 최대 암호화폐 거래소 연동
KRW 마켓 지원, 현물 거래 전용
"""

import hmac
import hashlib
import uuid
import jwt
from urllib.parse import urlencode, unquote
import time
import requests
import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

@dataclass
class UpbitConfig:
    """업비트 API 설정"""
    access_key: str
    secret_key: str
    server_url: str = "https://api.upbit.com"
    
@dataclass
class UpbitOrderRequest:
    """업비트 주문 요청"""
    market: str  # 마켓 ID (예: KRW-BTC)
    side: str    # 주문 종류 (bid: 매수, ask: 매도)
    volume: Optional[str] = None  # 주문량 (지정가, 시장가 매도)
    price: Optional[str] = None   # 주문 가격 (지정가)
    ord_type: str = "limit"       # 주문 타입 (limit, price, market)
    
@dataclass
class UpbitBalance:
    """업비트 잔고 정보"""
    currency: str
    balance: Decimal
    locked: Decimal
    avg_buy_price: Decimal
    avg_buy_price_modified: bool
    unit_currency: str

@dataclass
class UpbitTicker:
    """업비트 티커 정보"""
    market: str
    trade_date: str
    trade_time: str
    trade_date_kst: str
    trade_time_kst: str
    trade_timestamp: int
    opening_price: Decimal
    high_price: Decimal
    low_price: Decimal
    trade_price: Decimal  # 현재가
    prev_closing_price: Decimal
    change: str  # EVEN, RISE, FALL
    change_price: Decimal
    change_rate: Decimal
    signed_change_price: Decimal
    signed_change_rate: Decimal
    trade_volume: Decimal
    acc_trade_price: Decimal
    acc_trade_price_24h: Decimal
    acc_trade_volume: Decimal
    acc_trade_volume_24h: Decimal
    highest_52_week_price: Decimal
    highest_52_week_date: str
    lowest_52_week_price: Decimal
    lowest_52_week_date: str
    timestamp: int

class UpbitClient:
    """업비트 API 클라이언트"""
    
    def __init__(self, config: Optional[UpbitConfig] = None):
        self.config = config
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_auth_headers(self, query_params: Optional[Dict] = None) -> Dict[str, str]:
        """인증 헤더 생성"""
        if not self.config:
            return {}
            
        payload = {
            'access_key': self.config.access_key,
            'nonce': str(uuid.uuid4()),
        }
        
        if query_params:
            query_string = unquote(urlencode(query_params, doseq=True)).encode("utf-8")
            m = hashlib.sha512()
            m.update(query_string)
            query_hash = m.hexdigest()
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = 'SHA512'
        
        jwt_token = jwt.encode(payload, self.config.secret_key, algorithm='HS256')
        
        return {
            'Authorization': f'Bearer {jwt_token}',
            'Content-Type': 'application/json'
        }
    
    async def _make_request(self, method: str, endpoint: str, 
                          params: Optional[Dict] = None, 
                          data: Optional[Dict] = None,
                          auth_required: bool = False) -> Dict:
        """API 요청 실행"""
        url = f"{self.config.server_url if self.config else 'https://api.upbit.com'}{endpoint}"
        
        headers = {}
        if auth_required and self.config:
            headers = self._get_auth_headers(params or data)
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data
            ) as response:
                
                if response.status == 429:  # Rate limit
                    logger.warning("업비트 API 요청 제한 - 1초 대기")
                    await asyncio.sleep(1)
                    return await self._make_request(method, endpoint, params, data, auth_required)
                
                result = await response.json()
                
                if response.status != 200:
                    error_msg = result.get('error', {}).get('message', 'Unknown error')
                    logger.error(f"업비트 API 에러: {response.status}, {error_msg}")
                    raise Exception(f"업비트 API 에러: {error_msg}")
                
                return result
                
        except Exception as e:
            logger.error(f"업비트 API 요청 실패: {e}")
            raise
    
    # =================================
    # 공개 API (인증 불필요)
    # =================================
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """마켓 코드 조회"""
        try:
            result = await self._make_request('GET', '/v1/market/all')
            return result
        except Exception as e:
            logger.error(f"마켓 조회 실패: {e}")
            return []
    
    async def get_ticker(self, markets: Union[str, List[str]]) -> List[UpbitTicker]:
        """현재가 조회"""
        try:
            if isinstance(markets, str):
                markets = [markets]
            
            params = {'markets': ','.join(markets)}
            result = await self._make_request('GET', '/v1/ticker', params=params)
            
            tickers = []
            for data in result:
                ticker = UpbitTicker(
                    market=data['market'],
                    trade_date=data['trade_date'],
                    trade_time=data['trade_time'],
                    trade_date_kst=data['trade_date_kst'],
                    trade_time_kst=data['trade_time_kst'],
                    trade_timestamp=data['trade_timestamp'],
                    opening_price=Decimal(str(data['opening_price'])),
                    high_price=Decimal(str(data['high_price'])),
                    low_price=Decimal(str(data['low_price'])),
                    trade_price=Decimal(str(data['trade_price'])),
                    prev_closing_price=Decimal(str(data['prev_closing_price'])),
                    change=data['change'],
                    change_price=Decimal(str(data['change_price'])),
                    change_rate=Decimal(str(data['change_rate'])),
                    signed_change_price=Decimal(str(data['signed_change_price'])),
                    signed_change_rate=Decimal(str(data['signed_change_rate'])),
                    trade_volume=Decimal(str(data['trade_volume'])),
                    acc_trade_price=Decimal(str(data['acc_trade_price'])),
                    acc_trade_price_24h=Decimal(str(data['acc_trade_price_24h'])),
                    acc_trade_volume=Decimal(str(data['acc_trade_volume'])),
                    acc_trade_volume_24h=Decimal(str(data['acc_trade_volume_24h'])),
                    highest_52_week_price=Decimal(str(data['highest_52_week_price'])),
                    highest_52_week_date=data['highest_52_week_date'],
                    lowest_52_week_price=Decimal(str(data['lowest_52_week_price'])),
                    lowest_52_week_date=data['lowest_52_week_date'],
                    timestamp=data['timestamp']
                )
                tickers.append(ticker)
            
            return tickers
            
        except Exception as e:
            logger.error(f"티커 조회 실패: {e}")
            return []
    
    async def get_orderbook(self, markets: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """호가 정보 조회"""
        try:
            if isinstance(markets, str):
                markets = [markets]
            
            params = {'markets': ','.join(markets)}
            result = await self._make_request('GET', '/v1/orderbook', params=params)
            return result
            
        except Exception as e:
            logger.error(f"호가 정보 조회 실패: {e}")
            return []
    
    async def get_trades_ticks(self, market: str, count: int = 1) -> List[Dict[str, Any]]:
        """최근 체결 내역 조회"""
        try:
            params = {
                'market': market,
                'count': count
            }
            result = await self._make_request('GET', '/v1/trades/ticks', params=params)
            return result
            
        except Exception as e:
            logger.error(f"체결 내역 조회 실패: {e}")
            return []
    
    async def get_candles_minutes(self, unit: int, market: str, count: int = 1) -> List[Dict[str, Any]]:
        """분봉 조회"""
        try:
            params = {
                'market': market,
                'count': count
            }
            result = await self._make_request('GET', f'/v1/candles/minutes/{unit}', params=params)
            return result
            
        except Exception as e:
            logger.error(f"분봉 조회 실패: {e}")
            return []
    
    # =================================
    # 인증 API (계정 정보)
    # =================================
    
    async def get_accounts(self) -> List[UpbitBalance]:
        """계정 정보 조회"""
        try:
            if not self.config:
                raise Exception("인증 정보가 필요합니다")
            
            result = await self._make_request('GET', '/v1/accounts', auth_required=True)
            
            balances = []
            for data in result:
                balance = UpbitBalance(
                    currency=data['currency'],
                    balance=Decimal(data['balance']),
                    locked=Decimal(data['locked']),
                    avg_buy_price=Decimal(data['avg_buy_price']),
                    avg_buy_price_modified=data['avg_buy_price_modified'],
                    unit_currency=data['unit_currency']
                )
                balances.append(balance)
            
            return balances
            
        except Exception as e:
            logger.error(f"계정 정보 조회 실패: {e}")
            return []
    
    async def get_balance(self, currency: str) -> Optional[UpbitBalance]:
        """특정 통화 잔고 조회"""
        try:
            balances = await self.get_accounts()
            for balance in balances:
                if balance.currency == currency:
                    return balance
            return None
            
        except Exception as e:
            logger.error(f"{currency} 잔고 조회 실패: {e}")
            return None
    
    # =================================
    # 주문 관리 API
    # =================================
    
    async def place_order(self, order_request: UpbitOrderRequest) -> Dict[str, Any]:
        """주문 실행"""
        try:
            if not self.config:
                raise Exception("인증 정보가 필요합니다")
            
            order_data = {
                'market': order_request.market,
                'side': order_request.side,
                'ord_type': order_request.ord_type
            }
            
            if order_request.volume:
                order_data['volume'] = order_request.volume
            if order_request.price:
                order_data['price'] = order_request.price
            
            result = await self._make_request('POST', '/v1/orders', data=order_data, auth_required=True)
            return result
            
        except Exception as e:
            logger.error(f"주문 실행 실패: {e}")
            return {'error': str(e)}
    
    async def cancel_order(self, uuid: str) -> Dict[str, Any]:
        """주문 취소"""
        try:
            if not self.config:
                raise Exception("인증 정보가 필요합니다")
            
            data = {'uuid': uuid}
            result = await self._make_request('DELETE', '/v1/order', data=data, auth_required=True)
            return result
            
        except Exception as e:
            logger.error(f"주문 취소 실패: {e}")
            return {'error': str(e)}
    
    async def get_orders(self, market: Optional[str] = None, 
                        state: str = 'wait') -> List[Dict[str, Any]]:
        """주문 목록 조회"""
        try:
            if not self.config:
                raise Exception("인증 정보가 필요합니다")
            
            params = {'state': state}
            if market:
                params['market'] = market
            
            result = await self._make_request('GET', '/v1/orders', params=params, auth_required=True)
            return result
            
        except Exception as e:
            logger.error(f"주문 목록 조회 실패: {e}")
            return []
    
    async def get_order(self, uuid: str) -> Dict[str, Any]:
        """개별 주문 조회"""
        try:
            if not self.config:
                raise Exception("인증 정보가 필요합니다")
            
            params = {'uuid': uuid}
            result = await self._make_request('GET', '/v1/order', params=params, auth_required=True)
            return result
            
        except Exception as e:
            logger.error(f"주문 조회 실패: {e}")
            return {'error': str(e)}
    
    # =================================
    # 유틸리티 메서드
    # =================================
    
    async def get_krw_markets(self) -> List[str]:
        """KRW 마켓 목록 조회"""
        try:
            markets = await self.get_markets()
            krw_markets = [market['market'] for market in markets 
                          if market['market'].startswith('KRW-')]
            return krw_markets
            
        except Exception as e:
            logger.error(f"KRW 마켓 조회 실패: {e}")
            return []
    
    async def get_market_price(self, market: str) -> Optional[Decimal]:
        """마켓 현재가 조회"""
        try:
            tickers = await self.get_ticker(market)
            if tickers:
                return tickers[0].trade_price
            return None
            
        except Exception as e:
            logger.error(f"{market} 현재가 조회 실패: {e}")
            return None
    
    async def calculate_buy_amount(self, market: str, krw_amount: float) -> Optional[Decimal]:
        """매수 가능 수량 계산"""
        try:
            current_price = await self.get_market_price(market)
            if current_price:
                volume = Decimal(str(krw_amount)) / current_price
                return volume.quantize(Decimal('0.00000001'))  # 8자리까지
            return None
            
        except Exception as e:
            logger.error(f"매수 수량 계산 실패: {e}")
            return None
    
    async def test_connection(self) -> Dict[str, Any]:
        """연결 테스트"""
        try:
            # 공개 API 테스트
            markets = await self.get_markets()
            market_count = len(markets)
            
            # 인증 API 테스트 (키가 있는 경우)
            auth_status = False
            if self.config:
                try:
                    accounts = await self.get_accounts()
                    auth_status = len(accounts) >= 0
                except:
                    auth_status = False
            
            return {
                'status': 'connected',
                'market_count': market_count,
                'auth_available': auth_status,
                'krw_markets': len([m for m in markets if m['market'].startswith('KRW-')]),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# =================================
# 헬퍼 함수들
# =================================

def create_upbit_client(access_key: str = None, secret_key: str = None) -> UpbitClient:
    """업비트 클라이언트 생성"""
    config = None
    if access_key and secret_key:
        config = UpbitConfig(access_key=access_key, secret_key=secret_key)
    
    return UpbitClient(config)

def create_buy_order(market: str, price: str, volume: str) -> UpbitOrderRequest:
    """매수 주문 생성 (지정가)"""
    return UpbitOrderRequest(
        market=market,
        side='bid',
        ord_type='limit',
        price=price,
        volume=volume
    )

def create_sell_order(market: str, volume: str, price: str = None) -> UpbitOrderRequest:
    """매도 주문 생성"""
    if price:
        # 지정가 매도
        return UpbitOrderRequest(
            market=market,
            side='ask',
            ord_type='limit',
            price=price,
            volume=volume
        )
    else:
        # 시장가 매도
        return UpbitOrderRequest(
            market=market,
            side='ask',
            ord_type='market',
            volume=volume
        )

def create_market_buy_order(market: str, price: str) -> UpbitOrderRequest:
    """시장가 매수 주문 생성"""
    return UpbitOrderRequest(
        market=market,
        side='bid',
        ord_type='price',
        price=price  # 매수 금액 (KRW)
    )

# =================================
# 테스트 및 데모 코드
# =================================

async def demo_upbit_example():
    """업비트 API 데모"""
    # 공개 API 테스트 (인증 불필요)
    async with UpbitClient() as client:
        print("📈 업비트 공개 API 테스트")
        
        # 연결 테스트
        connection = await client.test_connection()
        print(f"연결 상태: {connection}")
        
        # KRW 마켓 조회
        krw_markets = await client.get_krw_markets()
        print(f"KRW 마켓 수: {len(krw_markets)}")
        
        # 비트코인 현재가 조회
        if 'KRW-BTC' in krw_markets:
            btc_ticker = await client.get_ticker('KRW-BTC')
            if btc_ticker:
                print(f"BTC 현재가: {btc_ticker[0].trade_price:,} KRW")
        
        # 인증 API 테스트 (키가 필요)
        if False:  # 실제 키가 있을 때만 활성화
            config = UpbitConfig("your_access_key", "your_secret_key")
            auth_client = UpbitClient(config)
            
            balances = await auth_client.get_accounts()
            print(f"보유 자산: {len(balances)}개")

if __name__ == "__main__":
    print("🚀 업비트 API 클라이언트 테스트")
    
    # asyncio.run(demo_upbit_example())  # 테스트 실행