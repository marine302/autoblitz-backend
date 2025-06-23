# app/strategies/utils/okx_private_client.py
"""
OKX Private API 클라이언트 - 실제 거래 실행을 위한 Private API 연동
선물 거래, 주문 실행, 포지션 관리 등의 기능 제공
"""

import hmac
import hashlib
import base64
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
from dataclasses import dataclass
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

@dataclass
class OKXConfig:
    """OKX API 설정"""
    api_key: str
    secret_key: str
    passphrase: str
    sandbox: bool = True  # 기본값: 샌드박스 모드
    
    @property
    def base_url(self) -> str:
        return "https://www.okx.com" if not self.sandbox else "https://www.okx.com"
    
    @property
    def environment(self) -> str:
        return "sandbox" if self.sandbox else "production"

@dataclass
class OrderRequest:
    """주문 요청 데이터"""
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'post_only', 'fok', 'ioc'
    size: str  # 주문 수량
    price: Optional[str] = None  # 지정가 주문시 가격
    client_order_id: Optional[str] = None
    
@dataclass
class PositionInfo:
    """포지션 정보"""
    symbol: str
    position_side: str  # 'long', 'short', 'net'
    size: Decimal
    available_size: Decimal
    average_price: Decimal
    unrealized_pnl: Decimal
    percentage: Decimal
    leverage: int
    margin_mode: str
    
class OKXPrivateClient:
    """OKX Private API 클라이언트"""
    
    def __init__(self, config: OKXConfig):
        self.config = config
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """API 서명 생성"""
        message = timestamp + method.upper() + request_path + body
        signature = hmac.new(
            self.config.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """API 요청 헤더 생성"""
        timestamp = datetime.now(timezone.utc).isoformat()[:-6] + 'Z'
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        return {
            'OK-ACCESS-KEY': self.config.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.config.passphrase,
            'Content-Type': 'application/json',
            'x-simulated-trading': '1' if self.config.sandbox else '0'
        }
    
    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
        """API 요청 실행"""
        url = f"{self.config.base_url}{endpoint}"
        body = json.dumps(data) if data else ""
        headers = self._get_headers(method, endpoint, body)
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=body if body else None
            ) as response:
                result = await response.json()
                
                if response.status != 200:
                    logger.error(f"API 요청 실패: {response.status}, {result}")
                    raise Exception(f"API Error: {result}")
                    
                if result.get('code') != '0':
                    logger.error(f"OKX API 에러: {result}")
                    raise Exception(f"OKX Error: {result.get('msg', 'Unknown error')}")
                    
                return result
                
        except Exception as e:
            logger.error(f"API 요청 중 에러 발생: {e}")
            raise
    
    # =================================
    # 계정 정보 관련 메서드
    # =================================
    
    async def get_account_info(self) -> Dict[str, Any]:
        """계정 정보 조회"""
        try:
            result = await self._make_request('GET', '/api/v5/account/config')
            return {
                'account_level': result['data'][0].get('acctLv', 'unknown'),
                'position_mode': result['data'][0].get('posMode', 'unknown'),
                'auto_loan': result['data'][0].get('autoLoan', False),
                'uid': result['data'][0].get('uid', 'unknown')
            }
        except Exception as e:
            logger.error(f"계정 정보 조회 실패: {e}")
            return {'error': str(e)}
    
    async def get_balance(self, currency: str = None) -> Dict[str, Any]:
        """잔고 조회"""
        try:
            params = {'ccy': currency} if currency else {}
            result = await self._make_request('GET', '/api/v5/account/balance', params=params)
            
            balances = {}
            for item in result['data'][0]['details']:
                balances[item['ccy']] = {
                    'available': float(item['availBal']),
                    'frozen': float(item['frozenBal']),
                    'total': float(item['bal'])
                }
            return balances
            
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            return {'error': str(e)}
    
    # =================================
    # 포지션 관리 관련 메서드
    # =================================
    
    async def get_positions(self, symbol: str = None) -> List[PositionInfo]:
        """포지션 조회"""
        try:
            params = {'instId': symbol} if symbol else {}
            result = await self._make_request('GET', '/api/v5/account/positions', params=params)
            
            positions = []
            for pos_data in result['data']:
                if float(pos_data['pos']) != 0:  # 포지션이 있는 경우만
                    position = PositionInfo(
                        symbol=pos_data['instId'],
                        position_side=pos_data['posSide'],
                        size=Decimal(pos_data['pos']),
                        available_size=Decimal(pos_data['availPos']),
                        average_price=Decimal(pos_data['avgPx']),
                        unrealized_pnl=Decimal(pos_data['upl']),
                        percentage=Decimal(pos_data['uplRatio']) * 100,
                        leverage=int(pos_data['lever']),
                        margin_mode=pos_data['mgnMode']
                    )
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"포지션 조회 실패: {e}")
            return []
    
    async def set_leverage(self, symbol: str, leverage: int, position_side: str = "cross") -> bool:
        """레버리지 설정"""
        try:
            data = {
                'instId': symbol,
                'lever': str(leverage),
                'mgnMode': position_side
            }
            
            result = await self._make_request('POST', '/api/v5/account/set-leverage', data=data)
            return True
            
        except Exception as e:
            logger.error(f"레버리지 설정 실패: {e}")
            return False
    
    # =================================
    # 주문 관리 관련 메서드  
    # =================================
    
    async def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        """주문 실행"""
        try:
            data = {
                'instId': order.symbol,
                'tdMode': 'cross',  # cross margin
                'side': order.side,
                'ordType': order.order_type,
                'sz': order.size
            }
            
            if order.price:
                data['px'] = order.price
            
            if order.client_order_id:
                data['clOrdId'] = order.client_order_id
            
            result = await self._make_request('POST', '/api/v5/trade/order', data=data)
            
            if result['data']:
                order_info = result['data'][0]
                return {
                    'order_id': order_info['ordId'],
                    'client_order_id': order_info.get('clOrdId'),
                    'status': 'submitted',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'error': 'Order placement failed', 'details': result}
                
        except Exception as e:
            logger.error(f"주문 실행 실패: {e}")
            return {'error': str(e)}
    
    async def cancel_order(self, symbol: str, order_id: str = None, client_order_id: str = None) -> bool:
        """주문 취소"""
        try:
            data = {'instId': symbol}
            
            if order_id:
                data['ordId'] = order_id
            elif client_order_id:
                data['clOrdId'] = client_order_id
            else:
                raise ValueError("주문 ID 또는 클라이언트 주문 ID가 필요합니다")
            
            result = await self._make_request('POST', '/api/v5/trade/cancel-order', data=data)
            return True
            
        except Exception as e:
            logger.error(f"주문 취소 실패: {e}")
            return False
    
    async def get_order_status(self, symbol: str, order_id: str = None, client_order_id: str = None) -> Dict[str, Any]:
        """주문 상태 조회"""
        try:
            params = {'instId': symbol}
            
            if order_id:
                params['ordId'] = order_id
            elif client_order_id:
                params['clOrdId'] = client_order_id
            else:
                raise ValueError("주문 ID 또는 클라이언트 주문 ID가 필요합니다")
            
            result = await self._make_request('GET', '/api/v5/trade/order', params=params)
            
            if result['data']:
                order_data = result['data'][0]
                return {
                    'order_id': order_data['ordId'],
                    'symbol': order_data['instId'],
                    'side': order_data['side'],
                    'type': order_data['ordType'],
                    'size': float(order_data['sz']),
                    'filled_size': float(order_data['fillSz']),
                    'price': float(order_data['px']) if order_data['px'] else None,
                    'average_price': float(order_data['avgPx']) if order_data['avgPx'] else None,
                    'status': order_data['state'],
                    'fee': float(order_data['fee']),
                    'created_at': order_data['cTime'],
                    'updated_at': order_data['uTime']
                }
            else:
                return {'error': 'Order not found'}
                
        except Exception as e:
            logger.error(f"주문 상태 조회 실패: {e}")
            return {'error': str(e)}
    
    # =================================
    # 시장 데이터 관련 메서드
    # =================================
    
    async def get_instruments(self, instrument_type: str = "SWAP") -> List[Dict[str, Any]]:
        """거래 가능한 상품 조회 (선물)"""
        try:
            params = {'instType': instrument_type}
            result = await self._make_request('GET', '/api/v5/public/instruments', params=params)
            
            instruments = []
            for inst in result['data']:
                instruments.append({
                    'symbol': inst['instId'],
                    'base_currency': inst['baseCcy'],
                    'quote_currency': inst['quoteCcy'],
                    'contract_value': float(inst['ctVal']),
                    'min_size': float(inst['minSz']),
                    'tick_size': float(inst['tickSz']),
                    'status': inst['state']
                })
            
            return instruments
            
        except Exception as e:
            logger.error(f"상품 조회 실패: {e}")
            return []
    
    # =================================
    # 유틸리티 메서드
    # =================================
    
    async def test_connection(self) -> Dict[str, Any]:
        """연결 테스트"""
        try:
            account_info = await self.get_account_info()
            balance = await self.get_balance('USDT')
            
            return {
                'status': 'connected',
                'environment': self.config.environment,
                'account_level': account_info.get('account_level'),
                'usdt_balance': balance.get('USDT', {}).get('available', 0),
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

def create_okx_private_client(api_key: str, secret_key: str, passphrase: str, sandbox: bool = True) -> OKXPrivateClient:
    """OKX Private 클라이언트 생성"""
    config = OKXConfig(
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
        sandbox=sandbox
    )
    return OKXPrivateClient(config)

def create_market_order(symbol: str, side: str, size: str, client_order_id: str = None) -> OrderRequest:
    """시장가 주문 생성"""
    return OrderRequest(
        symbol=symbol,
        side=side,
        order_type='market',
        size=size,
        client_order_id=client_order_id
    )

def create_limit_order(symbol: str, side: str, size: str, price: str, client_order_id: str = None) -> OrderRequest:
    """지정가 주문 생성"""
    return OrderRequest(
        symbol=symbol,
        side=side,
        order_type='limit',
        size=size,
        price=price,
        client_order_id=client_order_id
    )

# =================================
# 테스트 및 데모 코드
# =================================

async def demo_trading_example():
    """데모 거래 예시"""
    # 주의: 실제 API 키가 필요합니다
    config = OKXConfig(
        api_key="your_api_key",
        secret_key="your_secret_key", 
        passphrase="your_passphrase",
        sandbox=True
    )
    
    async with OKXPrivateClient(config) as client:
        # 연결 테스트
        connection_status = await client.test_connection()
        print(f"연결 상태: {connection_status}")
        
        # 계정 정보 조회
        account_info = await client.get_account_info()
        print(f"계정 정보: {account_info}")
        
        # 잔고 조회
        balance = await client.get_balance()
        print(f"잔고: {balance}")
        
        # 포지션 조회
        positions = await client.get_positions()
        print(f"현재 포지션: {len(positions)}개")
        
        # 시장가 매수 주문 (데모)
        order = create_market_order('BTC-USDT-SWAP', 'buy', '0.01')
        result = await client.place_order(order)
        print(f"주문 결과: {result}")

if __name__ == "__main__":
    # 테스트 실행 (실제 API 키 필요)
    print("🔧 OKX Private API 클라이언트 테스트")
    print("실제 사용을 위해서는 API 키 설정이 필요합니다.")
    
    # asyncio.run(demo_trading_example())  # 실제 키가 있을 때만 실행