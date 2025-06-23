"""
OKX API Client - 수정된 버전 (올바른 엔드포인트 사용)
"""
import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class OKXClientFixed:
    """수정된 OKX 거래소 API 클라이언트"""
    
    def __init__(self):
        self.base_url = "https://www.okx.com"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_multiple_tickers(self, symbols: List[str] = None) -> List[Dict]:
        """모든 또는 특정 코인의 티커 정보 조회"""
        try:
            # 올바른 엔드포인트 사용
            url = f"{self.base_url}/api/v5/market/tickers"
            params = {'instType': 'SPOT'}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    tickers = data.get('data', [])
                    
                    result = []
                    for ticker in tickers:
                        # USDT 페어만 필터링
                        if ticker['instId'].endswith('-USDT'):
                            try:
                                # 데이터 변환
                                processed_ticker = {
                                    'symbol': ticker['instId'],
                                    'price': float(ticker['last']),
                                    'volume_24h': float(ticker['vol24h']),
                                    'change_24h': ((float(ticker['last']) - float(ticker['open24h'])) / float(ticker['open24h'])) * 100 if float(ticker['open24h']) > 0 else 0,
                                    'high_24h': float(ticker['high24h']),
                                    'low_24h': float(ticker['low24h']),
                                    'timestamp': datetime.now()
                                }
                                result.append(processed_ticker)
                            except (ValueError, KeyError) as e:
                                logger.warning(f"데이터 처리 오류 {ticker['instId']}: {e}")
                                continue
                    
                    # 심볼 필터링 (요청된 심볼이 있다면)
                    if symbols:
                        result = [t for t in result if t['symbol'] in symbols]
                    
                    # 거래량 순으로 정렬
                    result.sort(key=lambda x: x['volume_24h'], reverse=True)
                    
                    return result
                else:
                    logger.error(f"API 호출 실패: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"티커 조회 오류: {e}")
            return []

class FixedMarketScanner:
    """수정된 실시간 시장 스캐너"""
    
    def __init__(self):
        self.okx_client = None
        self.scan_count = 0
        
    async def initialize(self):
        self.okx_client = OKXClientFixed()
        await self.okx_client.__aenter__()
        print("✅ 수정된 OKX 클라이언트 초기화 완료")
        
    async def cleanup(self):
        if self.okx_client:
            await self.okx_client.__aexit__(None, None, None)
    
    def check_signal(self, ticker: Dict) -> Optional[Dict]:
        """완화된 신호 조건 체크"""
        try:
            # 매우 관대한 조건
            conditions = {
                'has_volume': ticker['volume_24h'] > 100,  # 최소 거래량
                'price_movement': abs(ticker['change_24h']) > 0.5,  # 0.5% 이상 변동
                'reasonable_price': ticker['price'] > 0.001,  # 합리적 가격
            }
            
            met_conditions = sum(conditions.values())
            
            if met_conditions >= 2:
                return {
                    'symbol': ticker['symbol'],
                    'signal_type': 'BUY' if ticker['change_24h'] < 0 else 'SELL',
                    'confidence': met_conditions / len(conditions),
                    'conditions_met': [k for k, v in conditions.items() if v],
                    'market_data': ticker
                }
        except Exception as e:
            logger.warning(f"신호 체크 오류 {ticker.get('symbol', 'unknown')}: {e}")
        
        return None
    
    async def scan_top_coins(self, limit: int = 20):
        """상위 코인들 스캔"""
        try:
            print(f"🔍 상위 {limit}개 코인 스캔 시작...")
            
            # 모든 USDT 페어 가져오기
            all_tickers = await self.okx_client.get_multiple_tickers()
            
            if not all_tickers:
                print("❌ 티커 데이터를 가져올 수 없습니다")
                return []
            
            # 상위 코인만 선택
            top_tickers = all_tickers[:limit]
            
            print(f"📊 {len(top_tickers)}개 코인 데이터 수집 완료")
            
            # 신호 체크
            signals = []
            for ticker in top_tickers:
                signal = self.check_signal(ticker)
                if signal:
                    signals.append(signal)
            
            # 결과 출력
            print(f"\n🎯 스캔 결과 - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 80)
            
            if signals:
                for signal in signals:
                    market = signal['market_data']
                    print(f"📈 {signal['symbol']:12} | "
                          f"${market['price']:8.2f} | "
                          f"{market['change_24h']:+6.2f}% | "
                          f"Vol: {market['volume_24h']:10,.0f} | "
                          f"조건: {', '.join(signal['conditions_met'])}")
                print(f"\n총 {len(signals)}개 신호 발견!")
            else:
                print("❌ 조건 충족 신호 없음")
                # 상위 5개라도 출력
                print("\n📊 상위 5개 코인 현황:")
                for i, ticker in enumerate(top_tickers[:5], 1):
                    print(f"{i}. {ticker['symbol']:12} | "
                          f"${ticker['price']:8.2f} | "
                          f"{ticker['change_24h']:+6.2f}% | "
                          f"Vol: {ticker['volume_24h']:10,.0f}")
            
            print("=" * 80)
            return signals
            
        except Exception as e:
            print(f"❌ 스캔 오류: {e}")
            import traceback
            traceback.print_exc()
            return []

# 즉시 테스트
async def test_fixed_scanner():
    print("🔥 수정된 OKX 스캐너 테스트!")
    
    scanner = FixedMarketScanner()
    
    try:
        await scanner.initialize()
        signals = await scanner.scan_top_coins(30)  # 상위 30개 코인 스캔
        
        print(f"\n📈 최종 결과: {len(signals)}개 신호")
        
    finally:
        await scanner.cleanup()

if __name__ == "__main__":
    asyncio.run(test_fixed_scanner())
