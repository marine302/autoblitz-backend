"""
OKX API Client - 실시간 데이터 수집기
Created: 2025.06.24 15:40 KST
Purpose: 본사서버에서 실제 거래소 데이터 수집
"""
import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
import hmac
import base64
import time

logger = logging.getLogger(__name__)

class OKXClient:
    """OKX 거래소 API 클라이언트 (Public API 사용)"""
    
    def __init__(self):
        self.base_url = "https://www.okx.com"
        self.session = None
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def get_top_coins(self, limit: int = 100) -> List[str]:
        """거래량 상위 코인 목록 조회"""
        try:
            url = f"{self.base_url}/api/v5/public/instruments"
            params = {
                'instType': 'SPOT',
                'state': 'live'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    instruments = data.get('data', [])
                    
                    # USDT 페어만 필터링하고 상위 100개 선택
                    usdt_pairs = [
                        inst['instId'] for inst in instruments 
                        if inst['instId'].endswith('-USDT')
                    ]
                    
                    # 주요 코인 우선순위 적용
                    priority_coins = [
                        'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'XRP-USDT',
                        'ADA-USDT', 'SOL-USDT', 'DOT-USDT', 'LINK-USDT'
                    ]
                    
                    # 우선순위 코인 + 나머지
                    result = priority_coins + [coin for coin in usdt_pairs 
                                             if coin not in priority_coins]
                    
                    return result[:limit]
                else:
                    logger.error(f"Failed to get instruments: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting top coins: {e}")
            return []
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """특정 코인의 티커 정보 조회"""
        try:
            url = f"{self.base_url}/api/v5/public/tickers"
            params = {'instId': symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    tickers = data.get('data', [])
                    
                    if tickers:
                        ticker = tickers[0]
                        return {
                            'symbol': symbol,
                            'price': float(ticker['last']),
                            'volume_24h': float(ticker['vol24h']),
                            'change_24h': float(ticker['chg']),
                            'high_24h': float(ticker['high24h']),
                            'low_24h': float(ticker['low24h']),
                            'timestamp': datetime.now()
                        }
                else:
                    logger.warning(f"Failed to get ticker for {symbol}: {response.status}")
                    
        except Exception as e:
            logger.warning(f"Error getting ticker for {symbol}: {e}")
        
        return None
    
    async def get_multiple_tickers(self, symbols: List[str]) -> List[Dict]:
        """여러 코인의 티커 정보 배치 조회"""
        try:
            # OKX는 최대 20개씩 배치 요청 가능
            batch_size = 20
            all_tickers = []
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                
                url = f"{self.base_url}/api/v5/public/tickers"
                params = {'instType': 'SPOT'}
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        tickers = data.get('data', [])
                        
                        # 요청한 심볼들만 필터링
                        for ticker in tickers:
                            if ticker['instId'] in batch_symbols:
                                all_tickers.append({
                                    'symbol': ticker['instId'],
                                    'price': float(ticker['last']),
                                    'volume_24h': float(ticker['vol24h']),
                                    'change_24h': float(ticker['chg']),
                                    'high_24h': float(ticker['high24h']),
                                    'low_24h': float(ticker['low24h']),
                                    'timestamp': datetime.now()
                                })
                    
                    # API 호출 제한 준수 (100ms 대기)
                    await asyncio.sleep(0.1)
            
            return all_tickers
            
        except Exception as e:
            logger.error(f"Error getting multiple tickers: {e}")
            return []
    
    async def get_klines(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[Dict]:
        """K라인(캔들) 데이터 조회"""
        try:
            url = f"{self.base_url}/api/v5/public/candles"
            params = {
                'instId': symbol,
                'bar': timeframe,
                'limit': str(limit)
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    candles = data.get('data', [])
                    
                    klines = []
                    for candle in candles:
                        klines.append({
                            'timestamp': int(candle[0]),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5])
                        })
                    
                    return klines
                    
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
        
        return []

class RealTimeMarketScanner:
    """실시간 시장 스캐너 - OKX 연동"""
    
    def __init__(self):
        self.okx_client = None
        self.symbols = []
        self.scan_count = 0
        self.signal_history = []
        
    async def initialize(self):
        """초기화 - 코인 목록 로드"""
        self.okx_client = OKXClient()
        await self.okx_client.__aenter__()
        
        # 상위 50개 코인 로드 (API 제한 고려)
        self.symbols = await self.okx_client.get_top_coins(50)
        logger.info(f"✅ {len(self.symbols)}개 코인 로드 완료")
        
        # 로드된 코인 목록 출력
        print(f"\n📊 로드된 코인 목록 (상위 {len(self.symbols)}개):")
        for i, symbol in enumerate(self.symbols[:10], 1):
            print(f"  {i:2d}. {symbol}")
        if len(self.symbols) > 10:
            print(f"  ... 및 {len(self.symbols) - 10}개 추가")
    
    async def cleanup(self):
        """정리 작업"""
        if self.okx_client:
            await self.okx_client.__aexit__(None, None, None)
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI 계산 (간단한 버전)"""
        if len(prices) < period + 1:
            return 50.0  # 기본값
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def check_signal_conditions(self, ticker: Dict, klines: List[Dict]) -> Optional[Dict]:
        """신호 조건 체크 (실제 데이터 기반)"""
        try:
            # 가격 리스트 추출
            if len(klines) < 20:
                return None
                
            prices = [k['close'] for k in klines[-20:]]
            volumes = [k['volume'] for k in klines[-20:]]
            
            # RSI 계산
            rsi = self.calculate_rsi(prices)
            
            # 볼륨 비율 계산
            current_volume = volumes[-1] if volumes else 0
            avg_volume = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 가격 변화율
            price_change = ticker['change_24h']
            
            # 조건 체크 (완화된 조건)
            conditions = {
                'rsi_oversold': rsi < 35,
                'volume_surge': volume_ratio > 1.3,
                'price_drop': price_change < -2.0,
            }
            
            # 2개 이상 조건 충족시 신호 생성
            met_conditions = sum(conditions.values())
            
            if met_conditions >= 2:
                return {
                    'symbol': ticker['symbol'],
                    'signal_type': 'BUY',
                    'confidence': met_conditions / len(conditions),
                    'conditions_met': conditions,
                    'timestamp': ticker['timestamp'],
                    'market_data': {
                        'price': ticker['price'],
                        'rsi': rsi,
                        'volume_ratio': volume_ratio,
                        'price_change': price_change
                    }
                }
        
        except Exception as e:
            logger.warning(f"Error checking signal for {ticker.get('symbol', 'unknown')}: {e}")
        
        return None
    
    async def scan_market(self) -> List[Dict]:
        """실시간 시장 스캔"""
        start_time = datetime.now()
        signals = []
        
        try:
            logger.info(f"🔍 {len(self.symbols)}개 코인 실시간 스캔 시작...")
            
            # 모든 코인의 티커 정보 가져오기
            tickers = await self.okx_client.get_multiple_tickers(self.symbols)
            
            logger.info(f"📊 {len(tickers)}개 티커 데이터 수집 완료")
            
            # 각 티커에 대해 신호 체크
            for ticker in tickers:
                # K라인 데이터 가져오기 (성능을 위해 제한적으로)
                klines = await self.okx_client.get_klines(ticker['symbol'], '1m', 20)
                
                # 신호 체크
                signal = self.check_signal_conditions(ticker, klines)
                if signal:
                    signals.append(signal)
                    logger.info(f"✅ 신호 발견: {ticker['symbol']} (신뢰도: {signal['confidence']:.2f})")
                
                # API 호출 제한 준수
                await asyncio.sleep(0.05)  # 50ms 대기
        
        except Exception as e:
            logger.error(f"❌ 시장 스캔 오류: {e}")
        
        scan_duration = (datetime.now() - start_time).total_seconds()
        self.scan_count += 1
        
        logger.info(f"📊 스캔 #{self.scan_count} 완료: {len(signals)}개 신호 발견 "
                   f"(소요시간: {scan_duration:.2f}초)")
        
        return signals
    
    def log_signals(self, signals: List[Dict]) -> None:
        """신호 로깅"""
        if not signals:
            print(f"\n📊 실시간 스캔 완료 - {datetime.now().strftime('%H:%M:%S')}")
            print("❌ 조건 충족 신호 없음")
            return
        
        self.signal_history.extend(signals)
        
        print(f"\n🎯 실시간 신호 감지 - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        
        for signal in signals:
            conditions = signal['conditions_met']
            met = [k for k, v in conditions.items() if v]
            market = signal['market_data']
            
            print(f"📈 {signal['symbol']:12} | "
                  f"신뢰도: {signal['confidence']:.2f} | "
                  f"가격: ${market['price']:8.2f} | "
                  f"RSI: {market['rsi']:5.1f} | "
                  f"조건: {', '.join(met)}")
        
        print(f"총 {len(signals)}개 신호 | 누적: {len(self.signal_history)}개")
        print("=" * 80)

# 즉시 테스트 함수
async def test_okx_realtime():
    """OKX 실시간 테스트 실행"""
    print("🔥 OKX 실시간 테스트 시작!")
    
    scanner = RealTimeMarketScanner()
    
    try:
        # 초기화
        await scanner.initialize()
        
        # 실시간 스캔 실행
        signals = await scanner.scan_market()
        scanner.log_signals(signals)
        
        # 결과 요약
        print(f"\n📈 테스트 결과:")
        print(f"- 스캔 코인 수: {len(scanner.symbols)}개")
        print(f"- 발견 신호: {len(signals)}개")
        if signals:
            avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
            print(f"- 평균 신뢰도: {avg_confidence:.2f}")
        
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
    
    finally:
        await scanner.cleanup()

if __name__ == "__main__":
    asyncio.run(test_okx_realtime())