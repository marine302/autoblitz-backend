"""
Market Scanner - 100개 코인 자동 수집 및 실시간 모니터링
Created: 2025.06.24 14:58 KST
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
import json
import logging
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """시장 데이터 구조"""
    symbol: str
    price: float
    volume_24h: float
    change_24h: float
    timestamp: datetime
    rsi: Optional[float] = None
    volume_ratio: Optional[float] = None

class MarketScanner:
    """대량 코인 스캐닝 시스템"""
    
    def __init__(self):
        self.top_coins = self._get_top_100_coins()
        self.scan_count = 0
        self.signal_history = []
        
    def _get_top_100_coins(self) -> List[str]:
        """거래량 상위 100개 코인 리스트"""
        # 실제로는 거래소 API에서 가져올 것이지만, 지금은 고정 리스트 사용
        major_coins = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
            "DOT/USDT", "LINK/USDT", "LTC/USDT", "BCH/USDT", "UNI/USDT",
            "SOL/USDT", "DOGE/USDT", "MATIC/USDT", "AVAX/USDT", "ATOM/USDT",
            "VET/USDT", "FIL/USDT", "TRX/USDT", "ETC/USDT", "XLM/USDT"
        ]
        
        altcoins = [
            "ALGO/USDT", "SAND/USDT", "MANA/USDT", "CRV/USDT", "COMP/USDT",
            "SUSHI/USDT", "YFI/USDT", "SNX/USDT", "MKR/USDT", "AAVE/USDT",
            "BAT/USDT", "ZRX/USDT", "KNC/USDT", "LRC/USDT", "ENJ/USDT",
            "CHZ/USDT", "HOT/USDT", "ZIL/USDT", "ICX/USDT", "ONT/USDT",
            "QTUM/USDT", "ZEC/USDT", "DASH/USDT", "NEO/USDT", "WAVES/USDT",
            "LSK/USDT", "ARK/USDT", "STEEM/USDT", "XEM/USDT", "NANO/USDT"
        ]
        
        defi_coins = [
            "1INCH/USDT", "ALPHA/USDT", "BADGER/USDT", "CAKE/USDT", "CREAM/USDT",
            "DPI/USDT", "FTM/USDT", "HEGIC/USDT", "KAVA/USDT", "PERP/USDT",
            "RUNE/USDT", "SFINX/USDT", "TORN/USDT", "UMA/USDT", "WING/USDT",
            "YFII/USDT", "BNT/USDT", "CVP/USDT", "DODO/USDT", "FOR/USDT"
        ]
        
        nft_gaming = [
            "AXS/USDT", "SLP/USDT", "GALA/USDT", "FLOW/USDT", "THETA/USDT",
            "TFUEL/USDT", "CHR/USDT", "ALICE/USDT", "TLM/USDT", "ILV/USDT",
            "SUPER/USDT", "PYR/USDT", "SKILL/USDT", "TOWN/USDT", "HERO/USDT",
            "DDIM/USDT", "RACA/USDT", "WILD/USDT", "NFTB/USDT", "UFO/USDT"
        ]
        
        layer2_others = [
            "ROSE/USDT", "CELO/USDT", "NEAR/USDT", "ONE/USDT", "HBAR/USDT",
            "EGLD/USDT", "LUNA/USDT", "KSM/USDT", "AR/USDT", "STORJ/USDT"
        ]
        
        all_coins = major_coins + altcoins + defi_coins + nft_gaming + layer2_others
        return all_coins[:100]  # 상위 100개만
    
    def generate_sample_market_data(self, symbol: str) -> MarketData:
        """샘플 마켓 데이터 생성 (실제로는 거래소 API 호출)"""
        import random
        
        # 실제로는 거래소 API에서 가져올 데이터
        base_price = random.uniform(0.1, 50000)
        
        return MarketData(
            symbol=symbol,
            price=base_price,
            volume_24h=random.uniform(1000000, 100000000),
            change_24h=random.uniform(-15, 15),
            timestamp=datetime.now(),
            rsi=random.uniform(20, 80),
            volume_ratio=random.uniform(0.5, 3.0)
        )
    
    def check_signal_conditions(self, market_data: MarketData) -> Optional[Dict]:
        """신호 조건 체크 (기본전략 기반)"""
        
        # 완화된 조건으로 즉시 테스트 가능
        conditions = {
            'rsi_oversold': market_data.rsi < 35,  # 완화: 30 → 35
            'volume_surge': market_data.volume_ratio > 1.3,  # 완화: 1.5 → 1.3
            'price_drop': market_data.change_24h < -2,  # 2% 이상 하락
        }
        
        # 2개 이상 조건 충족시 신호 생성
        met_conditions = sum(conditions.values())
        
        if met_conditions >= 2:
            return {
                'symbol': market_data.symbol,
                'signal_type': 'BUY',
                'confidence': met_conditions / len(conditions),
                'conditions_met': conditions,
                'timestamp': market_data.timestamp,
                'market_data': market_data
            }
        
        return None
    
    def scan_single_market(self, symbol: str) -> Optional[Dict]:
        """단일 마켓 스캔"""
        try:
            market_data = self.generate_sample_market_data(symbol)
            signal = self.check_signal_conditions(market_data)
            
            if signal:
                logger.info(f"✅ 신호 발견: {symbol} (신뢰도: {signal['confidence']:.2f})")
                
            return signal
            
        except Exception as e:
            logger.error(f"❌ {symbol} 스캔 오류: {e}")
            return None
    
    def scan_all_markets(self) -> List[Dict]:
        """전체 마켓 스캔"""
        signals = []
        start_time = datetime.now()
        
        logger.info(f"🔍 {len(self.top_coins)}개 코인 스캔 시작...")
        
        for symbol in self.top_coins:
            signal = self.scan_single_market(symbol)
            if signal:
                signals.append(signal)
        
        scan_duration = (datetime.now() - start_time).total_seconds()
        self.scan_count += 1
        
        logger.info(f"📊 스캔 #{self.scan_count} 완료: {len(signals)}개 신호 발견 "
                   f"(소요시간: {scan_duration:.2f}초)")
        
        return signals
    
    def log_signals(self, signals: List[Dict]) -> None:
        """신호 로깅 및 저장"""
        if not signals:
            return
            
        # 신호 히스토리에 추가
        self.signal_history.extend(signals)
        
        # 콘솔 출력
        print(f"\n🎯 실시간 신호 감지 - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        for signal in signals:
            conditions = signal['conditions_met']
            met = [k for k, v in conditions.items() if v]
            
            print(f"📈 {signal['symbol']:12} | "
                  f"신뢰도: {signal['confidence']:.2f} | "
                  f"조건: {', '.join(met)}")
        
        print(f"총 {len(signals)}개 신호 | 누적: {len(self.signal_history)}개")
        print("=" * 60)
    
    def get_signal_statistics(self) -> Dict:
        """신호 통계 정보"""
        if not self.signal_history:
            return {"total_signals": 0}
        
        total = len(self.signal_history)
        high_confidence = len([s for s in self.signal_history if s['confidence'] >= 0.8])
        medium_confidence = len([s for s in self.signal_history if 0.5 <= s['confidence'] < 0.8])
        
        return {
            "total_signals": total,
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "avg_confidence": sum(s['confidence'] for s in self.signal_history) / total,
            "scan_count": self.scan_count,
            "signals_per_scan": total / max(self.scan_count, 1)
        }

class RealTimeScanner:
    """실시간 연속 스캐닝"""
    
    def __init__(self, interval_seconds: int = 60):
        self.scanner = MarketScanner()
        self.interval = interval_seconds
        self.running = False
        
    async def start_continuous_scan(self):
        """연속 스캐닝 시작"""
        self.running = True
        logger.info(f"🚀 실시간 스캐닝 시작 (간격: {self.interval}초)")
        
        while self.running:
            signals = self.scanner.scan_all_markets()
            self.scanner.log_signals(signals)
            
            # 통계 출력 (매 5회마다)
            if self.scanner.scan_count % 5 == 0:
                stats = self.scanner.get_signal_statistics()
                print(f"\n📊 누적 통계: 총 {stats['total_signals']}개 신호, "
                      f"평균 신뢰도: {stats['avg_confidence']:.2f}")
            
            await asyncio.sleep(self.interval)
    
    def stop(self):
        """스캐닝 중지"""
        self.running = False
        logger.info("⏸️ 실시간 스캐닝 중지")

# 즉시 테스트 함수
def quick_test():
    """즉시 테스트 실행"""
    print("🔥 즉시 테스트 시작!")
    scanner = MarketScanner()
    
    # 한 번 스캔 실행
    signals = scanner.scan_all_markets()
    scanner.log_signals(signals)
    
    # 통계 출력
    stats = scanner.get_signal_statistics()
    print(f"\n📈 테스트 결과:")
    print(f"- 스캔 코인 수: {len(scanner.top_coins)}개")
    print(f"- 발견 신호: {stats['total_signals']}개")
    if stats['total_signals'] > 0:
        print(f"- 평균 신뢰도: {stats['avg_confidence']:.2f}")
        print(f"- 고신뢰도 신호: {stats['high_confidence']}개")
    
    return scanner

if __name__ == "__main__":
    # 즉시 테스트 실행
    quick_test()