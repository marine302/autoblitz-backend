# 파일명: app/strategies/utils/scalping_technical_analysis.py
"""
단타로 전략을 위한 기술적 분석 모듈 확장
기존 technical_analysis.py에 추가할 내용
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime


class ScalpingTechnicalAnalysis:
    """단타로 전략 전용 기술적 분석"""
    
    def __init__(self):
        self.cache = {}  # 계산 결과 캐시
    
    def detect_price_spike(self, prices: np.ndarray, volume: np.ndarray, 
                          threshold: float = 0.02) -> Dict:
        """급등/급락 감지"""
        
        if len(prices) < 3:
            return {'spike_detected': False, 'direction': None, 'magnitude': 0}
        
        # 최근 2개 봉 비교
        current_price = prices[-1]
        prev_price = prices[-2]
        change_ratio = (current_price - prev_price) / prev_price
        
        # 거래량 급증 확인
        current_volume = volume[-1]
        avg_volume = np.mean(volume[-10:-1]) if len(volume) > 10 else volume[-2]
        volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
        
        spike_detected = abs(change_ratio) > threshold and volume_spike > 2.0
        
        return {
            'spike_detected': spike_detected,
            'direction': 'UP' if change_ratio > 0 else 'DOWN',
            'magnitude': abs(change_ratio),
            'volume_ratio': volume_spike,
            'strength': min(abs(change_ratio) * volume_spike * 10, 1.0)
        }
    
    def calculate_momentum_oscillator(self, prices: np.ndarray, period: int = 10) -> float:
        """모멘텀 오실레이터 (단기 추세 강도)"""
        
        if len(prices) < period + 1:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-(period + 1)]
        
        momentum = (current_price - past_price) / past_price
        
        # -1 ~ 1 사이로 정규화
        return np.tanh(momentum * 100)
    
    def detect_volume_breakout(self, volume: np.ndarray, lookback: int = 20) -> Dict:
        """거래량 돌파 감지"""
        
        if len(volume) < lookback + 1:
            return {'breakout': False, 'strength': 0}
        
        current_volume = volume[-1]
        avg_volume = np.mean(volume[-lookback:-1])
        std_volume = np.std(volume[-lookback:-1])
        
        # Z-Score 계산
        z_score = (current_volume - avg_volume) / std_volume if std_volume > 0 else 0
        
        breakout = z_score > 2.0  # 2 표준편차 이상
        
        return {
            'breakout': breakout,
            'z_score': z_score,
            'strength': min(z_score / 5.0, 1.0),  # 0-1 정규화
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
        }
    
    def calculate_price_velocity(self, prices: np.ndarray, period: int = 5) -> float:
        """가격 변화 속도 (1분당 변화율)"""
        
        if len(prices) < period + 1:
            return 0.0
        
        # 선형 회귀로 기울기 계산
        x = np.arange(period)
        y = prices[-period:]
        
        # 최소제곱법으로 기울기 계산
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # 가격 대비 비율로 정규화
        velocity = slope / np.mean(y) if np.mean(y) > 0 else 0
        
        return velocity
    
    def calculate_support_resistance_levels(self, high: np.ndarray, low: np.ndarray, 
                                          period: int = 20) -> Dict:
        """지지/저항선 계산 (단기)"""
        
        if len(high) < period or len(low) < period:
            return {'support': None, 'resistance': None, 'strength': 0}
        
        recent_high = high[-period:]
        recent_low = low[-period:]
        
        # 피벗 포인트 찾기
        resistance_levels = []
        support_levels = []
        
        # 고점 찾기 (저항선)
        for i in range(2, len(recent_high) - 2):
            if (recent_high[i] > recent_high[i-1] and recent_high[i] > recent_high[i-2] and
                recent_high[i] > recent_high[i+1] and recent_high[i] > recent_high[i+2]):
                resistance_levels.append(recent_high[i])
        
        # 저점 찾기 (지지선)
        for i in range(2, len(recent_low) - 2):
            if (recent_low[i] < recent_low[i-1] and recent_low[i] < recent_low[i-2] and
                recent_low[i] < recent_low[i+1] and recent_low[i] < recent_low[i+2]):
                support_levels.append(recent_low[i])
        
        # 가장 강한 지지/저항선 선택
        current_price = (high[-1] + low[-1]) / 2
        
        resistance = None
        if resistance_levels:
            # 현재 가격보다 위에 있는 저항선 중 가장 가까운 것
            upper_resistance = [r for r in resistance_levels if r > current_price]
            resistance = min(upper_resistance) if upper_resistance else max(resistance_levels)
        
        support = None
        if support_levels:
            # 현재 가격보다 아래에 있는 지지선 중 가장 가까운 것
            lower_support = [s for s in support_levels if s < current_price]
            support = max(lower_support) if lower_support else min(support_levels)
        
        # 강도 계산 (현재가와의 거리 기반)
        strength = 0
        if resistance and support:
            range_size = resistance - support
            strength = 1.0 - (abs(current_price - support) + abs(resistance - current_price)) / (2 * range_size)
        
        return {
            'support': support,
            'resistance': resistance,
            'strength': max(0, min(1, strength)),
            'current_price': current_price
        }
    
    def calculate_volatility_bands(self, prices: np.ndarray, period: int = 10, 
                                 multiplier: float = 1.5) -> Tuple[float, float, float]:
        """변동성 밴드 (단기 볼린저 밴드)"""
        
        if len(prices) < period:
            current_price = prices[-1] if len(prices) > 0 else 0
            return current_price, current_price, current_price
        
        recent_prices = prices[-period:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        upper_band = mean_price + (std_price * multiplier)
        lower_band = mean_price - (std_price * multiplier)
        
        return upper_band, mean_price, lower_band
    
    def detect_divergence(self, prices: np.ndarray, rsi: np.ndarray) -> Dict:
        """가격-RSI 다이버전스 감지"""
        
        if len(prices) < 10 or len(rsi) < 10:
            return {'divergence': False, 'type': None, 'strength': 0}
        
        # 최근 10개 봉에서 고점/저점 찾기
        price_peaks = []
        rsi_peaks = []
        
        for i in range(2, len(prices) - 2):
            # 가격 고점
            if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and
                prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                price_peaks.append((i, prices[i]))
                rsi_peaks.append((i, rsi[i]))
        
        if len(price_peaks) < 2:
            return {'divergence': False, 'type': None, 'strength': 0}
        
        # 최근 두 고점 비교
        recent_peaks = sorted(price_peaks, key=lambda x: x[0])[-2:]
        recent_rsi_peaks = sorted(rsi_peaks, key=lambda x: x[0])[-2:]
        
        if len(recent_peaks) == 2 and len(recent_rsi_peaks) == 2:
            # 가격 추세
            price_trend = recent_peaks[1][1] - recent_peaks[0][1]
            # RSI 추세  
            rsi_trend = recent_rsi_peaks[1][1] - recent_rsi_peaks[0][1]
            
            # 다이버전스 감지
            bearish_div = price_trend > 0 and rsi_trend < 0  # 가격 상승, RSI 하락
            bullish_div = price_trend < 0 and rsi_trend > 0  # 가격 하락, RSI 상승
            
            if bearish_div or bullish_div:
                strength = abs(price_trend) * abs(rsi_trend) / 100
                return {
                    'divergence': True,
                    'type': 'BEARISH' if bearish_div else 'BULLISH',
                    'strength': min(strength, 1.0),
                    'price_trend': price_trend,
                    'rsi_trend': rsi_trend
                }
        
        return {'divergence': False, 'type': None, 'strength': 0}
    
    def calculate_scalping_score(self, market_data: Dict) -> float:
        """단타로 적합성 점수 (0-1)"""
        
        scores = []
        
        # 변동성 점수 (적당한 변동성 선호)
        volatility = market_data.get('atr', 0)
        if 0.005 < volatility < 0.03:  # 0.5% ~ 3%
            vol_score = 1.0 - abs(volatility - 0.015) / 0.015
        else:
            vol_score = 0.2
        scores.append(vol_score)
        
        # 거래량 점수
        volume_ratio = market_data.get('volume_ratio', 1)
        vol_score = min(volume_ratio / 3.0, 1.0) if volume_ratio > 1.5 else 0.3
        scores.append(vol_score)
        
        # RSI 점수 (극값 선호)
        rsi = market_data.get('rsi', 50)
        if rsi < 35 or rsi > 65:
            rsi_score = 1.0 - abs(rsi - 50) / 50
        else:
            rsi_score = 0.4
        scores.append(rsi_score)
        
        # 가격 변화 점수
        price_change = abs(market_data.get('price_change', 0))
        change_score = min(price_change / 0.02, 1.0) if price_change > 0.005 else 0.2
        scores.append(change_score)
        
        # 전체 점수 (가중 평균)
        weights = [0.3, 0.25, 0.25, 0.2]  # 변동성 > 거래량 = RSI > 가격변화
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        return round(final_score, 3)


# 기존 TechnicalAnalysis 클래스에 추가할 메서드들
class TechnicalAnalysisExtended:
    """기존 TechnicalAnalysis 클래스 확장"""
    
    def __init__(self):
        self.scalping_ta = ScalpingTechnicalAnalysis()
    
    def calculate_stochastic_oscillator(self, high: np.ndarray, low: np.ndarray, 
                                      close: np.ndarray, k_period: int = 14, 
                                      d_period: int = 3) -> Tuple[float, float]:
        """스토캐스틱 오실레이터 계산"""
        
        if len(high) < k_period or len(low) < k_period or len(close) < k_period:
            return 50.0, 50.0
        
        # %K 계산
        recent_high = high[-k_period:]
        recent_low = low[-k_period:]
        current_close = close[-1]
        
        highest_high = np.max(recent_high)
        lowest_low = np.min(recent_low)
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D 계산 (단순화: %K의 이동평균)
        if len(close) >= k_period + d_period - 1:
            k_values = []
            for i in range(d_period):
                idx = -(i + 1)
                if len(high) >= k_period - i and len(low) >= k_period - i:
                    h = np.max(high[idx - k_period + 1:idx + 1])
                    l = np.min(low[idx - k_period + 1:idx + 1])
                    c = close[idx]
                    if h != l:
                        k_val = ((c - l) / (h - l)) * 100
                        k_values.append(k_val)
            
            d_percent = np.mean(k_values) if k_values else k_percent
        else:
            d_percent = k_percent
        
        return k_percent, d_percent
    
    def calculate_williams_r(self, high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, period: int = 14) -> float:
        """윌리엄스 %R 계산"""
        
        if len(high) < period or len(low) < period or len(close) < period:
            return -50.0
        
        recent_high = high[-period:]
        recent_low = low[-period:]
        current_close = close[-1]
        
        highest_high = np.max(recent_high)
        lowest_low = np.min(recent_low)
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        
        return williams_r
    
    def calculate_commodity_channel_index(self, high: np.ndarray, low: np.ndarray, 
                                        close: np.ndarray, period: int = 20) -> float:
        """CCI (Commodity Channel Index) 계산"""
        
        if len(high) < period or len(low) < period or len(close) < period:
            return 0.0
        
        # Typical Price 계산
        typical_prices = (high[-period:] + low[-period:] + close[-period:]) / 3
        
        # Simple Moving Average of Typical Price
        sma_tp = np.mean(typical_prices)
        
        # Mean Absolute Deviation
        mad = np.mean(np.abs(typical_prices - sma_tp))
        
        if mad == 0:
            return 0.0
        
        # CCI 계산
        current_tp = (high[-1] + low[-1] + close[-1]) / 3
        cci = (current_tp - sma_tp) / (0.015 * mad)
        
        return cci
    
    def calculate_money_flow_index(self, high: np.ndarray, low: np.ndarray, 
                                 close: np.ndarray, volume: np.ndarray, 
                                 period: int = 14) -> float:
        """MFI (Money Flow Index) 계산"""
        
        if len(high) < period + 1 or len(volume) < period + 1:
            return 50.0
        
        # Typical Price와 Money Flow 계산
        typical_prices = (high + low + close) / 3
        money_flows = typical_prices * volume
        
        positive_flow = 0
        negative_flow = 0
        
        # 최근 period 기간에 대해 계산
        for i in range(-period, 0):
            if i == -period:
                continue  # 첫 번째는 비교할 이전 값이 없음
            
            if typical_prices[i] > typical_prices[i-1]:
                positive_flow += money_flows[i]
            elif typical_prices[i] < typical_prices[i-1]:
                negative_flow += money_flows[i]
        
        if negative_flow == 0:
            return 100.0
        
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def detect_candlestick_patterns(self, open_prices: np.ndarray, high: np.ndarray, 
                                  low: np.ndarray, close: np.ndarray) -> Dict:
        """캔들스틱 패턴 감지"""
        
        if len(open_prices) < 3:
            return {'pattern': None, 'strength': 0, 'direction': None}
        
        o1, o2, o3 = open_prices[-3:]
        h1, h2, h3 = high[-3:]
        l1, l2, l3 = low[-3:]
        c1, c2, c3 = close[-3:]
        
        patterns = {}
        
        # 도지 패턴
        body_size = abs(c3 - o3)
        total_range = h3 - l3
        if total_range > 0 and body_size / total_range < 0.1:
            patterns['doji'] = {'strength': 0.7, 'direction': 'NEUTRAL'}
        
        # 해머 패턴
        upper_shadow = h3 - max(c3, o3)
        lower_shadow = min(c3, o3) - l3
        if (lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5 and
            total_range > 0):
            patterns['hammer'] = {'strength': 0.8, 'direction': 'BULLISH'}
        
        # 별똥별 패턴
        if (upper_shadow > body_size * 2 and lower_shadow < body_size * 0.5 and
            total_range > 0):
            patterns['shooting_star'] = {'strength': 0.8, 'direction': 'BEARISH'}
        
        # 강세 삼켜짐 패턴
        if (c2 < o2 and c3 > o3 and  # 2번째는 음봉, 3번째는 양봉
            o3 < c2 and c3 > o2):     # 3번째가 2번째를 완전히 감쌈
            patterns['bullish_engulfing'] = {'strength': 0.9, 'direction': 'BULLISH'}
        
        # 약세 삼켜짐 패턴
        if (c2 > o2 and c3 < o3 and  # 2번째는 양봉, 3번째는 음봉
            o3 > c2 and c3 < o2):     # 3번째가 2번째를 완전히 감쌈
            patterns['bearish_engulfing'] = {'strength': 0.9, 'direction': 'BEARISH'}
        
        # 가장 강한 패턴 반환
        if patterns:
            strongest = max(patterns.items(), key=lambda x: x[1]['strength'])
            return {
                'pattern': strongest[0],
                'strength': strongest[1]['strength'],
                'direction': strongest[1]['direction'],
                'all_patterns': patterns
            }
        
        return {'pattern': None, 'strength': 0, 'direction': None}


# 통합 분석 함수
def analyze_for_scalping(market_data: Dict) -> Dict:
    """단타로 전략을 위한 종합 분석"""
    
    # 필요한 데이터 추출
    ohlcv = market_data.get('ohlcv', [])
    if len(ohlcv) < 20:
        return {'error': '데이터 부족', 'scalping_score': 0}
    
    # 배열로 변환
    opens = np.array([float(d['open']) for d in ohlcv[-50:]])
    highs = np.array([float(d['high']) for d in ohlcv[-50:]])
    lows = np.array([float(d['low']) for d in ohlcv[-50:]])
    closes = np.array([float(d['close']) for d in ohlcv[-50:]])
    volumes = np.array([float(d['volume']) for d in ohlcv[-50:]])
    
    # 분석 도구 초기화
    scalping_ta = ScalpingTechnicalAnalysis()
    extended_ta = TechnicalAnalysisExtended()
    
    # 종합 분석 수행
    analysis = {
        # 급등/급락 감지
        'price_spike': scalping_ta.detect_price_spike(closes, volumes),
        
        # 거래량 돌파
        'volume_breakout': scalping_ta.detect_volume_breakout(volumes),
        
        # 모멘텀 분석
        'momentum': scalping_ta.calculate_momentum_oscillator(closes),
        'price_velocity': scalping_ta.calculate_price_velocity(closes),
        
        # 지지/저항선
        'support_resistance': scalping_ta.calculate_support_resistance_levels(highs, lows),
        
        # 변동성 밴드
        'volatility_bands': scalping_ta.calculate_volatility_bands(closes),
        
        # 다이버전스 (RSI 필요시 계산)
        'rsi': market_data.get('rsi', 50),
        
        # 확장 지표들
        'stochastic': extended_ta.calculate_stochastic_oscillator(highs, lows, closes),
        'williams_r': extended_ta.calculate_williams_r(highs, lows, closes),
        'cci': extended_ta.calculate_commodity_channel_index(highs, lows, closes),
        'mfi': extended_ta.calculate_money_flow_index(highs, lows, closes, volumes),
        
        # 캔들스틱 패턴
        'candlestick': extended_ta.detect_candlestick_patterns(opens, highs, lows, closes),
        
        # 종합 점수
        'scalping_score': scalping_ta.calculate_scalping_score({
            'atr': market_data.get('atr', 0.01),
            'volume_ratio': volumes[-1] / np.mean(volumes[:-1]) if len(volumes) > 1 else 1,
            'rsi': market_data.get('rsi', 50),
            'price_change': (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
        })
    }
    
    return analysis


# 단타로 시그널 강도 계산
def calculate_scalping_signal_strength(analysis: Dict) -> Dict:
    """단타로 시그널 강도 계산"""
    
    bullish_signals = []
    bearish_signals = []
    
    # 가격 급등/급락
    spike = analysis.get('price_spike', {})
    if spike.get('spike_detected'):
        strength = spike.get('strength', 0)
        if spike.get('direction') == 'UP':
            bullish_signals.append(('price_spike', strength))
        else:
            bearish_signals.append(('price_spike', strength))
    
    # 거래량 돌파
    vol_breakout = analysis.get('volume_breakout', {})
    if vol_breakout.get('breakout'):
        strength = vol_breakout.get('strength', 0)
        # 거래량 돌파는 방향성 확인 필요
        momentum = analysis.get('momentum', 0)
        if momentum > 0:
            bullish_signals.append(('volume_breakout', strength))
        else:
            bearish_signals.append(('volume_breakout', strength))
    
    # 기술적 지표들
    rsi = analysis.get('rsi', 50)
    if rsi < 30:
        bullish_signals.append(('rsi_oversold', (30 - rsi) / 30))
    elif rsi > 70:
        bearish_signals.append(('rsi_overbought', (rsi - 70) / 30))
    
    # 스토캐스틱
    stoch = analysis.get('stochastic', (50, 50))
    if stoch[0] < 20 and stoch[1] < 20:
        bullish_signals.append(('stochastic_oversold', 0.7))
    elif stoch[0] > 80 and stoch[1] > 80:
        bearish_signals.append(('stochastic_overbought', 0.7))
    
    # 캔들스틱 패턴
    candle = analysis.get('candlestick', {})
    if candle.get('pattern'):
        strength = candle.get('strength', 0)
        direction = candle.get('direction')
        if direction == 'BULLISH':
            bullish_signals.append(('candlestick', strength))
        elif direction == 'BEARISH':
            bearish_signals.append(('candlestick', strength))
    
    # 최종 강도 계산
    bullish_strength = sum(s[1] for s in bullish_signals) / len(bullish_signals) if bullish_signals else 0
    bearish_strength = sum(s[1] for s in bearish_signals) / len(bearish_signals) if bearish_signals else 0
    
    # 신호 결정
    if bullish_strength > bearish_strength and bullish_strength > 0.6:
        signal = 'BUY'
        confidence = bullish_strength
    elif bearish_strength > bullish_strength and bearish_strength > 0.6:
        signal = 'SELL'  
        confidence = bearish_strength
    else:
        signal = 'HOLD'
        confidence = 0.0
    
    return {
        'signal': signal,
        'confidence': confidence,
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,
        'bullish_strength': bullish_strength,
        'bearish_strength': bearish_strength,
        'scalping_score': analysis.get('scalping_score', 0)
    }