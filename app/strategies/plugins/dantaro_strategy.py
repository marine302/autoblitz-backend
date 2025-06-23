# 📁 app/strategies/plugins/dantaro_strategy.py
"""
작업명: Step_9-3 단타로 전략 플러그인 구현
진행상황: 2/4 완료 (기본전략 API화 완료, 단타로전략 구현 중)
이전작업: 기본전략 API 구현 완료
다음작업: API 연동 및 통합 테스트
생성시간: 2025.06.24 01:16
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class DantaroConfig:
    """단타로 전략 설정"""
    base_amount: float = 5500.0        # 기본 투자 금액
    interval_percent: float = 2.0      # 분할매수 간격 (%)
    multiplier: float = 2.0            # 물량 증가 배수
    max_stages: int = 7                # 최대 단계
    profit_target: float = 0.5         # 목표 수익률 (%)
    max_wait_time: int = 4320          # 최대 대기시간 (분, 3일)

@dataclass 
class DantaroPosition:
    """단타로 포지션 정보"""
    symbol: str
    current_stage: int = 0             # 현재 진행 단계
    total_quantity: float = 0.0        # 총 보유 수량
    total_cost: float = 0.0            # 총 투자 금액
    avg_price: float = 0.0             # 평균 매수가
    target_price: float = 0.0          # 목표 매도가
    stages: List[Dict] = None          # 각 단계별 정보
    created_at: datetime = None
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = []
        if self.created_at is None:
            self.created_at = datetime.now()

class DantaroStrategy:
    """
    단타로 (물타기) 전략 구현
    
    핵심 로직:
    1. 첫 매수 후 가격 하락시 2배씩 물타기
    2. 평균가 + 0.5% 도달시 전량 매도
    3. 최대 7단계까지 진행
    """
    
    def __init__(self, config: DantaroConfig = None):
        self.config = config or DantaroConfig()
        self.positions: Dict[str, DantaroPosition] = {}
        self.logger = logging.getLogger(f"strategy.dantaro")
        
        self.logger.info(f"단타로 전략 초기화: 기본금액 {self.config.base_amount}원, "
                        f"간격 {self.config.interval_percent}%, "
                        f"최대 {self.config.max_stages}단계")
    
    def calculate_stage_amounts(self) -> List[float]:
        """각 단계별 투자 금액 계산"""
        amounts = []
        for stage in range(1, self.config.max_stages + 1):
            amount = self.config.base_amount * (self.config.multiplier ** (stage - 1))
            amounts.append(amount)
        
        return amounts
    
    def get_total_required_amount(self) -> float:
        """전체 단계 완료에 필요한 총 금액"""
        stage_amounts = self.calculate_stage_amounts()
        return sum(stage_amounts)
    
    def analyze_entry_signal(self, market_data: Dict) -> Optional[Dict]:
        """
        진입 신호 분석
        
        단타로 진입 조건:
        1. 기존 포지션이 없을 때
        2. 적당한 변동성 (1-5% 하락)
        3. 충분한 거래량
        """
        symbol = market_data['symbol']
        
        # 이미 포지션이 있으면 진입하지 않음
        if symbol in self.positions:
            return None
        
        price = market_data['price']
        change_24h = market_data['change_24h']
        volume_24h = market_data['volume_24h']
        
        # 진입 조건 체크
        conditions = {
            'price_drop': -5.0 <= change_24h <= -1.0,  # 1-5% 하락
            'sufficient_volume': volume_24h > 1000000,   # 충분한 거래량
            'reasonable_price': price > 0.001           # 합리적 가격
        }
        
        met_conditions = sum(conditions.values())
        
        if met_conditions >= 2:  # 2개 이상 조건 충족
            return {
                'action': 'ENTRY',
                'symbol': symbol,
                'stage': 1,
                'price': price,
                'amount': self.config.base_amount,
                'confidence': met_conditions / len(conditions),
                'reason': f"단타로 진입: {', '.join([k for k, v in conditions.items() if v])}",
                'conditions_met': [k for k, v in conditions.items() if v]
            }
        
        return None
    
    def analyze_add_signal(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        추가 매수 신호 분석 (물타기)
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # 최대 단계 도달 체크
        if position.current_stage >= self.config.max_stages:
            return None
        
        # 다음 단계 진입 가격 계산
        last_stage_price = position.stages[-1]['price'] if position.stages else position.avg_price
        next_stage_price = last_stage_price * (1 - self.config.interval_percent / 100)
        
        # 가격이 다음 단계 조건에 도달했는지 확인
        if current_price <= next_stage_price:
            next_stage = position.current_stage + 1
            next_amount = self.config.base_amount * (self.config.multiplier ** (next_stage - 1))
            
            return {
                'action': 'ADD',
                'symbol': symbol,
                'stage': next_stage,
                'price': current_price,
                'amount': next_amount,
                'confidence': 0.8,
                'reason': f"단타로 {next_stage}단계 추가매수",
                'trigger_price': next_stage_price
            }
        
        return None
    
    def analyze_exit_signal(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        매도 신호 분석
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # 목표 가격 도달 체크
        if current_price >= position.target_price:
            profit_amount = (current_price - position.avg_price) * position.total_quantity
            profit_rate = (current_price - position.avg_price) / position.avg_price * 100
            
            return {
                'action': 'EXIT',
                'symbol': symbol,
                'price': current_price,
                'quantity': position.total_quantity,
                'profit_amount': profit_amount,
                'profit_rate': profit_rate,
                'confidence': 1.0,
                'reason': f"목표가 도달: {profit_rate:.2f}% 수익"
            }
        
        # 시간 초과 체크 (3일 경과)
        time_elapsed = (datetime.now() - position.created_at).total_seconds() / 60
        if time_elapsed > self.config.max_wait_time:
            return {
                'action': 'EXIT',
                'symbol': symbol,
                'price': current_price,
                'quantity': position.total_quantity,
                'profit_amount': (current_price - position.avg_price) * position.total_quantity,
                'confidence': 0.5,
                'reason': f"시간 초과 매도 ({time_elapsed/1440:.1f}일 경과)"
            }
        
        return None
    
    def execute_stage(self, signal: Dict) -> bool:
        """
        단계별 매수 실행 (시뮬레이션)
        """
        try:
            symbol = signal['symbol']
            action = signal['action']
            price = signal['price']
            amount = signal['amount']
            
            if action == 'ENTRY':
                # 신규 포지션 생성
                quantity = amount / price
                position = DantaroPosition(
                    symbol=symbol,
                    current_stage=1,
                    total_quantity=quantity,
                    total_cost=amount,
                    avg_price=price
                )
                
                # 목표가 설정 (평균가 + 0.5%)
                position.target_price = price * (1 + self.config.profit_target / 100)
                
                # 첫 단계 정보 추가
                position.stages.append({
                    'stage': 1,
                    'price': price,
                    'amount': amount,
                    'quantity': quantity,
                    'timestamp': datetime.now()
                })
                
                self.positions[symbol] = position
                self.logger.info(f"단타로 진입: {symbol} 1단계, {price:.4f}원, {amount:.0f}원")
                
            elif action == 'ADD':
                # 기존 포지션에 추가
                position = self.positions[symbol]
                stage = signal['stage']
                quantity = amount / price
                
                # 포지션 업데이트
                position.current_stage = stage
                position.total_quantity += quantity
                position.total_cost += amount
                position.avg_price = position.total_cost / position.total_quantity
                
                # 목표가 재설정
                position.target_price = position.avg_price * (1 + self.config.profit_target / 100)
                
                # 단계 정보 추가
                position.stages.append({
                    'stage': stage,
                    'price': price,
                    'amount': amount,
                    'quantity': quantity,
                    'timestamp': datetime.now()
                })
                
                self.logger.info(f"단타로 추가: {symbol} {stage}단계, "
                               f"평균가: {position.avg_price:.4f}원, "
                               f"목표가: {position.target_price:.4f}원")
            
            return True
            
        except Exception as e:
            self.logger.error(f"단계 실행 오류: {e}")
            return False
    
    def execute_exit(self, signal: Dict) -> Dict:
        """
        매도 실행 (시뮬레이션)
        """
        try:
            symbol = signal['symbol']
            
            if symbol not in self.positions:
                return {'success': False, 'error': 'Position not found'}
            
            position = self.positions[symbol]
            exit_price = signal['price']
            
            # 수익 계산
            profit_amount = (exit_price - position.avg_price) * position.total_quantity
            profit_rate = (exit_price - position.avg_price) / position.avg_price * 100
            
            result = {
                'success': True,
                'symbol': symbol,
                'stages_completed': position.current_stage,
                'total_cost': position.total_cost,
                'exit_value': exit_price * position.total_quantity,
                'profit_amount': profit_amount,
                'profit_rate': profit_rate,
                'duration_minutes': (datetime.now() - position.created_at).total_seconds() / 60,
                'reason': signal['reason']
            }
            
            # 포지션 삭제
            del self.positions[symbol]
            
            self.logger.info(f"단타로 완료: {symbol}, "
                           f"수익률: {profit_rate:.2f}%, "
                           f"수익금: {profit_amount:.0f}원, "
                           f"{position.current_stage}단계 완료")
            
            return result
            
        except Exception as e:
            self.logger.error(f"매도 실행 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_signal(self, market_data: Dict) -> Optional[Dict]:
        """
        시장 데이터를 기반으로 신호 생성
        """
        symbol = market_data['symbol']
        current_price = market_data['price']
        
        # 1. 매도 신호 우선 체크
        exit_signal = self.analyze_exit_signal(symbol, current_price)
        if exit_signal:
            return exit_signal
        
        # 2. 추가 매수 신호 체크
        add_signal = self.analyze_add_signal(symbol, current_price)
        if add_signal:
            return add_signal
        
        # 3. 신규 진입 신호 체크
        entry_signal = self.analyze_entry_signal(market_data)
        if entry_signal:
            return entry_signal
        
        return None
    
    def get_position_status(self, symbol: str) -> Optional[Dict]:
        """포지션 상태 조회"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        return {
            'symbol': symbol,
            'current_stage': position.current_stage,
            'avg_price': position.avg_price,
            'target_price': position.target_price,
            'total_quantity': position.total_quantity,
            'total_cost': position.total_cost,
            'stages_info': position.stages,
            'duration_minutes': (datetime.now() - position.created_at).total_seconds() / 60
        }
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """모든 포지션 상태 조회"""
        return {symbol: self.get_position_status(symbol) 
                for symbol in self.positions.keys()}
    
    def get_strategy_stats(self) -> Dict:
        """전략 통계 정보"""
        total_required = self.get_total_required_amount()
        stage_amounts = self.calculate_stage_amounts()
        
        return {
            'config': {
                'base_amount': self.config.base_amount,
                'interval_percent': self.config.interval_percent,
                'multiplier': self.config.multiplier,
                'max_stages': self.config.max_stages,
                'profit_target': self.config.profit_target
            },
            'requirements': {
                'total_required_amount': total_required,
                'stage_amounts': stage_amounts
            },
            'current_positions': len(self.positions),
            'active_symbols': list(self.positions.keys())
        }

def create_dantaro_strategy(config: DantaroConfig = None) -> DantaroStrategy:
    """단타로 전략 인스턴스 생성"""
    return DantaroStrategy(config)

# 테스트 함수
def test_dantaro_strategy():
    """단타로 전략 테스트"""
    print("🔥 단타로 전략 테스트 시작!")
    
    # 전략 생성
    strategy = create_dantaro_strategy()
    stats = strategy.get_strategy_stats()
    
    print(f"📊 전략 설정:")
    print(f"  - 기본 금액: {stats['config']['base_amount']:,.0f}원")
    print(f"  - 총 필요 금액: {stats['requirements']['total_required_amount']:,.0f}원")
    print(f"  - 단계별 금액: {[f'{amt:,.0f}' for amt in stats['requirements']['stage_amounts'][:3]]}")
    
    # 샘플 시장 데이터로 테스트
    sample_data = {
        'symbol': 'BTC-USDT',
        'price': 50000.0,
        'change_24h': -3.5,  # 3.5% 하락
        'volume_24h': 50000000
    }
    
    # 진입 신호 테스트
    signal = strategy.generate_signal(sample_data)
    if signal:
        print(f"✅ 진입 신호 생성: {signal['reason']}")
        strategy.execute_stage(signal)
        
        # 포지션 확인
        position = strategy.get_position_status('BTC-USDT')
        print(f"📈 포지션 생성: 평균가 {position['avg_price']:,.0f}원, 목표가 {position['target_price']:,.0f}원")
    else:
        print("❌ 진입 신호 없음")

if __name__ == "__main__":
    test_dantaro_strategy()