# 파일: app/strategies/core/strategy_manager.py
# 경로: /workspaces/autoblitz-backend/app/strategies/core/strategy_manager.py

"""
오토블리츠 전략 플러그인 매니저

전략의 동적 로딩, 관리, 실행을 담당하는 핵심 모듈입니다.
플러그인 아키텍처를 통해 런타임에 전략을 로드하고 관리합니다.
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Optional, Type, Any
from pathlib import Path
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .strategy_base import StrategyBase, StrategyConfig, StrategyStatus, MarketData, TradingSignal

logger = logging.getLogger(__name__)


class StrategyLoadError(Exception):
    """전략 로드 관련 예외"""
    pass


class StrategyManager:
    """
    전략 플러그인 매니저
    
    전략의 동적 로딩, 실행, 관리를 담당합니다.
    """
    
    def __init__(self, strategy_dir: str = "app/strategies/plugins"):
        """
        전략 매니저 초기화
        
        Args:
            strategy_dir: 전략 플러그인 디렉토리 경로
        """
        self.strategy_dir = Path(strategy_dir)
        self.loaded_strategies: Dict[str, Type[StrategyBase]] = {}
        self.active_strategies: Dict[str, StrategyBase] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        
        # 스레드 풀 (비동기 작업용)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self.logger = logging.getLogger(f"{__name__}.StrategyManager")
        self.logger.info("전략 매니저 초기화 완료")
    
    def discover_strategies(self) -> List[str]:
        """
        전략 플러그인 디렉토리에서 사용 가능한 전략 스캔
        
        Returns:
            List[str]: 발견된 전략 파일 목록
        """
        strategy_files = []
        
        try:
            if not self.strategy_dir.exists():
                self.logger.warning(f"전략 디렉토리가 존재하지 않음: {self.strategy_dir}")
                return strategy_files
            
            for file_path in self.strategy_dir.glob("*.py"):
                if file_path.name.startswith("__"):
                    continue
                    
                strategy_files.append(file_path.stem)
                self.logger.debug(f"전략 파일 발견: {file_path.name}")
            
            self.logger.info(f"총 {len(strategy_files)}개 전략 파일 발견")
            return strategy_files
            
        except Exception as e:
            self.logger.error(f"전략 스캔 중 오류: {e}")
            return []
    
    def load_strategy(self, strategy_name: str) -> bool:
        """
        특정 전략 로드
        
        Args:
            strategy_name: 로드할 전략 이름
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            if strategy_name in self.loaded_strategies:
                self.logger.info(f"전략 '{strategy_name}'은 이미 로드됨")
                return True
            
            # 전략 모듈 동적 import
            module_path = f"app.strategies.plugins.{strategy_name}"
            
            try:
                # 모듈이 이미 로드된 경우 리로드
                if module_path in sys.modules:
                    importlib.reload(sys.modules[module_path])
                else:
                    importlib.import_module(module_path)
                
                module = sys.modules[module_path]
                
            except ImportError as e:
                raise StrategyLoadError(f"전략 모듈 import 실패: {e}")
            
            # StrategyBase를 상속받은 클래스 찾기
            strategy_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, StrategyBase) and 
                    obj is not StrategyBase and
                    obj.__module__ == module_path):
                    strategy_class = obj
                    break
            
            if not strategy_class:
                raise StrategyLoadError(f"유효한 전략 클래스를 찾을 수 없음: {strategy_name}")
            
            # 전략 클래스 등록
            self.loaded_strategies[strategy_name] = strategy_class
            self.logger.info(f"전략 '{strategy_name}' 로드 성공")
            
            return True
            
        except Exception as e:
            self.logger.error(f"전략 '{strategy_name}' 로드 실패: {e}")
            return False
    
    def load_all_strategies(self) -> Dict[str, bool]:
        """
        모든 사용 가능한 전략 로드
        
        Returns:
            Dict[str, bool]: 전략별 로드 결과
        """
        strategy_files = self.discover_strategies()
        results = {}
        
        for strategy_name in strategy_files:
            results[strategy_name] = self.load_strategy(strategy_name)
        
        loaded_count = sum(results.values())
        self.logger.info(f"전략 로드 완료: {loaded_count}/{len(strategy_files)}개 성공")
        
        return results
    
    def create_strategy_instance(
        self, 
        strategy_name: str, 
        config: StrategyConfig
    ) -> Optional[StrategyBase]:
        """
        전략 인스턴스 생성
        
        Args:
            strategy_name: 전략 이름
            config: 전략 설정
            
        Returns:
            StrategyBase: 생성된 전략 인스턴스
        """
        try:
            if strategy_name not in self.loaded_strategies:
                if not self.load_strategy(strategy_name):
                    return None
            
            strategy_class = self.loaded_strategies[strategy_name]
            instance = strategy_class(config)
            
            # 전략 설정 저장
            self.strategy_configs[strategy_name] = config
            
            self.logger.info(f"전략 인스턴스 생성 완료: {strategy_name}")
            return instance
            
        except Exception as e:
            self.logger.error(f"전략 인스턴스 생성 실패 '{strategy_name}': {e}")
            return None
    
    def start_strategy(self, strategy_name: str, config: StrategyConfig) -> bool:
        """
        전략 시작
        
        Args:
            strategy_name: 시작할 전략 이름
            config: 전략 설정
            
        Returns:
            bool: 시작 성공 여부
        """
        try:
            # 이미 실행 중인 전략 확인
            if strategy_name in self.active_strategies:
                existing_strategy = self.active_strategies[strategy_name]
                if existing_strategy.get_status() == StrategyStatus.ACTIVE:
                    self.logger.warning(f"전략 '{strategy_name}'은 이미 실행 중")
                    return True
            
            # 전략 인스턴스 생성
            strategy_instance = self.create_strategy_instance(strategy_name, config)
            if not strategy_instance:
                return False
            
            # 전략 시작
            if strategy_instance.start():
                self.active_strategies[strategy_name] = strategy_instance
                self.logger.info(f"전략 '{strategy_name}' 시작됨")
                return True
            else:
                self.logger.error(f"전략 '{strategy_name}' 시작 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"전략 시작 중 오류 '{strategy_name}': {e}")
            return False
    
    def stop_strategy(self, strategy_name: str) -> bool:
        """
        전략 정지
        
        Args:
            strategy_name: 정지할 전략 이름
            
        Returns:
            bool: 정지 성공 여부
        """
        try:
            if strategy_name not in self.active_strategies:
                self.logger.warning(f"활성 전략 목록에 '{strategy_name}' 없음")
                return True
            
            strategy = self.active_strategies[strategy_name]
            if strategy.stop():
                del self.active_strategies[strategy_name]
                self.logger.info(f"전략 '{strategy_name}' 정지됨")
                return True
            else:
                self.logger.error(f"전략 '{strategy_name}' 정지 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"전략 정지 중 오류 '{strategy_name}': {e}")
            return False
    
    def pause_strategy(self, strategy_name: str) -> bool:
        """전략 일시정지"""
        if strategy_name in self.active_strategies:
            return self.active_strategies[strategy_name].pause()
        return False
    
    def resume_strategy(self, strategy_name: str) -> bool:
        """전략 재개"""
        if strategy_name in self.active_strategies:
            return self.active_strategies[strategy_name].resume()
        return False
    
    def get_strategy_status(self, strategy_name: str) -> Optional[StrategyStatus]:
        """전략 상태 조회"""
        if strategy_name in self.active_strategies:
            return self.active_strategies[strategy_name].get_status()
        return None
    
    def get_all_strategy_statuses(self) -> Dict[str, StrategyStatus]:
        """모든 활성 전략 상태 조회"""
        return {
            name: strategy.get_status()
            for name, strategy in self.active_strategies.items()
        }
    
    def process_market_data(
        self, 
        strategy_name: str, 
        market_data: MarketData
    ) -> Optional[TradingSignal]:
        """
        특정 전략으로 시장 데이터 처리
        
        Args:
            strategy_name: 전략 이름
            market_data: 시장 데이터
            
        Returns:
            TradingSignal: 생성된 거래 신호
        """
        try:
            if strategy_name not in self.active_strategies:
                return None
            
            strategy = self.active_strategies[strategy_name]
            
            # 전략이 활성 상태인지 확인
            if strategy.get_status() != StrategyStatus.ACTIVE:
                return None
            
            # 시장 데이터 분석
            signal = strategy.analyze(market_data)
            
            if signal and strategy.validate_signal(signal):
                self.logger.debug(f"전략 '{strategy_name}'에서 신호 생성: {signal.signal_type.value}")
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"시장 데이터 처리 중 오류 '{strategy_name}': {e}")
            return None
    
    async def process_market_data_async(
        self, 
        strategy_name: str, 
        market_data: MarketData
    ) -> Optional[TradingSignal]:
        """비동기 시장 데이터 처리"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.process_market_data,
            strategy_name,
            market_data
        )
    
    def process_all_strategies(
        self, 
        market_data: MarketData
    ) -> Dict[str, Optional[TradingSignal]]:
        """
        모든 활성 전략으로 시장 데이터 처리
        
        Args:
            market_data: 시장 데이터
            
        Returns:
            Dict[str, TradingSignal]: 전략별 생성된 신호
        """
        signals = {}
        
        for strategy_name in self.active_strategies:
            signal = self.process_market_data(strategy_name, market_data)
            signals[strategy_name] = signal
        
        return signals
    
    def get_loaded_strategies(self) -> List[str]:
        """로드된 전략 목록 반환"""
        return list(self.loaded_strategies.keys())
    
    def get_active_strategies(self) -> List[str]:
        """활성 전략 목록 반환"""
        return list(self.active_strategies.keys())
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """전략 설정 조회"""
        return self.strategy_configs.get(strategy_name)
    
    def update_strategy_config(
        self, 
        strategy_name: str, 
        new_params: Dict[str, Any]
    ) -> bool:
        """
        전략 설정 업데이트
        
        Args:
            strategy_name: 전략 이름
            new_params: 새로운 매개변수
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            if strategy_name in self.active_strategies:
                success = self.active_strategies[strategy_name].update_config(new_params)
                if success and strategy_name in self.strategy_configs:
                    self.strategy_configs[strategy_name].parameters.update(new_params)
                return success
            return False
            
        except Exception as e:
            self.logger.error(f"전략 설정 업데이트 실패 '{strategy_name}': {e}")
            return False
    
    def get_performance_metrics(self, strategy_name: str) -> Optional[Dict[str, float]]:
        """전략 성능 지표 조회"""
        if strategy_name in self.active_strategies:
            return self.active_strategies[strategy_name].get_performance_metrics()
        return None
    
    def get_all_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """모든 활성 전략의 성능 지표 조회"""
        return {
            name: strategy.get_performance_metrics()
            for name, strategy in self.active_strategies.items()
        }
    
    def validate_strategy_requirements(self, strategy_name: str) -> bool:
        """
        전략 요구사항 검증
        
        Args:
            strategy_name: 검증할 전략 이름
            
        Returns:
            bool: 요구사항 충족 여부
        """
        try:
            if strategy_name not in self.loaded_strategies:
                return False
            
            strategy_class = self.loaded_strategies[strategy_name]
            
            # 필수 메서드 존재 확인
            required_methods = ['analyze', 'get_position_size', 'validate_signal', 'get_required_parameters']
            for method in required_methods:
                if not hasattr(strategy_class, method):
                    self.logger.error(f"전략 '{strategy_name}'에 필수 메서드 '{method}' 누락")
                    return False
            
            self.logger.info(f"전략 '{strategy_name}' 요구사항 검증 통과")
            return True
            
        except Exception as e:
            self.logger.error(f"전략 요구사항 검증 실패 '{strategy_name}': {e}")
            return False
    
    def reload_strategy(self, strategy_name: str) -> bool:
        """
        전략 리로드 (개발 시 유용)
        
        Args:
            strategy_name: 리로드할 전략 이름
            
        Returns:
            bool: 리로드 성공 여부
        """
        try:
            # 활성 전략이면 먼저 정지
            was_active = False
            config = None
            
            if strategy_name in self.active_strategies:
                was_active = True
                config = self.active_strategies[strategy_name].get_config()
                self.stop_strategy(strategy_name)
            
            # 로드된 전략에서 제거
            if strategy_name in self.loaded_strategies:
                del self.loaded_strategies[strategy_name]
            
            # 다시 로드
            if self.load_strategy(strategy_name):
                # 이전에 활성 상태였다면 다시 시작
                if was_active and config:
                    self.start_strategy(strategy_name, config)
                
                self.logger.info(f"전략 '{strategy_name}' 리로드 완료")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"전략 리로드 실패 '{strategy_name}': {e}")
            return False
    
    def stop_all_strategies(self) -> Dict[str, bool]:
        """모든 활성 전략 정지"""
        results = {}
        
        # 활성 전략 목록 복사 (반복 중 수정 방지)
        active_strategies = list(self.active_strategies.keys())
        
        for strategy_name in active_strategies:
            results[strategy_name] = self.stop_strategy(strategy_name)
        
        self.logger.info(f"모든 전략 정지 완료: {sum(results.values())}/{len(results)}개 성공")
        return results
    
    def get_manager_status(self) -> Dict[str, Any]:
        """매니저 상태 정보 반환"""
        return {
            'loaded_strategies': len(self.loaded_strategies),
            'active_strategies': len(self.active_strategies),
            'strategy_list': {
                'loaded': list(self.loaded_strategies.keys()),
                'active': list(self.active_strategies.keys())
            },
            'status_summary': self.get_all_strategy_statuses(),
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 모든 전략 정지
            self.stop_all_strategies()
            
            # 스레드 풀 종료
            self.thread_pool.shutdown(wait=True)
            
            self.logger.info("전략 매니저 정리 완료")
            
        except Exception as e:
            self.logger.error(f"매니저 정리 중 오류: {e}")
    
    def __del__(self):
        """소멸자"""
        try:
            self.cleanup()
        except:
            pass


# 전역 전략 매니저 인스턴스
_strategy_manager_instance: Optional[StrategyManager] = None


def get_strategy_manager() -> StrategyManager:
    """
    전략 매니저 싱글톤 인스턴스 반환
    
    Returns:
        StrategyManager: 전역 전략 매니저 인스턴스
    """
    global _strategy_manager_instance
    
    if _strategy_manager_instance is None:
        _strategy_manager_instance = StrategyManager()
    
    return _strategy_manager_instance


def reset_strategy_manager():
    """전략 매니저 인스턴스 초기화 (테스트용)"""
    global _strategy_manager_instance
    
    if _strategy_manager_instance:
        _strategy_manager_instance.cleanup()
    
    _strategy_manager_instance = None