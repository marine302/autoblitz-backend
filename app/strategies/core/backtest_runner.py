# 파일명: app/strategies/core/backtest_runner.py
"""
백테스트 실행기
백테스트 실행을 관리하고 결과를 처리합니다.
"""

import asyncio
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum

from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResults
from .performance_analyzer import PerformanceAnalyzer, AnalysisType
from .strategy_base import StrategyBase

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """실행 모드"""
    SINGLE = "SINGLE"           # 단일 백테스트
    BATCH = "BATCH"             # 배치 실행
    OPTIMIZATION = "OPTIMIZATION"  # 파라미터 최적화
    MONTE_CARLO = "MONTE_CARLO"    # 몬테카를로 시뮬레이션


class OptimizationMethod(Enum):
    """최적화 방법"""
    GRID_SEARCH = "GRID_SEARCH"
    RANDOM_SEARCH = "RANDOM_SEARCH"
    GENETIC_ALGORITHM = "GENETIC_ALGORITHM"
    BAYESIAN = "BAYESIAN"


@dataclass
class BacktestJob:
    """백테스트 작업"""
    job_id: str
    strategy_config: Dict[str, Any]
    backtest_config: BacktestConfig
    market_data: List[Dict]
    metadata: Dict[str, Any] = None
    status: str = "PENDING"
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class OptimizationConfig:
    """최적화 설정"""
    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    objective_function: str = "sharpe_ratio"  # sharpe_ratio, total_return, calmar_ratio
    maximize: bool = True
    max_iterations: int = 100
    population_size: int = 50  # GA용
    mutation_rate: float = 0.1  # GA용
    crossover_rate: float = 0.8  # GA용
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001


@dataclass
class BacktestSuite:
    """백테스트 스위트"""
    suite_id: str
    name: str
    description: str
    jobs: List[BacktestJob]
    results: Dict[str, BacktestResults] = None
    summary: Dict[str, Any] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None


class BacktestRunner:
    """백테스트 실행기"""
    
    def __init__(self, max_workers: int = 4, results_dir: str = "backtest_results"):
        self.max_workers = max_workers
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.analyzer = PerformanceAnalyzer()
        self.active_jobs: Dict[str, BacktestJob] = {}
        self.completed_jobs: Dict[str, BacktestResults] = {}
        
        # 진행상황 콜백
        self.progress_callbacks: List[Callable] = []
        
        logger.info(f"백테스트 실행기 초기화: {max_workers} workers")
    
    def add_progress_callback(self, callback: Callable[[str, float], None]):
        """진행상황 콜백 추가"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, job_id: str, progress: float):
        """진행상황 알림"""
        for callback in self.progress_callbacks:
            try:
                callback(job_id, progress)
            except Exception as e:
                logger.warning(f"진행상황 콜백 오류: {e}")
    
    async def run_single_backtest(self, 
                                strategy: StrategyBase,
                                config: BacktestConfig,
                                market_data: List[Dict],
                                job_id: Optional[str] = None) -> BacktestResults:
        """단일 백테스트 실행"""
        
        if not job_id:
            job_id = f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"단일 백테스트 시작: {job_id}")
        
        try:
            # 백테스트 실행
            engine = BacktestEngine(strategy, config)
            results = await asyncio.get_event_loop().run_in_executor(
                None, engine.run, market_data
            )
            
            # 성과 분석
            analysis = self.analyzer.analyze(results, AnalysisType.DETAILED)
            results.analysis = analysis
            
            # 결과 저장
            await self._save_results(job_id, results)
            
            logger.info(f"단일 백테스트 완료: {job_id}")
            return results
            
        except Exception as e:
            logger.error(f"단일 백테스트 실패 ({job_id}): {str(e)}")
            raise
    
    async def run_batch_backtest(self, 
                                jobs: List[BacktestJob],
                                execution_mode: ExecutionMode = ExecutionMode.BATCH) -> Dict[str, BacktestResults]:
        """배치 백테스트 실행"""
        
        logger.info(f"배치 백테스트 시작: {len(jobs)}개 작업")
        
        results = {}
        
        if execution_mode == ExecutionMode.BATCH:
            # 병렬 실행
            tasks = []
            for job in jobs:
                task = self._execute_single_job(job)
                tasks.append(task)
            
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for job, result in zip(jobs, completed_results):
                if isinstance(result, Exception):
                    logger.error(f"작업 실패 ({job.job_id}): {result}")
                else:
                    results[job.job_id] = result
        
        logger.info(f"배치 백테스트 완료: {len(results)}개 성공")
        return results
    
    async def run_parameter_optimization(self,
                                       strategy_class: type,
                                       base_config: BacktestConfig,
                                       market_data: List[Dict],
                                       parameter_ranges: Dict[str, List],
                                       optimization_config: OptimizationConfig) -> Dict[str, Any]:
        """파라미터 최적화 실행"""
        
        logger.info(f"파라미터 최적화 시작: {optimization_config.method.value}")
        
        if optimization_config.method == OptimizationMethod.GRID_SEARCH:
            return await self._grid_search_optimization(
                strategy_class, base_config, market_data, parameter_ranges, optimization_config
            )
        elif optimization_config.method == OptimizationMethod.RANDOM_SEARCH:
            return await self._random_search_optimization(
                strategy_class, base_config, market_data, parameter_ranges, optimization_config
            )
        else:
            raise NotImplementedError(f"최적화 방법 미구현: {optimization_config.method}")
    
    async def _grid_search_optimization(self,
                                      strategy_class: type,
                                      base_config: BacktestConfig,
                                      market_data: List[Dict],
                                      parameter_ranges: Dict[str, List],
                                      optimization_config: OptimizationConfig) -> Dict[str, Any]:
        """그리드 서치 최적화"""
        
        import itertools
        
        # 파라미터 조합 생성
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"그리드 서치: {len(param_combinations)}개 조합 테스트")
        
        best_score = float('-inf') if optimization_config.maximize else float('inf')
        best_params = None
        best_results = None
        
        results_history = []
        
        for i, param_combo in enumerate(param_combinations):
            # 파라미터 설정
            params = dict(zip(param_names, param_combo))
            
            try:
                # 전략 생성
                strategy = strategy_class(**params)
                
                # 백테스트 실행
                engine = BacktestEngine(strategy, base_config)
                results = await asyncio.get_event_loop().run_in_executor(
                    None, engine.run, market_data
                )
                
                # 목적 함수 계산
                score = self._calculate_objective_score(results, optimization_config.objective_function)
                
                # 최적해 업데이트
                is_better = (score > best_score) if optimization_config.maximize else (score < best_score)
                
                if is_better:
                    best_score = score
                    best_params = params.copy()
                    best_results = results
                
                results_history.append({
                    'iteration': i + 1,
                    'parameters': params,
                    'score': score,
                    'is_best': is_better
                })
                
                # 진행상황 알림
                progress = (i + 1) / len(param_combinations)
                self._notify_progress(f"optimization", progress)
                
                logger.debug(f"반복 {i+1}/{len(param_combinations)}: "
                           f"점수={score:.4f}, 최고={best_score:.4f}")
                
            except Exception as e:
                logger.warning(f"파라미터 조합 실패 ({params}): {e}")
                continue
        
        return {
            'method': 'grid_search',
            'best_parameters': best_params,
            'best_score': best_score,
            'best_results': best_results,
            'total_iterations': len(param_combinations),
            'successful_iterations': len(results_history),
            'results_history': results_history
        }
    
    async def _random_search_optimization(self,
                                        strategy_class: type,
                                        base_config: BacktestConfig,
                                        market_data: List[Dict],
                                        parameter_ranges: Dict[str, List],
                                        optimization_config: OptimizationConfig) -> Dict[str, Any]:
        """랜덤 서치 최적화"""
        
        import random
        
        logger.info(f"랜덤 서치: {optimization_config.max_iterations}회 반복")
        
        best_score = float('-inf') if optimization_config.maximize else float('inf')
        best_params = None
        best_results = None
        
        results_history = []
        no_improvement_count = 0
        
        for iteration in range(optimization_config.max_iterations):
            # 랜덤 파라미터 생성
            params = {}
            for param_name, param_range in parameter_ranges.items():
                params[param_name] = random.choice(param_range)
            
            try:
                # 전략 생성
                strategy = strategy_class(**params)
                
                # 백테스트 실행
                engine = BacktestEngine(strategy, base_config)
                results = await asyncio.get_event_loop().run_in_executor(
                    None, engine.run, market_data
                )
                
                # 목적 함수 계산
                score = self._calculate_objective_score(results, optimization_config.objective_function)
                
                # 최적해 업데이트
                is_better = (score > best_score) if optimization_config.maximize else (score < best_score)
                
                if is_better:
                    improvement = abs(score - best_score)
                    if improvement >= optimization_config.early_stopping_min_delta:
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    best_score = score
                    best_params = params.copy()
                    best_results = results
                else:
                    no_improvement_count += 1
                
                results_history.append({
                    'iteration': iteration + 1,
                    'parameters': params,
                    'score': score,
                    'is_best': is_better
                })
                
                # 조기 종료 확인
                if no_improvement_count >= optimization_config.early_stopping_patience:
                    logger.info(f"조기 종료: {no_improvement_count}회 개선 없음")
                    break
                
                # 진행상황 알림
                progress = (iteration + 1) / optimization_config.max_iterations
                self._notify_progress(f"optimization", progress)
                
            except Exception as e:
                logger.warning(f"랜덤 파라미터 실패 ({params}): {e}")
                continue
        
        return {
            'method': 'random_search',
            'best_parameters': best_params,
            'best_score': best_score,
            'best_results': best_results,
            'total_iterations': len(results_history),
            'early_stopped': no_improvement_count >= optimization_config.early_stopping_patience,
            'results_history': results_history
        }
    
    def _calculate_objective_score(self, results: BacktestResults, objective_function: str) -> float:
        """목적 함수 점수 계산"""
        
        if objective_function == "sharpe_ratio":
            return results.sharpe_ratio
        elif objective_function == "total_return":
            return results.total_return
        elif objective_function == "calmar_ratio":
            return results.calmar_ratio
        elif objective_function == "profit_factor":
            return results.profit_factor if hasattr(results, 'profit_factor') else 0
        elif objective_function == "win_rate":
            return results.win_rate
        elif objective_function == "max_drawdown":
            return -results.max_drawdown  # 낮을수록 좋음
        else:
            logger.warning(f"알 수 없는 목적 함수: {objective_function}")
            return 0.0
    
    async def _execute_single_job(self, job: BacktestJob) -> BacktestResults:
        """단일 작업 실행"""
        
        job.status = "RUNNING"
        job.started_at = datetime.now()
        
        try:
            # 전략 생성 (동적)
            strategy_config = job.strategy_config
            strategy_class_name = strategy_config.pop('class_name', 'ScalpingStrategy')
            
            # 여기서는 간단화 - 실제로는 동적 임포트 필요
            from ..plugins.scalping_strategy import create_scalping_strategy
            strategy = create_scalping_strategy()
            
            # 백테스트 실행
            engine = BacktestEngine(strategy, job.backtest_config)
            results = await asyncio.get_event_loop().run_in_executor(
                None, engine.run, job.market_data
            )
            
            job.status = "COMPLETED"
            job.completed_at = datetime.now()
            
            return results
            
        except Exception as e:
            job.status = "FAILED"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"작업 실행 실패 ({job.job_id}): {e}")
            raise
    
    async def _save_results(self, job_id: str, results: BacktestResults):
        """결과 저장"""
        
        try:
            # JSON 형식으로 저장 (기본 정보)
            json_path = self.results_dir / f"{job_id}.json"
            json_data = {
                'job_id': job_id,
                'config': asdict(results.config),
                'total_return': results.total_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'total_trades': len(results.trades),
                'created_at': datetime.now().isoformat()
            }
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            # Pickle 형식으로 전체 결과 저장
            pickle_path = self.results_dir / f"{job_id}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
            
            logger.debug(f"결과 저장 완료: {job_id}")
            
        except Exception as e:
            logger.error(f"결과 저장 실패 ({job_id}): {e}")
    
    async def load_results(self, job_id: str) -> Optional[BacktestResults]:
        """결과 로드"""
        
        pickle_path = self.results_dir / f"{job_id}.pkl"
        
        if not pickle_path.exists():
            return None
        
        try:
            with open(pickle_path, 'rb') as f:
                results = pickle.load(f)
            
            logger.debug(f"결과 로드 완료: {job_id}")
            return results
            
        except Exception as e:
            logger.error(f"결과 로드 실패 ({job_id}): {e}")
            return None
    
    def list_saved_results(self) -> List[Dict[str, Any]]:
        """저장된 결과 목록"""
        
        results = []
        
        for json_file in self.results_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                results.append(data)
            except Exception as e:
                logger.warning(f"결과 파일 읽기 실패 ({json_file}): {e}")
        
        return sorted(results, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def cleanup_old_results(self, days_to_keep: int = 30):
        """오래된 결과 정리"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_count = 0
        
        for file_path in self.results_dir.glob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    file_path.unlink()
                    cleaned_count += 1
        
        logger.info(f"오래된 결과 {cleaned_count}개 정리 완료")
    
    def create_comparison_report(self, job_ids: List[str]) -> Dict[str, Any]:
        """비교 리포트 생성"""
        
        comparison_data = []
        
        for job_id in job_ids:
            json_path = self.results_dir / f"{job_id}.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                comparison_data.append(data)
        
        if not comparison_data:
            return {'error': '비교할 데이터가 없습니다'}
        
        # 비교 분석
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades']
        comparison = {}
        
        for metric in metrics:
            values = [data.get(metric, 0) for data in comparison_data]
            comparison[metric] = {
                'values': values,
                'best': max(values),
                'worst': min(values),
                'average': sum(values) / len(values),
                'best_job': job_ids[values.index(max(values))]
            }
        
        return {
            'compared_jobs': job_ids,
            'comparison': comparison,
            'job_data': comparison_data
        }


def create_sample_optimization_config() -> OptimizationConfig:
    """샘플 최적화 설정 생성"""
    
    return OptimizationConfig(
        method=OptimizationMethod.RANDOM_SEARCH,
        objective_function="sharpe_ratio",
        maximize=True,
        max_iterations=50,
        early_stopping_patience=10
    )


def create_sample_parameter_ranges() -> Dict[str, List]:
    """샘플 파라미터 범위 생성"""
    
    return {
        'target_profit': [0.005, 0.008, 0.010, 0.012, 0.015],
        'stop_loss': [0.002, 0.003, 0.004, 0.005],
        'rsi_oversold': [20, 25, 30, 35],
        'rsi_overbought': [65, 70, 75, 80],
        'max_daily_trades': [20, 30, 40, 50]
    }


# 사용 예시
if __name__ == "__main__":
    import asyncio
    from ..plugins.scalping_strategy import create_scalping_strategy
    from .backtest_engine import create_sample_market_data
    
    async def main():
        # 백테스트 실행기 생성
        runner = BacktestRunner(max_workers=2)
        
        # 전략 생성
        strategy = create_scalping_strategy()
        
        # 백테스트 설정
        config = BacktestConfig(
            start_date="2024-01-01T00:00:00",
            end_date="2024-01-31T23:59:59",
            initial_balance=10000
        )
        
        # 샘플 데이터
        market_data = create_sample_market_data(days=30)
        
        # 단일 백테스트 실행
        results = await runner.run_single_backtest(strategy, config, market_data)
        
        print(f"🎯 백테스트 완료:")
        print(f"총 수익률: {results.total_return:.2f}%")
        print(f"샤프 비율: {results.sharpe_ratio:.2f}")
        print(f"총 거래: {len(results.trades)}건")
    
    # 실행
    asyncio.run(main())