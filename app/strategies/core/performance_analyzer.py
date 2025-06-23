# 파일명: app/strategies/core/performance_analyzer.py
"""
성과 분석 모듈
백테스트 결과를 상세 분석하고 다양한 성과 지표를 계산합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from enum import Enum

from .backtest_engine import BacktestResults, Trade

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """분석 유형"""
    BASIC = "BASIC"
    DETAILED = "DETAILED"
    COMPREHENSIVE = "COMPREHENSIVE"


@dataclass
class PerformanceMetrics:
    """성과 지표"""
    # 수익률 지표
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_return: float = 0.0
    daily_return: float = 0.0
    compound_annual_growth_rate: float = 0.0
    
    # 리스크 지표
    volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0
    
    # 리스크 조정 수익률
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # 거래 성과 지표
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    kelly_criterion: float = 0.0
    
    # 일관성 지표
    consistency_score: float = 0.0
    stability_ratio: float = 0.0
    
    # 벤치마크 대비 지표
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None


@dataclass
class TradingStatistics:
    """거래 통계"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    win_rate: float = 0.0
    loss_rate: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    
    avg_holding_time: Optional[timedelta] = None
    max_holding_time: Optional[timedelta] = None
    min_holding_time: Optional[timedelta] = None
    
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # 연속 거래 통계
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0
    
    # 시간대별 성과
    hourly_performance: Dict[int, float] = None
    daily_performance: Dict[str, float] = None
    monthly_performance: Dict[str, float] = None


@dataclass
class RiskAnalysis:
    """리스크 분석"""
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    recovery_time: int = 0
    
    value_at_risk_95: float = 0.0
    value_at_risk_99: float = 0.0
    conditional_var_95: float = 0.0
    conditional_var_99: float = 0.0
    
    downside_deviation: float = 0.0
    ulcer_index: float = 0.0
    
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    worst_month: float = 0.0
    worst_week: float = 0.0
    worst_day: float = 0.0


class PerformanceAnalyzer:
    """성과 분석기"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate  # 무위험 수익률 (연 2%)
        
    def analyze(self, results: BacktestResults, 
               analysis_type: AnalysisType = AnalysisType.DETAILED) -> Dict[str, Any]:
        """성과 분석 실행"""
        
        logger.info(f"성과 분석 시작: {analysis_type.value}")
        
        analysis = {
            'summary': self._create_summary(results),
            'metrics': self._calculate_performance_metrics(results),
            'trading_stats': self._calculate_trading_statistics(results),
            'risk_analysis': self._calculate_risk_analysis(results)
        }
        
        if analysis_type in [AnalysisType.DETAILED, AnalysisType.COMPREHENSIVE]:
            analysis.update({
                'monthly_returns': self._calculate_monthly_returns(results),
                'drawdown_analysis': self._analyze_drawdowns(results),
                'correlation_analysis': self._analyze_correlations(results)
            })
        
        if analysis_type == AnalysisType.COMPREHENSIVE:
            analysis.update({
                'rolling_metrics': self._calculate_rolling_metrics(results),
                'scenario_analysis': self._perform_scenario_analysis(results),
                'attribution_analysis': self._perform_attribution_analysis(results)
            })
        
        logger.info("성과 분석 완료")
        return analysis
    
    def _create_summary(self, results: BacktestResults) -> Dict[str, Any]:
        """요약 정보 생성"""
        
        if not results.trades:
            return {
                'status': 'NO_TRADES',
                'message': '거래 기록이 없습니다',
                'period': f"{results.config.start_date} ~ {results.config.end_date}",
                'initial_balance': results.config.initial_balance
            }
        
        final_balance = results.equity_curve[-1] if results.equity_curve else results.config.initial_balance
        
        return {
            'status': 'SUCCESS',
            'period': f"{results.config.start_date} ~ {results.config.end_date}",
            'initial_balance': results.config.initial_balance,
            'final_balance': final_balance,
            'total_return': results.total_return,
            'total_trades': len(results.trades),
            'win_rate': results.win_rate,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'best_trade': max(results.trades, key=lambda t: t.pnl or 0).pnl if results.trades else 0,
            'worst_trade': min(results.trades, key=lambda t: t.pnl or 0).pnl if results.trades else 0
        }
    
    def _calculate_performance_metrics(self, results: BacktestResults) -> PerformanceMetrics:
        """성과 지표 계산"""
        
        metrics = PerformanceMetrics()
        
        if not results.equity_curve or len(results.equity_curve) < 2:
            return metrics
        
        # 기본 수익률 지표
        metrics.total_return = results.total_return
        metrics.annual_return = results.annual_return
        metrics.monthly_return = results.monthly_return
        metrics.daily_return = results.daily_return
        
        # 복리 연간 성장률 (CAGR)
        initial_value = results.equity_curve[0]
        final_value = results.equity_curve[-1]
        total_days = len(results.equity_curve)
        
        if initial_value > 0 and total_days > 0:
            metrics.compound_annual_growth_rate = (
                (final_value / initial_value) ** (365 / total_days) - 1
            ) * 100
        
        # 수익률 시계열 계산
        returns = np.array(results.equity_curve[1:]) / np.array(results.equity_curve[:-1]) - 1
        
        if len(returns) > 0:
            # 변동성 지표
            metrics.volatility = np.std(returns) * np.sqrt(252) * 100  # 연환산
            
            # 하방 변동성 (음수 수익률만)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                metrics.downside_volatility = np.std(negative_returns) * np.sqrt(252) * 100
            
            # VaR 및 CVaR
            metrics.value_at_risk_95 = np.percentile(returns, 5) * 100
            metrics.conditional_var_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * 100
            
            # 리스크 조정 수익률
            daily_risk_free = self.risk_free_rate / 252
            excess_returns = returns - daily_risk_free
            
            if metrics.volatility > 0:
                metrics.sharpe_ratio = np.mean(excess_returns) * 252 / (metrics.volatility / 100)
            
            if metrics.downside_volatility > 0:
                metrics.sortino_ratio = np.mean(excess_returns) * 252 / (metrics.downside_volatility / 100)
            
            # 칼마 비율
            if results.max_drawdown > 0:
                metrics.calmar_ratio = metrics.annual_return / results.max_drawdown
            
            # 오메가 비율 (간단화)
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) > 0:
                metrics.omega_ratio = np.sum(positive_returns) / abs(np.sum(negative_returns))
        
        # 거래 기반 지표
        if results.trades:
            wins = [t for t in results.trades if t.pnl and t.pnl > 0]
            losses = [t for t in results.trades if t.pnl and t.pnl < 0]
            
            metrics.win_rate = len(wins) / len(results.trades) * 100
            
            if wins and losses:
                avg_win = np.mean([t.pnl for t in wins])
                avg_loss = abs(np.mean([t.pnl for t in losses]))
                
                metrics.profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
                
                # 기댓값
                win_prob = len(wins) / len(results.trades)
                loss_prob = len(losses) / len(results.trades)
                metrics.expectancy = (win_prob * avg_win) - (loss_prob * avg_loss)
                
                # 켈리 공식
                if avg_loss > 0:
                    metrics.kelly_criterion = (win_prob * avg_win - loss_prob * avg_loss) / avg_win
        
        # 최대 낙폭
        metrics.max_drawdown = results.max_drawdown
        metrics.max_drawdown_duration = results.max_drawdown_duration
        
        return metrics
    
    def _calculate_trading_statistics(self, results: BacktestResults) -> TradingStatistics:
        """거래 통계 계산"""
        
        stats = TradingStatistics()
        
        if not results.trades:
            return stats
        
        # 기본 통계
        stats.total_trades = len(results.trades)
        
        wins = [t for t in results.trades if t.pnl and t.pnl > 0]
        losses = [t for t in results.trades if t.pnl and t.pnl < 0]
        breakevens = [t for t in results.trades if t.pnl == 0]
        
        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.breakeven_trades = len(breakevens)
        
        stats.win_rate = (stats.winning_trades / stats.total_trades) * 100
        stats.loss_rate = (stats.losing_trades / stats.total_trades) * 100
        
        # 손익 통계
        if wins:
            win_amounts = [t.pnl for t in wins]
            stats.avg_win = np.mean(win_amounts)
            stats.max_win = max(win_amounts)
        
        if losses:
            loss_amounts = [abs(t.pnl) for t in losses]  # 절댓값
            stats.avg_loss = np.mean(loss_amounts)
            stats.max_loss = max(loss_amounts)
        
        # 수익 팩터 및 기댓값
        if stats.avg_loss > 0:
            stats.profit_factor = stats.avg_win / stats.avg_loss
        
        if stats.total_trades > 0:
            stats.expectancy = (stats.win_rate/100 * stats.avg_win) - (stats.loss_rate/100 * stats.avg_loss)
        
        # 보유 시간 통계
        holding_times = [t.holding_time for t in results.trades if t.holding_time]
        if holding_times:
            stats.avg_holding_time = sum(holding_times, timedelta()) / len(holding_times)
            stats.max_holding_time = max(holding_times)
            stats.min_holding_time = min(holding_times)
        
        # 연속 거래 통계
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in results.trades:
            if trade.pnl and trade.pnl > 0:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            elif trade.pnl and trade.pnl < 0:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
        
        stats.max_consecutive_wins = max_win_streak
        stats.max_consecutive_losses = max_loss_streak
        stats.current_streak = current_streak
        
        # 시간대별 성과 (간단화)
        stats.hourly_performance = {}
        stats.daily_performance = {}
        stats.monthly_performance = {}
        
        for trade in results.trades:
            if trade.entry_time and trade.pnl:
                hour = trade.entry_time.hour
                day = trade.entry_time.strftime('%A')
                month = trade.entry_time.strftime('%B')
                
                stats.hourly_performance[hour] = stats.hourly_performance.get(hour, 0) + trade.pnl
                stats.daily_performance[day] = stats.daily_performance.get(day, 0) + trade.pnl
                stats.monthly_performance[month] = stats.monthly_performance.get(month, 0) + trade.pnl
        
        return stats
    
    def _calculate_risk_analysis(self, results: BacktestResults) -> RiskAnalysis:
        """리스크 분석"""
        
        risk = RiskAnalysis()
        
        if not results.equity_curve or len(results.equity_curve) < 2:
            return risk
        
        # 기본 리스크 지표
        risk.max_drawdown = results.max_drawdown
        risk.max_drawdown_duration = results.max_drawdown_duration
        
        # 수익률 계산
        returns = np.array(results.equity_curve[1:]) / np.array(results.equity_curve[:-1]) - 1
        
        if len(returns) > 0:
            # VaR 및 CVaR
            risk.value_at_risk_95 = np.percentile(returns, 5) * 100
            risk.value_at_risk_99 = np.percentile(returns, 1) * 100
            
            var_95_threshold = np.percentile(returns, 5)
            var_99_threshold = np.percentile(returns, 1)
            
            tail_returns_95 = returns[returns <= var_95_threshold]
            tail_returns_99 = returns[returns <= var_99_threshold]
            
            if len(tail_returns_95) > 0:
                risk.conditional_var_95 = np.mean(tail_returns_95) * 100
            
            if len(tail_returns_99) > 0:
                risk.conditional_var_99 = np.mean(tail_returns_99) * 100
            
            # 하방 편차
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                risk.downside_deviation = np.std(negative_returns) * 100
            
            # 통계적 특성
            risk.skewness = float(pd.Series(returns).skew())
            risk.kurtosis = float(pd.Series(returns).kurtosis())
            
            # 테일 비율 (상위 5% vs 하위 5%)
            upper_tail = np.percentile(returns, 95)
            lower_tail = np.percentile(returns, 5)
            if lower_tail != 0:
                risk.tail_ratio = abs(upper_tail / lower_tail)
            
            # 최악의 기간
            if len(returns) >= 30:  # 월간
                monthly_returns = [returns[i:i+30] for i in range(0, len(returns)-29, 30)]
                monthly_cumulative = [np.prod(1 + r) - 1 for r in monthly_returns]
                if monthly_cumulative:
                    risk.worst_month = min(monthly_cumulative) * 100
            
            if len(returns) >= 7:  # 주간
                weekly_returns = [returns[i:i+7] for i in range(0, len(returns)-6, 7)]
                weekly_cumulative = [np.prod(1 + r) - 1 for r in weekly_returns]
                if weekly_cumulative:
                    risk.worst_week = min(weekly_cumulative) * 100
            
            risk.worst_day = min(returns) * 100
        
        return risk
    
    def _calculate_monthly_returns(self, results: BacktestResults) -> Dict[str, float]:
        """월별 수익률 계산"""
        
        if not results.timestamps or not results.equity_curve:
            return {}
        
        monthly_returns = {}
        
        # 월별 그룹화
        df = pd.DataFrame({
            'timestamp': results.timestamps,
            'equity': results.equity_curve[1:]  # 첫 번째 제외
        })
        
        if len(df) == 0:
            return {}
        
        df['month'] = df['timestamp'].dt.to_period('M')
        
        # 각 월의 첫날과 마지막날 equity로 수익률 계산
        for month, group in df.groupby('month'):
            if len(group) >= 2:
                start_equity = group.iloc[0]['equity']
                end_equity = group.iloc[-1]['equity']
                
                if start_equity > 0:
                    monthly_return = ((end_equity - start_equity) / start_equity) * 100
                    monthly_returns[str(month)] = monthly_return
        
        return monthly_returns
    
    def _analyze_drawdowns(self, results: BacktestResults) -> Dict[str, Any]:
        """낙폭 분석"""
        
        if not results.equity_curve:
            return {}
        
        equity = np.array(results.equity_curve)
        drawdowns = []
        peak = equity[0]
        in_drawdown = False
        drawdown_start = 0
        
        for i, value in enumerate(equity):
            if value > peak:
                if in_drawdown:
                    # 낙폭 회복 완료
                    drawdown_duration = i - drawdown_start
                    max_dd_in_period = (peak - min(equity[drawdown_start:i+1])) / peak
                    
                    drawdowns.append({
                        'start_index': drawdown_start,
                        'end_index': i,
                        'duration': drawdown_duration,
                        'max_drawdown': max_dd_in_period * 100,
                        'recovery_index': i
                    })
                    in_drawdown = False
                
                peak = value
            elif value < peak and not in_drawdown:
                # 새로운 낙폭 시작
                in_drawdown = True
                drawdown_start = i
        
        # 분석 결과
        analysis = {
            'total_drawdown_periods': len(drawdowns),
            'avg_drawdown': np.mean([dd['max_drawdown'] for dd in drawdowns]) if drawdowns else 0,
            'avg_duration': np.mean([dd['duration'] for dd in drawdowns]) if drawdowns else 0,
            'max_duration': max([dd['duration'] for dd in drawdowns]) if drawdowns else 0,
            'drawdown_details': drawdowns[:5]  # 상위 5개만
        }
        
        return analysis
    
    def _analyze_correlations(self, results: BacktestResults) -> Dict[str, Any]:
        """상관관계 분석 (플레이스홀더)"""
        
        # 실제 구현에서는 시장 데이터와의 상관관계 분석
        return {
            'market_correlation': None,
            'sector_correlation': None,
            'note': '벤치마크 데이터가 필요합니다'
        }
    
    def _calculate_rolling_metrics(self, results: BacktestResults) -> Dict[str, Any]:
        """롤링 지표 계산"""
        
        if not results.equity_curve or len(results.equity_curve) < 30:
            return {}
        
        equity = np.array(results.equity_curve)
        returns = equity[1:] / equity[:-1] - 1
        
        window = min(30, len(returns) // 4)  # 30일 또는 전체의 1/4
        
        rolling_sharpe = []
        rolling_volatility = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            
            if len(window_returns) > 0:
                vol = np.std(window_returns) * np.sqrt(252)
                excess_returns = window_returns - self.risk_free_rate/252
                sharpe = np.mean(excess_returns) * 252 / vol if vol > 0 else 0
                
                rolling_sharpe.append(sharpe)
                rolling_volatility.append(vol * 100)
        
        return {
            'rolling_sharpe_ratio': rolling_sharpe,
            'rolling_volatility': rolling_volatility,
            'window_size': window
        }
    
    def _perform_scenario_analysis(self, results: BacktestResults) -> Dict[str, Any]:
        """시나리오 분석 (플레이스홀더)"""
        
        return {
            'stress_test': 'Not implemented',
            'monte_carlo': 'Not implemented',
            'sensitivity_analysis': 'Not implemented'
        }
    
    def _perform_attribution_analysis(self, results: BacktestResults) -> Dict[str, Any]:
        """성과 기여도 분석"""
        
        if not results.trades:
            return {}
        
        # 거래별 기여도
        trade_contributions = []
        for trade in results.trades:
            if trade.pnl:
                contribution = trade.pnl / results.config.initial_balance * 100
                trade_contributions.append({
                    'trade_id': trade.trade_id,
                    'contribution': contribution,
                    'symbol': trade.symbol,
                    'side': trade.side
                })
        
        # 상위/하위 기여 거래
        sorted_trades = sorted(trade_contributions, key=lambda x: x['contribution'], reverse=True)
        
        return {
            'top_contributors': sorted_trades[:5],
            'worst_contributors': sorted_trades[-5:],
            'total_positive_contribution': sum(t['contribution'] for t in trade_contributions if t['contribution'] > 0),
            'total_negative_contribution': sum(t['contribution'] for t in trade_contributions if t['contribution'] < 0)
        }


def generate_performance_report(analysis: Dict[str, Any], format_type: str = "text") -> str:
    """성과 분석 리포트 생성"""
    
    if format_type == "text":
        return _generate_text_report(analysis)
    elif format_type == "html":
        return _generate_html_report(analysis)
    else:
        return str(analysis)


def _generate_text_report(analysis: Dict[str, Any]) -> str:
    """텍스트 형식 리포트 생성"""
    
    summary = analysis.get('summary', {})
    metrics = analysis.get('metrics', PerformanceMetrics())
    trading_stats = analysis.get('trading_stats', TradingStatistics())
    risk = analysis.get('risk_analysis', RiskAnalysis())
    
    report = f"""
🎯 백테스트 성과 분석 리포트
{'='*50}

📊 요약 정보:
- 기간: {summary.get('period', 'N/A')}
- 초기 자본: ${summary.get('initial_balance', 0):,.2f}
- 최종 자본: ${summary.get('final_balance', 0):,.2f}
- 총 수익률: {summary.get('total_return', 0):.2f}%

📈 수익률 지표:
- 연 수익률: {metrics.annual_return:.2f}%
- 월 수익률: {metrics.monthly_return:.2f}%
- 복리연성장률(CAGR): {metrics.compound_annual_growth_rate:.2f}%

⚖️ 리스크 조정 수익률:
- 샤프 비율: {metrics.sharpe_ratio:.2f}
- 소르티노 비율: {metrics.sortino_ratio:.2f}
- 칼마 비율: {metrics.calmar_ratio:.2f}

🎲 거래 통계:
- 총 거래: {trading_stats.total_trades}건
- 승률: {trading_stats.win_rate:.1f}%
- 평균 수익: ${trading_stats.avg_win:.2f}
- 평균 손실: ${trading_stats.avg_loss:.2f}
- 수익 팩터: {trading_stats.profit_factor:.2f}

⚠️ 리스크 분석:
- 최대 낙폭: {risk.max_drawdown:.2f}%
- 낙폭 지속기간: {risk.max_drawdown_duration}일
- VaR(95%): {risk.value_at_risk_95:.2f}%
- 변동성: {metrics.volatility:.2f}%

✨ 종합 평가:
"""
    
    # 종합 평가 로직
    score = 0
    if metrics.sharpe_ratio > 1.0:
        score += 2
    elif metrics.sharpe_ratio > 0.5:
        score += 1
    
    if trading_stats.win_rate > 60:
        score += 2
    elif trading_stats.win_rate > 50:
        score += 1
    
    if risk.max_drawdown < 10:
        score += 2
    elif risk.max_drawdown < 20:
        score += 1
    
    if score >= 5:
        evaluation = "🌟 우수한 성과"
    elif score >= 3:
        evaluation = "✅ 양호한 성과"
    elif score >= 1:
        evaluation = "⚠️ 보통 성과"
    else:
        evaluation = "❌ 개선 필요"
    
    report += f"- {evaluation} (점수: {score}/6)\n"
    
    return report


def _generate_html_report(analysis: Dict[str, Any]) -> str:
    """HTML 형식 리포트 생성 (간단 버전)"""
    
    summary = analysis.get('summary', {})
    
    return f"""
    <html>
    <head><title>백테스트 성과 분석</title></head>
    <body>
        <h1>백테스트 성과 분석 리포트</h1>
        <h2>요약</h2>
        <p>기간: {summary.get('period', 'N/A')}</p>
        <p>총 수익률: {summary.get('total_return', 0):.2f}%</p>
        <p>총 거래: {summary.get('total_trades', 0)}건</p>
        <p>승률: {summary.get('win_rate', 0):.1f}%</p>
    </body>
    </html>
    """


# 사용 예시
if __name__ == "__main__":
    # 예시는 backtest_engine.py와 연동하여 실행
    pass