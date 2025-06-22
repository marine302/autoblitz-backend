# 파일: app/monitoring/metrics.py
# 경로: /workspaces/autoblitz-backend/app/monitoring/metrics.py
"""
AutoBlitz 커스텀 메트릭 수집기
API 성능, 봇 상태, 시스템 리소스 모니터링
"""

import time
import psutil
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging

from .cloudwatch import (
    MetricData, MetricUnit, send_metric, send_metrics,
    cloudwatch_client
)

logger = logging.getLogger(__name__)

@dataclass
class APIMetrics:
    """API 성능 메트릭"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: datetime
    user_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class BotMetrics:
    """봇 성능 메트릭"""
    bot_id: str
    user_id: str
    strategy: str
    trades_count: int
    success_rate: float
    profit_loss: float
    active: bool
    timestamp: datetime

@dataclass
class SystemMetrics:
    """시스템 리소스 메트릭"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_connections: int
    timestamp: datetime

class MetricsCollector:
    """메트릭 수집 및 전송 관리자"""
    
    def __init__(self):
        self.api_metrics: List[APIMetrics] = []
        self.bot_metrics: List[BotMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        self.batch_size = 20
        self.flush_interval = 60  # 60초마다 플러시
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """메트릭 수집 시작"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())
        logger.info("메트릭 수집기 시작")
    
    async def stop(self):
        """메트릭 수집 중지"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # 남은 메트릭 플러시
        await self.flush_all()
        logger.info("메트릭 수집기 중지")
    
    async def record_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        user_id: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """API 호출 메트릭 기록"""
        metric = APIMetrics(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            error_message=error_message
        )
        
        self.api_metrics.append(metric)
        
        # 즉시 전송할 중요 메트릭
        await self._send_api_metrics_immediate(metric)
    
    async def record_bot_performance(
        self,
        bot_id: str,
        user_id: str,
        strategy: str,
        trades_count: int,
        success_rate: float,
        profit_loss: float,
        active: bool
    ):
        """봇 성능 메트릭 기록"""
        metric = BotMetrics(
            bot_id=bot_id,
            user_id=user_id,
            strategy=strategy,
            trades_count=trades_count,
            success_rate=success_rate,
            profit_loss=profit_loss,
            active=active,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.bot_metrics.append(metric)
        
        # 즉시 전송할 중요 메트릭
        await self._send_bot_metrics_immediate(metric)
    
    async def record_system_metrics(self):
        """시스템 리소스 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # 네트워크 연결 수
            connections = len(psutil.net_connections())
            
            metric = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_usage_percent=disk_usage_percent,
                active_connections=connections,
                timestamp=datetime.now(timezone.utc)
            )
            
            self.system_metrics.append(metric)
            
            # 즉시 전송
            await self._send_system_metrics_immediate(metric)
            
        except Exception as e:
            logger.error(f"시스템 메트릭 수집 실패: {e}")
    
    async def _send_api_metrics_immediate(self, metric: APIMetrics):
        """중요 API 메트릭 즉시 전송"""
        metrics_to_send = []
        
        # API 응답시간
        metrics_to_send.append(MetricData(
            metric_name="API_ResponseTime",
            value=metric.response_time,
            unit=MetricUnit.MILLISECONDS,
            dimensions={
                "Endpoint": metric.endpoint,
                "Method": metric.method,
                "StatusCode": str(metric.status_code)
            }
        ))
        
        # API 호출 수
        metrics_to_send.append(MetricData(
            metric_name="API_CallCount",
            value=1,
            unit=MetricUnit.COUNT,
            dimensions={
                "Endpoint": metric.endpoint,
                "Method": metric.method,
                "StatusCode": str(metric.status_code)
            }
        ))
        
        # 에러 메트릭
        if metric.status_code >= 400:
            metrics_to_send.append(MetricData(
                metric_name="API_ErrorCount",
                value=1,
                unit=MetricUnit.COUNT,
                dimensions={
                    "Endpoint": metric.endpoint,
                    "StatusCode": str(metric.status_code)
                }
            ))
        
        await send_metrics(metrics_to_send)
    
    async def _send_bot_metrics_immediate(self, metric: BotMetrics):
        """중요 봇 메트릭 즉시 전송"""
        metrics_to_send = []
        
        # 봇 활성 수
        metrics_to_send.append(MetricData(
            metric_name="Bot_ActiveCount",
            value=1 if metric.active else 0,
            unit=MetricUnit.COUNT,
            dimensions={
                "Strategy": metric.strategy,
                "UserId": metric.user_id
            }
        ))
        
        # 봇 성공률
        if metric.success_rate >= 0:
            metrics_to_send.append(MetricData(
                metric_name="Bot_SuccessRate",
                value=metric.success_rate,
                unit=MetricUnit.PERCENT,
                dimensions={
                    "BotId": metric.bot_id,
                    "Strategy": metric.strategy
                }
            ))
        
        # 봇 수익률
        metrics_to_send.append(MetricData(
            metric_name="Bot_ProfitLoss",
            value=metric.profit_loss,
            unit=MetricUnit.COUNT,  # 금액
            dimensions={
                "BotId": metric.bot_id,
                "Strategy": metric.strategy
            }
        ))
        
        # 거래 횟수
        if metric.trades_count > 0:
            metrics_to_send.append(MetricData(
                metric_name="Bot_TradesCount",
                value=metric.trades_count,
                unit=MetricUnit.COUNT,
                dimensions={
                    "BotId": metric.bot_id,
                    "Strategy": metric.strategy
                }
            ))
        
        await send_metrics(metrics_to_send)
    
    async def _send_system_metrics_immediate(self, metric: SystemMetrics):
        """시스템 메트릭 즉시 전송"""
        metrics_to_send = []
        
        # CPU 사용률
        metrics_to_send.append(MetricData(
            metric_name="System_CPUUsage",
            value=metric.cpu_percent,
            unit=MetricUnit.PERCENT
        ))
        
        # 메모리 사용률
        metrics_to_send.append(MetricData(
            metric_name="System_MemoryUsage",
            value=metric.memory_percent,
            unit=MetricUnit.PERCENT
        ))
        
        # 메모리 사용량 (MB)
        metrics_to_send.append(MetricData(
            metric_name="System_MemoryUsedMB",
            value=metric.memory_used_mb,
            unit=MetricUnit.MEGABYTES
        ))
        
        # 디스크 사용률
        metrics_to_send.append(MetricData(
            metric_name="System_DiskUsage",
            value=metric.disk_usage_percent,
            unit=MetricUnit.PERCENT
        ))
        
        # 활성 연결 수
        metrics_to_send.append(MetricData(
            metric_name="System_ActiveConnections",
            value=metric.active_connections,
            unit=MetricUnit.COUNT
        ))
        
        await send_metrics(metrics_to_send)
    
    async def flush_all(self):
        """모든 메트릭 플러시"""
        try:
            # 배치로 나머지 메트릭들 전송
            all_metrics = []
            
            # API 메트릭 배치 처리
            for api_metric in self.api_metrics:
                # 추가 집계 메트릭들 (배치에서만)
                pass
            
            # 메트릭 클리어
            self.api_metrics.clear()
            self.bot_metrics.clear()
            self.system_metrics.clear()
            
            if all_metrics:
                await send_metrics(all_metrics)
                logger.info(f"메트릭 플러시 완료: {len(all_metrics)}개")
            
        except Exception as e:
            logger.error(f"메트릭 플러시 실패: {e}")
    
    async def _flush_loop(self):
        """주기적 메트릭 플러시"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_all()
                
                # 시스템 메트릭 주기적 수집
                await self.record_system_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"메트릭 플러시 루프 오류: {e}")

# 글로벌 메트릭 수집기
metrics_collector = MetricsCollector()

@asynccontextmanager
async def measure_api_call(endpoint: str, method: str, user_id: Optional[str] = None):
    """API 호출 시간 측정 컨텍스트 매니저"""
    start_time = time.time()
    status_code = 200
    error_message = None
    
    try:
        yield
    except Exception as e:
        status_code = 500
        error_message = str(e)
        raise
    finally:
        response_time = (time.time() - start_time) * 1000  # 밀리초
        await metrics_collector.record_api_call(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            user_id=user_id,
            error_message=error_message
        )

# 편의 함수들
async def record_api_performance(
    endpoint: str,
    method: str,
    status_code: int,
    response_time: float,
    user_id: Optional[str] = None
):
    """API 성능 기록"""
    await metrics_collector.record_api_call(
        endpoint, method, status_code, response_time, user_id
    )

async def record_bot_status(
    bot_id: str,
    user_id: str,
    strategy: str,
    trades: int = 0,
    success_rate: float = 0.0,
    profit: float = 0.0,
    active: bool = True
):
    """봇 상태 기록"""
    await metrics_collector.record_bot_performance(
        bot_id, user_id, strategy, trades, success_rate, profit, active
    )

async def record_system_status():
    """시스템 상태 기록"""
    await metrics_collector.record_system_metrics()