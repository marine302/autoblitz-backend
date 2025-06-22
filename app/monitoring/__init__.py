# 파일: app/monitoring/__init__.py
# 경로: /workspaces/autoblitz-backend/app/monitoring/__init__.py
"""
AutoBlitz 모니터링 시스템 통합
CloudWatch, 메트릭 수집, 알림 시스템 통합 관리
"""

import logging
from .cloudwatch import cloudwatch_client, send_metric, send_metrics, MetricData, MetricUnit
from .metrics import (
    metrics_collector, 
    measure_api_call,
    record_api_performance,
    record_bot_status,
    record_system_status
)
from .alerts import alert_manager, trigger_custom_alert, check_metric_threshold, AlertSeverity

logger = logging.getLogger(__name__)

# 모니터링 시스템 전체 관리 클래스
class MonitoringSystem:
    """통합 모니터링 시스템"""
    
    def __init__(self):
        self.cloudwatch = cloudwatch_client
        self.metrics = metrics_collector
        self.alerts = alert_manager
        self._initialized = False
    
    async def initialize(self):
        """모니터링 시스템 초기화"""
        if self._initialized:
            return
        
        try:
            # 메트릭 수집기 시작
            await self.metrics.start()
            logger.info("메트릭 수집기 시작 완료")
            
            # 알림 시스템 시작
            await self.alerts.start()
            logger.info("알림 시스템 시작 완료")
            
            # 초기 시스템 메트릭 수집
            await record_system_status()
            
            # 시스템 시작 알림
            await trigger_custom_alert(
                name="system_startup",
                message="AutoBlitz 모니터링 시스템이 시작되었습니다.",
                severity=AlertSeverity.INFO
            )
            
            self._initialized = True
            logger.info("모니터링 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"모니터링 시스템 초기화 실패: {e}")
            raise
    
    async def shutdown(self):
        """모니터링 시스템 종료"""
        if not self._initialized:
            return
        
        try:
            # 시스템 종료 알림
            await trigger_custom_alert(
                name="system_shutdown",
                message="AutoBlitz 모니터링 시스템이 종료됩니다.",
                severity=AlertSeverity.INFO
            )
            
            # 메트릭 수집기 중지
            await self.metrics.stop()
            logger.info("메트릭 수집기 중지 완료")
            
            # 알림 시스템 중지
            await self.alerts.stop()
            logger.info("알림 시스템 중지 완료")
            
            self._initialized = False
            logger.info("모니터링 시스템 종료 완료")
            
        except Exception as e:
            logger.error(f"모니터링 시스템 종료 실패: {e}")
    
    async def health_check(self) -> dict:
        """모니터링 시스템 상태 확인"""
        status = {
            "initialized": self._initialized,
            "cloudwatch_available": self.cloudwatch.cloudwatch is not None,
            "metrics_running": self.metrics._running,
            "alerts_running": self.alerts._running,
            "active_alerts": len(self.alerts.get_active_alerts()),
            "alert_rules": len(self.alerts.rules)
        }
        
        # 최근 시스템 메트릭 수집
        if self._initialized:
            await record_system_status()
        
        return status
    
    def get_dashboard_data(self) -> dict:
        """대시보드용 데이터 조회"""
        return {
            "system_status": self._initialized,
            "active_alerts": self.alerts.get_active_alerts(),
            "alert_stats": self.alerts.get_alert_stats(),
            "recent_alerts": self.alerts.get_alert_history(limit=20)
        }

# 글로벌 모니터링 시스템 인스턴스
monitoring_system = MonitoringSystem()

# 편의 함수들
async def init_monitoring():
    """모니터링 시스템 초기화"""
    await monitoring_system.initialize()

async def shutdown_monitoring():
    """모니터링 시스템 종료"""
    await monitoring_system.shutdown()

async def get_monitoring_status():
    """모니터링 상태 조회"""
    return await monitoring_system.health_check()

# 자주 사용되는 함수들을 패키지 레벨에서 노출
__all__ = [
    # CloudWatch
    'cloudwatch_client',
    'send_metric',
    'send_metrics',
    'MetricData',
    'MetricUnit',
    
    # 메트릭 수집
    'metrics_collector',
    'measure_api_call',
    'record_api_performance',
    'record_bot_status',
    'record_system_status',
    
    # 알림
    'alert_manager',
    'trigger_custom_alert',
    'check_metric_threshold',
    'AlertSeverity',
    
    # 통합 시스템
    'monitoring_system',
    'init_monitoring',
    'shutdown_monitoring',
    'get_monitoring_status'
]