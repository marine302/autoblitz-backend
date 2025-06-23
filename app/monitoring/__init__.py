# 파일명: app/monitoring/__init__.py (개발용 더미)
"""개발 환경용 모니터링 더미"""

from app.core.monitoring import *

try:
    from app.core.monitoring import measure_api_call
except ImportError:
    def measure_api_call(*args, **kwargs):
        pass

try:
    from app.core.monitoring import record_api_performance
except ImportError:
    def record_api_performance(*args, **kwargs):
        pass

class MonitoringSystem:
    def __init__(self):
        pass
    
    async def initialize(self):
        print("📊 [DEV] 모니터링 시스템 더미 모드")
        pass
    
    async def shutdown(self):
        pass
    
    async def health_check(self):
        return {
            "status": "healthy",
            "cloudwatch_available": False,
            "environment": "development"
        }
    
    def get_dashboard_data(self):
        return {
            "cpu_usage": 5.0,
            "memory_usage": 20.0,
            "active_bots": 0,
            "total_trades": 0
        }

monitoring_system = MonitoringSystem()

async def init_monitoring():
    await monitoring_system.initialize()

async def shutdown_monitoring():
    await monitoring_system.shutdown()

async def get_monitoring_status():
    return await monitoring_system.health_check()

# 더미 함수들
def send_metric(*args, **kwargs):
    pass

def send_metrics(*args, **kwargs):
    pass

class MetricData:
    pass

class MetricUnit:
    pass

cloudwatch_client = None

__all__ = [
    'monitoring_system',
    'init_monitoring', 
    'shutdown_monitoring',
    'get_monitoring_status',
    'send_metric',
    'send_metrics',
    'MetricData',
    'MetricUnit',
    'cloudwatch_client',
    'measure_api_call',
    'record_api_performance'
]
