# 파일명: app/core/monitoring.py (개발용 더미)
"""개발 환경용 모니터링 더미"""

class SystemMonitor:
    def __init__(self):
        pass
    
    def get_system_metrics(self):
        return {'cpu_percent': 5.0, 'memory_percent': 20.0}
    
    def send_metrics_to_cloudwatch(self, metrics):
        return True

class PerformanceTracker:
    def __init__(self):
        self.metrics = {}
    
    def track_api_call(self, endpoint, duration, status_code):
        pass
    
    def get_performance_summary(self):
        return {'system': {'cpu_percent': 5.0}}

def get_performance_tracker():
    return PerformanceTracker()

performance_tracker = PerformanceTracker()
