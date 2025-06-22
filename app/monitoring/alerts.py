# 파일: app/monitoring/alerts.py
# 경로: /workspaces/autoblitz-backend/app/monitoring/alerts.py
"""
AutoBlitz 실시간 알림 시스템
임계값 모니터링 및 자동 알림 발송
"""

import asyncio
import smtplib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

from .cloudwatch import cloudwatch_client, send_metric, MetricUnit

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """알림 심각도"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """알림 채널"""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"

@dataclass
class AlertRule:
    """알림 규칙"""
    name: str
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    duration: int  # 지속시간 (초)
    severity: AlertSeverity
    channels: List[AlertChannel]
    message_template: str
    enabled: bool = True
    cooldown: int = 300  # 재알림 방지 (초)

@dataclass
class Alert:
    """알림 데이터"""
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    message: str
    timestamp: datetime
    dimensions: Optional[Dict[str, str]] = None

class AlertManager:
    """알림 관리자"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_time: Dict[str, datetime] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # 기본 알림 규칙 등록
        self._register_default_rules()
    
    def _register_default_rules(self):
        """기본 알림 규칙 등록"""
        default_rules = [
            # API 응답시간 알림
            AlertRule(
                name="api_response_time_high",
                metric_name="API_ResponseTime",
                threshold=1000.0,  # 1초
                comparison="gt",
                duration=60,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL],
                message_template="API 응답시간이 {current_value:.2f}ms로 임계값 {threshold}ms를 초과했습니다."
            ),
            
            # API 에러율 알림
            AlertRule(
                name="api_error_rate_high",
                metric_name="API_ErrorCount",
                threshold=10.0,
                comparison="gt",
                duration=300,
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.EMAIL],
                message_template="API 에러가 5분간 {current_value}회 발생했습니다."
            ),
            
            # 시스템 CPU 사용률 알림
            AlertRule(
                name="system_cpu_high",
                metric_name="System_CPUUsage",
                threshold=80.0,
                comparison="gt",
                duration=300,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL],
                message_template="시스템 CPU 사용률이 {current_value:.1f}%로 높습니다."
            ),
            
            # 시스템 메모리 사용률 알림
            AlertRule(
                name="system_memory_high",
                metric_name="System_MemoryUsage",
                threshold=85.0,
                comparison="gt",
                duration=300,
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.EMAIL],
                message_template="시스템 메모리 사용률이 {current_value:.1f}%로 위험 수준입니다."
            ),
            
            # 봇 성공률 낮음 알림
            AlertRule(
                name="bot_success_rate_low",
                metric_name="Bot_SuccessRate",
                threshold=70.0,
                comparison="lt",
                duration=600,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL],
                message_template="봇 성공률이 {current_value:.1f}%로 낮습니다."
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: AlertRule):
        """알림 규칙 추가"""
        self.rules[rule.name] = rule
        logger.info(f"알림 규칙 추가: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """알림 규칙 제거"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"알림 규칙 제거: {rule_name}")
    
    def enable_rule(self, rule_name: str):
        """알림 규칙 활성화"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            logger.info(f"알림 규칙 활성화: {rule_name}")
    
    def disable_rule(self, rule_name: str):
        """알림 규칙 비활성화"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            logger.info(f"알림 규칙 비활성화: {rule_name}")
    
    async def check_metric(
        self,
        metric_name: str,
        value: float,
        dimensions: Optional[Dict[str, str]] = None
    ):
        """메트릭 값 확인 및 알림 판정"""
        for rule_name, rule in self.rules.items():
            if not rule.enabled or rule.metric_name != metric_name:
                continue
            
            if self._should_trigger_alert(rule, value):
                await self._trigger_alert(rule, value, dimensions)
            elif rule_name in self.active_alerts:
                await self._resolve_alert(rule_name)
    
    def _should_trigger_alert(self, rule: AlertRule, value: float) -> bool:
        """알림 발생 조건 확인"""
        if rule.comparison == "gt":
            return value > rule.threshold
        elif rule.comparison == "lt":
            return value < rule.threshold
        elif rule.comparison == "eq":
            return value == rule.threshold
        elif rule.comparison == "gte":
            return value >= rule.threshold
        elif rule.comparison == "lte":
            return value <= rule.threshold
        return False
    
    async def _trigger_alert(
        self,
        rule: AlertRule,
        value: float,
        dimensions: Optional[Dict[str, str]] = None
    ):
        """알림 발생"""
        # 쿨다운 확인
        if self._is_in_cooldown(rule.name):
            return
        
        # 알림 생성
        alert = Alert(
            rule_name=rule.name,
            metric_name=rule.metric_name,
            current_value=value,
            threshold=rule.threshold,
            severity=rule.severity,
            message=rule.message_template.format(
                current_value=value,
                threshold=rule.threshold
            ),
            timestamp=datetime.now(timezone.utc),
            dimensions=dimensions
        )
        
        # 활성 알림 등록
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        self.last_alert_time[rule.name] = alert.timestamp
        
        # 알림 전송
        await self._send_alert(alert, rule.channels)
        
        # 알림 메트릭 기록
        await send_metric(
            name="Alert_Triggered",
            value=1,
            unit=MetricUnit.COUNT,
            dimensions={
                "RuleName": rule.name,
                "Severity": rule.severity.value,
                "MetricName": rule.metric_name
            }
        )
        
        logger.warning(f"알림 발생: {rule.name} - {alert.message}")
    
    async def _resolve_alert(self, rule_name: str):
        """알림 해결"""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            del self.active_alerts[rule_name]
            
            # 해결 알림 전송
            resolve_message = f"알림 해결: {alert.message}"
            logger.info(f"알림 해결: {rule_name}")
            
            # 해결 메트릭 기록
            await send_metric(
                name="Alert_Resolved",
                value=1,
                unit=MetricUnit.COUNT,
                dimensions={
                    "RuleName": rule_name,
                    "Severity": alert.severity.value
                }
            )
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """쿨다운 시간 확인"""
        if rule_name not in self.last_alert_time:
            return False
        
        rule = self.rules[rule_name]
        last_time = self.last_alert_time[rule_name]
        cooldown_end = last_time + timedelta(seconds=rule.cooldown)
        
        return datetime.now(timezone.utc) < cooldown_end
    
    async def _send_alert(self, alert: Alert, channels: List[AlertChannel]):
        """알림 전송"""
        for channel in channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert)
                # SMS는 향후 구현
                
            except Exception as e:
                logger.error(f"알림 전송 실패 ({channel.value}): {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """이메일 알림 전송"""
        # 실제 환경에서는 SMTP 설정 필요
        logger.info(f"이메일 알림: {alert.message}")
        # 개발 환경에서는 로그만 출력
    
    async def _send_slack_alert(self, alert: Alert):
        """Slack 알림 전송"""
        # 실제 환경에서는 Slack API 연동 필요
        logger.info(f"Slack 알림: {alert.message}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """웹훅 알림 전송"""
        # 실제 환경에서는 HTTP 웹훅 호출 필요
        logger.info(f"웹훅 알림: {alert.message}")
    
    async def start(self):
        """알림 시스템 시작"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("알림 시스템 시작")
    
    async def stop(self):
        """알림 시스템 중지"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("알림 시스템 중지")
    
    async def _monitor_loop(self):
        """모니터링 루프"""
        while self._running:
            try:
                # 주기적으로 CloudWatch에서 메트릭 조회 및 확인
                await self._check_cloudwatch_metrics()
                await asyncio.sleep(60)  # 1분마다 확인
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
    
    async def _check_cloudwatch_metrics(self):
        """CloudWatch 메트릭 확인"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=5)
            
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                # 최근 5분간 메트릭 조회
                datapoints = await cloudwatch_client.get_metric_statistics(
                    metric_name=rule.metric_name,
                    start_time=start_time,
                    end_time=end_time,
                    period=300,
                    statistics=['Average', 'Maximum']
                )
                
                if datapoints:
                    # 최신 데이터포인트 확인
                    latest = datapoints[-1]
                    avg_value = latest.get('Average', 0)
                    max_value = latest.get('Maximum', 0)
                    
                    # 임계값 확인 (최대값 기준)
                    await self.check_metric(rule.metric_name, max_value)
                    
        except Exception as e:
            logger.error(f"CloudWatch 메트릭 확인 실패: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 목록 조회"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """알림 히스토리 조회"""
        return self.alert_history[-limit:]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """알림 통계"""
        total_alerts = len(self.alert_history)
        active_count = len(self.active_alerts)
        
        # 심각도별 통계
        severity_stats = {}
        for severity in AlertSeverity:
            count = sum(1 for alert in self.alert_history 
                       if alert.severity == severity)
            severity_stats[severity.value] = count
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_count,
            "severity_stats": severity_stats,
            "rules_count": len(self.rules),
            "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled)
        }

# 글로벌 알림 매니저
alert_manager = AlertManager()

# 편의 함수들
async def trigger_custom_alert(
    name: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    channels: List[AlertChannel] = None
):
    """커스텀 알림 발생"""
    if channels is None:
        channels = [AlertChannel.EMAIL]
    
    alert = Alert(
        rule_name=f"custom_{name}",
        metric_name="Custom",
        current_value=0,
        threshold=0,
        severity=severity,
        message=message,
        timestamp=datetime.now(timezone.utc)
    )
    
    await alert_manager._send_alert(alert, channels)

async def check_metric_threshold(
    metric_name: str,
    value: float,
    dimensions: Optional[Dict[str, str]] = None
):
    """메트릭 임계값 확인"""
    await alert_manager.check_metric(metric_name, value, dimensions)