# 파일: app/monitoring/cloudwatch.py
# 경로: /workspaces/autoblitz-backend/app/monitoring/cloudwatch.py
"""
AutoBlitz CloudWatch 모니터링 시스템
실시간 메트릭 수집 및 알림
"""

import boto3
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class MetricUnit(Enum):
    """CloudWatch 메트릭 단위"""
    COUNT = "Count"
    PERCENT = "Percent"
    SECONDS = "Seconds"
    MILLISECONDS = "Milliseconds"
    BYTES = "Bytes"
    KILOBYTES = "Kilobytes"
    MEGABYTES = "Megabytes"
    BITS_PER_SECOND = "Bits/Second"
    COUNT_PER_SECOND = "Count/Second"

@dataclass
class MetricData:
    """메트릭 데이터 구조"""
    metric_name: str
    value: Union[int, float]
    unit: MetricUnit
    timestamp: Optional[datetime] = None
    dimensions: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        
        if self.dimensions is None:
            self.dimensions = {}

class AutoBlitzCloudWatch:
    """AutoBlitz CloudWatch 클라이언트"""
    
    def __init__(self, namespace: str = "AutoBlitz/Production"):
        self.namespace = namespace
        self.region = getattr(settings, 'aws_region', 'ap-northeast-2')
        
        try:
            self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)
            self.logs = boto3.client('logs', region_name=self.region)
            logger.info(f"CloudWatch 클라이언트 초기화 완료: {self.namespace}")
        except Exception as e:
            logger.warning(f"CloudWatch 클라이언트 초기화 실패: {e}")
            self.cloudwatch = None
            self.logs = None
    
    async def put_metric(self, metric: MetricData) -> bool:
        """단일 메트릭 전송"""
        if not self.cloudwatch:
            logger.warning("CloudWatch 클라이언트 없음, 메트릭 무시")
            return False
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self._put_metric_sync,
                metric
            )
            
            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                logger.debug(f"메트릭 전송 성공: {metric.metric_name}")
                return True
            else:
                logger.error(f"메트릭 전송 실패: {response}")
                return False
                
        except Exception as e:
            logger.error(f"메트릭 전송 중 오류: {e}")
            return False
    
    def _put_metric_sync(self, metric: MetricData) -> Dict:
        """동기 메트릭 전송"""
        metric_data = {
            'MetricName': metric.metric_name,
            'Value': metric.value,
            'Unit': metric.unit.value,
            'Timestamp': metric.timestamp
        }
        
        if metric.dimensions:
            metric_data['Dimensions'] = [
                {'Name': key, 'Value': value}
                for key, value in metric.dimensions.items()
            ]
        
        return self.cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=[metric_data]
        )
    
    async def put_metrics_batch(self, metrics: List[MetricData]) -> bool:
        """배치 메트릭 전송 (최대 20개)"""
        if not self.cloudwatch:
            logger.warning("CloudWatch 클라이언트 없음, 배치 메트릭 무시")
            return False
        
        if not metrics:
            return True
        
        # CloudWatch 제한: 한번에 최대 20개
        batch_size = 20
        success_count = 0
        
        for i in range(0, len(metrics), batch_size):
            batch = metrics[i:i + batch_size]
            
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._put_metrics_batch_sync,
                    batch
                )
                
                if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                    success_count += len(batch)
                    logger.debug(f"배치 메트릭 전송 성공: {len(batch)}개")
                else:
                    logger.error(f"배치 메트릭 전송 실패: {response}")
                    
            except Exception as e:
                logger.error(f"배치 메트릭 전송 중 오류: {e}")
        
        return success_count == len(metrics)
    
    def _put_metrics_batch_sync(self, metrics: List[MetricData]) -> Dict:
        """동기 배치 메트릭 전송"""
        metric_data = []
        
        for metric in metrics:
            data = {
                'MetricName': metric.metric_name,
                'Value': metric.value,
                'Unit': metric.unit.value,
                'Timestamp': metric.timestamp
            }
            
            if metric.dimensions:
                data['Dimensions'] = [
                    {'Name': key, 'Value': value}
                    for key, value in metric.dimensions.items()
                ]
            
            metric_data.append(data)
        
        return self.cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=metric_data
        )
    
    async def create_alarm(
        self,
        alarm_name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "GreaterThanThreshold",
        evaluation_periods: int = 2,
        period: int = 300,  # 5분
        statistic: str = "Average",
        dimensions: Optional[Dict[str, str]] = None
    ) -> bool:
        """CloudWatch 알람 생성"""
        if not self.cloudwatch:
            logger.warning("CloudWatch 클라이언트 없음, 알람 생성 무시")
            return False
        
        try:
            alarm_dimensions = []
            if dimensions:
                alarm_dimensions = [
                    {'Name': key, 'Value': value}
                    for key, value in dimensions.items()
                ]
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.cloudwatch.put_metric_alarm,
                {
                    'AlarmName': alarm_name,
                    'ComparisonOperator': comparison,
                    'EvaluationPeriods': evaluation_periods,
                    'MetricName': metric_name,
                    'Namespace': self.namespace,
                    'Period': period,
                    'Statistic': statistic,
                    'Threshold': threshold,
                    'ActionsEnabled': True,
                    'AlarmDescription': f'AutoBlitz 자동 알람: {metric_name}',
                    'Dimensions': alarm_dimensions,
                    'Unit': MetricUnit.COUNT.value
                }
            )
            
            logger.info(f"알람 생성 완료: {alarm_name}")
            return True
            
        except Exception as e:
            logger.error(f"알람 생성 실패: {e}")
            return False
    
    async def get_metric_statistics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        period: int = 300,
        statistics: List[str] = None,
        dimensions: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """메트릭 통계 조회"""
        if not self.cloudwatch:
            logger.warning("CloudWatch 클라이언트 없음, 통계 조회 무시")
            return []
        
        if statistics is None:
            statistics = ['Average', 'Sum', 'Maximum', 'Minimum']
        
        try:
            metric_dimensions = []
            if dimensions:
                metric_dimensions = [
                    {'Name': key, 'Value': value}
                    for key, value in dimensions.items()
                ]
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.cloudwatch.get_metric_statistics,
                {
                    'Namespace': self.namespace,
                    'MetricName': metric_name,
                    'Dimensions': metric_dimensions,
                    'StartTime': start_time,
                    'EndTime': end_time,
                    'Period': period,
                    'Statistics': statistics
                }
            )
            
            return sorted(
                response['Datapoints'],
                key=lambda x: x['Timestamp']
            )
            
        except Exception as e:
            logger.error(f"메트릭 통계 조회 실패: {e}")
            return []
    
    async def send_log(
        self,
        log_group: str,
        log_stream: str,
        message: str,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """CloudWatch Logs 전송"""
        if not self.logs:
            logger.warning("CloudWatch Logs 클라이언트 없음")
            return False
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        try:
            # 로그 그룹 생성 (존재하지 않으면)
            await self._ensure_log_group_exists(log_group)
            
            # 로그 스트림 생성 (존재하지 않으면)
            await self._ensure_log_stream_exists(log_group, log_stream)
            
            # 로그 이벤트 전송
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.logs.put_log_events,
                {
                    'logGroupName': log_group,
                    'logStreamName': log_stream,
                    'logEvents': [
                        {
                            'timestamp': int(timestamp.timestamp() * 1000),
                            'message': message
                        }
                    ]
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"로그 전송 실패: {e}")
            return False
    
    async def _ensure_log_group_exists(self, log_group: str):
        """로그 그룹 존재 확인 및 생성"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.logs.describe_log_groups,
                {'logGroupNamePrefix': log_group}
            )
        except self.logs.exceptions.ResourceNotFoundException:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.logs.create_log_group,
                {'logGroupName': log_group}
            )
    
    async def _ensure_log_stream_exists(self, log_group: str, log_stream: str):
        """로그 스트림 존재 확인 및 생성"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.logs.describe_log_streams,
                {
                    'logGroupName': log_group,
                    'logStreamNamePrefix': log_stream
                }
            )
        except self.logs.exceptions.ResourceNotFoundException:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.logs.create_log_stream,
                {
                    'logGroupName': log_group,
                    'logStreamName': log_stream
                }
            )

# 글로벌 CloudWatch 인스턴스
cloudwatch_client = AutoBlitzCloudWatch()

# 편의 함수들
async def send_metric(
    name: str,
    value: Union[int, float],
    unit: MetricUnit = MetricUnit.COUNT,
    dimensions: Optional[Dict[str, str]] = None
) -> bool:
    """간편 메트릭 전송"""
    metric = MetricData(
        metric_name=name,
        value=value,
        unit=unit,
        dimensions=dimensions
    )
    return await cloudwatch_client.put_metric(metric)

async def send_metrics(metrics: List[MetricData]) -> bool:
    """간편 배치 메트릭 전송"""
    return await cloudwatch_client.put_metrics_batch(metrics)