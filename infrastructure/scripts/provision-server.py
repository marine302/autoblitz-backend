# 파일: infrastructure/scripts/provision-server.py
# 경로: /workspaces/autoblitz-backend/infrastructure/scripts/provision-server.py
"""
AutoBlitz 사용자 서버 프로비저닝 자동화 스크립트
t4g.nano 스팟 인스턴스 생성 및 관리
"""

import boto3
import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from botocore.exceptions import ClientError

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    """서버 설정 정보"""
    username: str
    key_pair_name: str = "autoblitz-key"
    spot_price: str = "0.005"  # $0.005/hour
    region: str = "ap-northeast-2"  # Seoul
    instance_type: str = "t4g.nano"

class AutoBlitzServerProvisioner:
    """AutoBlitz 사용자 서버 프로비저닝 클래스"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.session = boto3.Session(region_name=config.region)
        self.cloudformation = self.session.client('cloudformation')
        self.ec2 = self.session.client('ec2')
        
    def create_user_server(self) -> Dict:
        """사용자 서버 생성"""
        stack_name = f"autoblitz-server-{self.config.username}"
        
        try:
            logger.info(f"사용자 서버 생성 시작: {self.config.username}")
            
            # CloudFormation 템플릿 파라미터
            parameters = [
                {
                    'ParameterKey': 'UserName',
                    'ParameterValue': self.config.username
                },
                {
                    'ParameterKey': 'KeyPairName', 
                    'ParameterValue': self.config.key_pair_name
                },
                {
                    'ParameterKey': 'SpotPrice',
                    'ParameterValue': self.config.spot_price
                }
            ]
            
            # CloudFormation 템플릿 읽기
            template_path = "infrastructure/cloudformation/user-server.yaml"
            with open(template_path, 'r') as f:
                template_body = f.read()
            
            # 스택 생성
            response = self.cloudformation.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=['CAPABILITY_NAMED_IAM'],
                Tags=[
                    {'Key': 'Project', 'Value': 'AutoBlitz'},
                    {'Key': 'User', 'Value': self.config.username},
                    {'Key': 'Type', 'Value': 'UserServer'}
                ]
            )
            
            logger.info(f"CloudFormation 스택 생성 요청 완료: {response['StackId']}")
            
            # 스택 생성 완료 대기
            return self.wait_for_stack_creation(stack_name)
            
        except ClientError as e:
            logger.error(f"서버 생성 실패: {e}")
            raise
    
    def wait_for_stack_creation(self, stack_name: str) -> Dict:
        """스택 생성 완료 대기"""
        logger.info(f"스택 생성 완료 대기 중: {stack_name}")
        
        waiter = self.cloudformation.get_waiter('stack_create_complete')
        
        try:
            waiter.wait(
                StackName=stack_name,
                WaiterConfig={
                    'Delay': 30,  # 30초마다 확인
                    'MaxAttempts': 20  # 최대 10분 대기
                }
            )
            
            # 스택 출력 정보 가져오기
            response = self.cloudformation.describe_stacks(StackName=stack_name)
            stack = response['Stacks'][0]
            
            if stack['StackStatus'] == 'CREATE_COMPLETE':
                logger.info(f"스택 생성 완료: {stack_name}")
                return self.parse_stack_outputs(stack.get('Outputs', []))
            else:
                raise Exception(f"스택 생성 실패: {stack['StackStatus']}")
                
        except Exception as e:
            logger.error(f"스택 생성 대기 중 오류: {e}")
            raise
    
    def parse_stack_outputs(self, outputs: List[Dict]) -> Dict:
        """스택 출력 정보 파싱"""
        result = {}
        for output in outputs:
            result[output['OutputKey']] = output['OutputValue']
        
        logger.info("서버 프로비저닝 완료:")
        for key, value in result.items():
            logger.info(f"  {key}: {value}")
        
        return result
    
    def attach_eip_to_instance(self, eip_allocation_id: str) -> bool:
        """EIP를 스팟 인스턴스에 연결"""
        try:
            # 스팟 플릿에서 인스턴스 ID 찾기
            spot_fleet_id = self.get_spot_fleet_id()
            if not spot_fleet_id:
                logger.error("스팟 플릿 ID를 찾을 수 없습니다")
                return False
            
            # 스팟 플릿의 인스턴스 조회
            response = self.ec2.describe_spot_fleet_instances(
                SpotFleetRequestId=spot_fleet_id
            )
            
            if not response['ActiveInstances']:
                logger.error("활성 인스턴스를 찾을 수 없습니다")
                return False
            
            instance_id = response['ActiveInstances'][0]['InstanceId']
            
            # EIP 연결
            self.ec2.associate_address(
                InstanceId=instance_id,
                AllocationId=eip_allocation_id
            )
            
            logger.info(f"EIP 연결 완료: {instance_id}")
            return True
            
        except ClientError as e:
            logger.error(f"EIP 연결 실패: {e}")
            return False
    
    def get_spot_fleet_id(self) -> Optional[str]:
        """스팟 플릿 ID 조회"""
        try:
            stack_name = f"autoblitz-server-{self.config.username}"
            response = self.cloudformation.describe_stacks(StackName=stack_name)
            
            for output in response['Stacks'][0].get('Outputs', []):
                if output['OutputKey'] == 'SpotFleetId':
                    return output['OutputValue']
            
            return None
            
        except Exception as e:
            logger.error(f"스팟 플릿 ID 조회 실패: {e}")
            return None
    
    def get_server_status(self) -> Dict:
        """서버 상태 조회"""
        try:
            stack_name = f"autoblitz-server-{self.config.username}"
            
            # 스택 상태 확인
            response = self.cloudformation.describe_stacks(StackName=stack_name)
            stack_status = response['Stacks'][0]['StackStatus']
            
            result = {
                'stack_status': stack_status,
                'username': self.config.username,
                'region': self.config.region
            }
            
            if stack_status == 'CREATE_COMPLETE':
                outputs = self.parse_stack_outputs(
                    response['Stacks'][0].get('Outputs', [])
                )
                result.update(outputs)
                
                # 인스턴스 상태 확인
                spot_fleet_id = outputs.get('SpotFleetId')
                if spot_fleet_id:
                    instance_status = self.get_instance_status(spot_fleet_id)
                    result.update(instance_status)
            
            return result
            
        except ClientError as e:
            logger.error(f"서버 상태 조회 실패: {e}")
            return {'error': str(e)}
    
    def get_instance_status(self, spot_fleet_id: str) -> Dict:
        """인스턴스 상태 조회"""
        try:
            response = self.ec2.describe_spot_fleet_instances(
                SpotFleetRequestId=spot_fleet_id
            )
            
            if response['ActiveInstances']:
                instance_id = response['ActiveInstances'][0]['InstanceId']
                
                # 인스턴스 세부 정보 조회
                instance_response = self.ec2.describe_instances(
                    InstanceIds=[instance_id]
                )
                
                instance = instance_response['Reservations'][0]['Instances'][0]
                
                return {
                    'instance_id': instance_id,
                    'instance_state': instance['State']['Name'],
                    'public_ip': instance.get('PublicIpAddress', 'N/A'),
                    'private_ip': instance.get('PrivateIpAddress', 'N/A'),
                    'launch_time': instance['LaunchTime'].isoformat()
                }
            else:
                return {'instance_status': 'No active instances'}
                
        except Exception as e:
            logger.error(f"인스턴스 상태 조회 실패: {e}")
            return {'instance_error': str(e)}
    
    def delete_user_server(self) -> bool:
        """사용자 서버 삭제"""
        stack_name = f"autoblitz-server-{self.config.username}"
        
        try:
            logger.info(f"사용자 서버 삭제 시작: {self.config.username}")
            
            self.cloudformation.delete_stack(StackName=stack_name)
            
            # 삭제 완료 대기
            waiter = self.cloudformation.get_waiter('stack_delete_complete')
            waiter.wait(
                StackName=stack_name,
                WaiterConfig={
                    'Delay': 30,
                    'MaxAttempts': 20
                }
            )
            
            logger.info(f"서버 삭제 완료: {self.config.username}")
            return True
            
        except ClientError as e:
            logger.error(f"서버 삭제 실패: {e}")
            return False

def calculate_monthly_cost(instance_type: str = "t4g.nano", hours_per_month: int = 744) -> float:
    """월간 비용 계산"""
    # t4g.nano 스팟 가격: $0.005/hour
    spot_prices = {
        "t4g.nano": 0.005,
        "t4g.micro": 0.0084,
        "t4g.small": 0.0168
    }
    
    hourly_rate = spot_prices.get(instance_type, 0.005)
    monthly_cost = hourly_rate * hours_per_month
    
    return round(monthly_cost, 2)

def main():
    """메인 함수 - 예제 사용법"""
    # 서버 설정
    config = ServerConfig(
        username="testuser01",
        key_pair_name="autoblitz-key",
        spot_price="0.005"
    )
    
    # 프로비저너 생성
    provisioner = AutoBlitzServerProvisioner(config)
    
    # 예제: 서버 생성
    try:
        result = provisioner.create_user_server()
        print(f"서버 생성 결과: {json.dumps(result, indent=2)}")
        
        # 월간 예상 비용 출력
        cost = calculate_monthly_cost()
        print(f"예상 월간 비용: ${cost}")
        
    except Exception as e:
        logger.error(f"서버 생성 중 오류: {e}")

if __name__ == "__main__":
    main()