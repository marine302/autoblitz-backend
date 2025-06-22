
📋 AutoBlitz 서버 프로비저닝 배포 가이드

1. 🔑 AWS 설정
   aws configure
   # Access Key, Secret Key, Region (ap-northeast-2) 설정

2. 🔐 키 페어 생성
   aws ec2 create-key-pair --key-name autoblitz-key --query 'KeyMaterial' --output text > autoblitz-key.pem
   chmod 400 autoblitz-key.pem

3. 🚀 서버 생성
   cd infrastructure/scripts
   python provision-server.py

4. 📊 서버 상태 확인
   # 스크립트 내 get_server_status() 메소드 사용

5. 💰 비용 모니터링
   # AWS 콘솔 > Billing > Cost Explorer에서 확인

⚠️ 주의사항:
   - 스팟 인스턴스는 AWS가 회수할 수 있음
   - 중요 데이터는 S3에 백업 필수
   - 월간 비용 알림 설정 권장
