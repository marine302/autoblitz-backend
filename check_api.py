import asyncio
import aiohttp
import json

async def check_okx_api():
    print('🔍 OKX API 직접 테스트!')
    
    async with aiohttp.ClientSession() as session:
        try:
            # 1. 기본 연결 테스트
            url = "https://www.okx.com/api/v5/public/time"
            print(f'📡 서버 시간 조회: {url}')
            
            async with session.get(url) as response:
                print(f'   상태 코드: {response.status}')
                if response.status == 200:
                    data = await response.json()
                    print(f'   응답: {data}')
                else:
                    text = await response.text()
                    print(f'   에러 응답: {text[:200]}')
            
            print()
            
            # 2. 티커 정보 조회 테스트
            url2 = "https://www.okx.com/api/v5/public/tickers"
            params = {'instType': 'SPOT'}
            print(f'📊 티커 정보 조회: {url2}')
            print(f'   파라미터: {params}')
            
            async with session.get(url2, params=params) as response:
                print(f'   상태 코드: {response.status}')
                if response.status == 200:
                    data = await response.json()
                    print(f'   데이터 개수: {len(data.get("data", []))}개')
                    
                    # 처음 3개만 출력
                    tickers = data.get('data', [])[:3]
                    for i, ticker in enumerate(tickers, 1):
                        print(f'   {i}. {ticker["instId"]}: ${ticker["last"]}, 변동: {ticker["chg"]}%')
                        
                else:
                    text = await response.text()
                    print(f'   에러 응답: {text[:200]}')
                    
        except Exception as e:
            print(f'❌ 연결 에러: {e}')
            import traceback
            traceback.print_exc()

asyncio.run(check_okx_api())
