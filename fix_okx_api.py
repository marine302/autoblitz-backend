import asyncio
import aiohttp
import json

async def find_working_endpoint():
    print('�� OKX API 올바른 엔드포인트 찾기!')
    
    # 시도할 여러 엔드포인트들
    endpoints = [
        "https://www.okx.com/api/v5/market/tickers",
        "https://www.okx.com/api/v5/public/tickers",
        "https://www.okx.com/api/v5/market/ticker",
        "https://aws.okx.com/api/v5/market/tickers",
        "https://okx.com/api/v5/market/tickers"
    ]
    
    params_list = [
        {'instType': 'SPOT'},
        {},
        {'instId': 'BTC-USDT'}
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, url in enumerate(endpoints, 1):
            print(f'\n{i}. 테스트: {url}')
            
            for j, params in enumerate(params_list, 1):
                try:
                    print(f'   {j}) 파라미터: {params}')
                    async with session.get(url, params=params) as response:
                        print(f'      상태: {response.status}')
                        
                        if response.status == 200:
                            data = await response.json()
                            if 'data' in data and data['data']:
                                print(f'      ✅ 성공! 데이터 {len(data["data"])}개')
                                
                                # 첫 번째 데이터 출력
                                first_item = data['data'][0]
                                print(f'      샘플: {first_item}')
                                
                                return url, params  # 성공한 엔드포인트 반환
                        else:
                            text = await response.text()
                            print(f'      에러: {text[:100]}')
                            
                except Exception as e:
                    print(f'      예외: {e}')
    
    print('\n❌ 모든 엔드포인트 실패!')
    return None, None

# 대안: 무료 API 테스트
async def try_alternative_apis():
    print('\n🔄 대안 API 테스트 - CoinGecko')
    
    async with aiohttp.ClientSession() as session:
        try:
            # CoinGecko API (무료, API 키 불필요)
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 10,
                'page': 1
            }
            
            async with session.get(url, params=params) as response:
                print(f'CoinGecko 상태: {response.status}')
                
                if response.status == 200:
                    data = await response.json()
                    print(f'✅ CoinGecko 성공! {len(data)}개 코인')
                    
                    for i, coin in enumerate(data[:3], 1):
                        print(f'{i}. {coin["symbol"].upper()}: ${coin["current_price"]:,.2f}, '
                              f'변동: {coin["price_change_percentage_24h"]:+.2f}%')
                    
                    return True
                    
        except Exception as e:
            print(f'CoinGecko 에러: {e}')
    
    return False

async def main():
    # OKX API 테스트
    working_url, working_params = await find_working_endpoint()
    
    if working_url:
        print(f'\n✅ 작동하는 OKX 엔드포인트 발견!')
        print(f'URL: {working_url}')
        print(f'파라미터: {working_params}')
    else:
        print(f'\n⚠️ OKX API 접근 불가, 대안 API 시도...')
        success = await try_alternative_apis()
        
        if success:
            print('\n💡 CoinGecko API를 사용하여 진행 가능!')

asyncio.run(main())
