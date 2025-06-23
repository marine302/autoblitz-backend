import asyncio
from app.strategies.utils.okx_client import RealTimeMarketScanner

async def debug_data():
    print('🔍 실제 데이터 구조 확인!')
    scanner = RealTimeMarketScanner()
    
    try:
        await scanner.initialize()
        
        # 첫 5개 코인만 상세 분석
        symbols = scanner.symbols[:5]
        print(f'📊 상위 5개 코인 상세 분석: {symbols}')
        
        tickers = await scanner.okx_client.get_multiple_tickers(symbols)
        
        print(f'\n📈 실제 티커 데이터:')
        for i, ticker in enumerate(tickers[:3], 1):
            print(f'{i}. {ticker["symbol"]}:')
            print(f'   가격: ${ticker["price"]:,.2f}')
            print(f'   24h 변동: {ticker["change_24h"]:+.2f}%')
            print(f'   24h 거래량: {ticker["volume_24h"]:,.0f}')
            
            # 조건 확인
            conditions = []
            if abs(ticker['change_24h']) > 1.0:
                conditions.append('1%이상변동')
            if ticker['volume_24h'] > 1000:
                conditions.append('거래량OK')
            
            print(f'   충족조건: {conditions}')
            print(f'   신호가능: {len(conditions) >= 2}')
            print()
        
        # K라인 데이터도 확인
        if tickers:
            symbol = tickers[0]['symbol']
            klines = await scanner.okx_client.get_klines(symbol, '1m', 5)
            print(f'📊 {symbol} K라인 데이터 ({len(klines)}개):')
            for i, kline in enumerate(klines[:3], 1):
                print(f'   {i}. 종가: ${kline["close"]:,.2f}, 거래량: {kline["volume"]:,.0f}')
    
    except Exception as e:
        print(f'❌ 에러: {e}')
        import traceback
        traceback.print_exc()
    
    finally:
        await scanner.cleanup()

if __name__ == "__main__":
    asyncio.run(debug_data())
