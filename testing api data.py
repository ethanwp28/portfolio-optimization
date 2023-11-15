from alpha_vantage.timeseries import TimeSeries
import pandas as pd

def get_market_data(symbol, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol, interval='1min', outputsize='compact')
    return data.head()  

# Usage
api_key = '16SIXTINQJAUF0MZ'  
symbol = 'VTHR' 
market_data = get_market_data(symbol, api_key)
print(market_data)
