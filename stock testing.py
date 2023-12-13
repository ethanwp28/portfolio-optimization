# testing data
# 'VONWIX9CETIDLBPD'

# import quandl
# quandl.ApiConfig.api_key = 'eVDLjMLSRfBvi3Zg8sYy'
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

def fetch_stock_data_av(tickers, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data = {}
    for ticker in tickers:
        # Fetch the daily data for each ticker
        ticker_data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
        data[ticker] = ticker_data['4. close']  # Use the closing price

    # Combine the data into a single DataFrame
    combined_data = pd.concat(data.values(), axis=1, keys=data.keys())
    return combined_data

# Example usage
api_key = 'VONWIX9CETIDLBPD'  # Replace with your actual API key
tickers = ['AAPL', 'MSFT', 'GOOG']
data = fetch_stock_data_av(tickers, api_key)
print(data)