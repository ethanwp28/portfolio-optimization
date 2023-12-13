# CAPM Portfolio Optimization
# Part 1 
# Importing libraries
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import statsmodels.api as sm
from statsmodels import regression

# Creating a list of Stock Tickers
stocks = ['AMZN','SPY']
pf_data = pd.DataFrame()
# Pulling closing price   
for stock in stocks:
    pf_data[stock] = wb.DataReader(stock, data_source = 'yahoo', start = '2008-1-1')['Adj Close']
num_stocks = len(stocks)

# Compute log returns
sec_returns = np.log(pf_data / pf_data.shift(1))

# Covariance matrix
cov = sec_returns.cov() * 252
cov_with_market = cov.iloc[0,1]

# Calculating Beta
AMZN_beta = cov_with_market / market_var
round(AMZN_beta,2)

# Calculate the expected return
AMZN_er = 0.0121 + AMZN_beta * 0.05
round(AMZN_er,2)

# Part 2 - For a Portfolio
# Creating a list of Stock Tickers

stocks = ['HSBC','JPM','TSLA','WMT','AMZN','COST','SPY']
pf_data = pd.DataFrame()
# Pulling closing price   
for stock in stocks:
    pf_data[stock] = wb.DataReader(stock, data_source = 'yahoo', start = '2015-1-1')['Adj Close']
num_stocks = len(stocks)

# Daily percentage change
pf_data_returns = pf_data.pct_change(1)
pf_data_returns = pf_data_returns[1:]

# Create 2 empty dictionaries
beta = {}
alpha = {}

for i in pf_data_returns.columns:
    if i != 'SPY':
        pf_data_returns.plot(kind = 'scatter', x = 'SPY', y = i)
        b,a = np.polyfit(pf_data_returns['SPY'], pf_data_returns[i],    1)
        plt.plot(pf_data_returns['SPY'], b * pf_data_returns['SPY'] + a, '-', color = 'r')  
        beta[i] = b    
        alpha[i] = a
        plt.show()

keys = list(beta.keys())

ER = {}
rf = 0 
rm = pf_data_returns['SPY'].mean() * 252

for i in keys:
    ER[i] = rf + (beta[i] * (rm-rf))
    print('Expected Return based on CAPM for {} is {}%'.format(i,round(ER[i]*100,2)))

portfolio_weights = 1/6 * np.ones(6) 
ER_portfolio = sum(list(ER.values()) * portfolio_weights)
print('Expected Return based on CAPM for the portfolio is {}%\n'.format(round(ER_portfolio*100,2)))

