# Porftolio Optimization Methods Comparison

# Data Gathering: Portfolio Setup

from pandas_datareader import data as web
from datetime import datetime

# Stock Ticker List for the portfolio
symbols = ['AAPL','GOOG','AMZN','FB']

start_date = datetime(2015,1,1)
end_date = datetime.now()
SOURCE = 'yahoo'

data = web.DataReader(name=symbols,
                    data_source=SOURCE,
                    start=start_date,
                    end=end_date)

# Define calc_returns() function to calculate returns for daily prices of the individual stocks in the portfolio, over a defined period.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def calc_returns(price_data, resample=None, ret_type="arithmatic"):
    """
    Parameters
        price_data: price timeseries pd.DataFrame object.
        resample:   DateOffset, Timedelta or str. `None` for not resampling. Default: None
                    More on Dateoffsets : https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        ret_type:   return calculation type. \"arithmatic\" or \"log\"

    Returns:
        returns timeseries pd.DataFrame object
    """
    if ret_type=="arithmatic":
        ret = price_data.pct_change().dropna()
    elif ret_type=="log":
        ret = np.log(price_data/price_data.shift()).dropna()
    else:
        raise ValueError("ret_type: return calculation type is not valid. use \"arithmatic\" or \"log\"")

    if resample != None:
        if ret_type=="arithmatic":
            ret = ret.resample(resample).apply(lambda df: (df+1).cumprod(axis=0).iloc[-1]) - 1
        elif ret_type=="log":
            ret = ret.resample(resample).apply(lambda df: df.sum(axis=0))
    return(ret)

# Define calc_returns_stats to track the expected returns of the stocks and the historical returns covariance matrix

def calc_returns_stats(returns):
    """
    Parameters
        returns: returns timeseries pd.DataFrame object

    Returns:
        mean_returns: Avereage of returns
        cov_matrix: returns Covariance matrix
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return(mean_returns, cov_matrix)

def portfolio(weights, mean_returns, cov_matrix):

    portfolio_return = np.dot(weights.reshape(1,-1), mean_returns.values.reshape(-1,1))
    portfolio_var = np.dot(np.dot(weights.reshape(1,-1), cov_matrix.values), weights.reshape(-1,1))
    portfolio_std = np.sqrt(portfolio_var)

    return(np.squeeze(portfolio_return),np.squeeze(portfolio_var),np.squeeze(portfolio_std))

# Monte Carlo Sim
num_iter = 500000

porfolio_var_list = []
porfolio_ret_list = []
w_list =[]

max_sharpe = 0
max_sharpe_var = None
max_sharpe_ret = None
max_sharpe_w = None

daily_ret = calc_returns(adj_close, resample=None, ret_type="log")
mean_returns, cov_matrix = calc_returns_stats(daily_ret)

for i in range(1,num_iter+1):
    rand_weights = np.random.random(len(symbols))
    rand_weights = rand_weights/np.sum(rand_weights)

    porfolio_ret, porfolio_var, portfolio_std = portfolio(rand_weights, mean_returns, cov_matrix)

    # Anuualizing
    porfolio_ret = porfolio_ret * 252
    porfolio_var = porfolio_var * 252
    portfolio_std = portfolio_std * (252**0.5)

    sharpe = (porfolio_ret/(porfolio_var**0.5)).item()
    if sharpe > max_sharpe:
        max_sharpe = sharpe
        max_sharpe_var = porfolio_var.item()
        max_sharpe_ret = porfolio_ret.item()
        max_sharpe_w = rand_weights

    porfolio_var_list.append(porfolio_var)
    porfolio_ret_list.append(porfolio_ret)
    w_list.append(rand_weights)
    if ((i/num_iter)*100)%10 == 0:
        print(f'%{round((i/num_iter)*100)}...',end='')

