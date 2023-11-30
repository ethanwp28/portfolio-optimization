import pandas as pd
import numpy as np
from pypfopt import BlackLittermanModel, risk_models, expected_returns, EfficientFrontier
import yfinance as yf

tickers = ["AAPL","MSFT", "GE", "LMT", "PFE", "NVO", "AMZN", "TSLA", "WFC", "BLK"]

data = yf.download(tickers, start="2018-01-01", end="2022-12-31")
prices = data['Adj Close']

mean_historical_returns = expected_returns.mean_historical_return(prices)

S = risk_models.sample_cov(prices)

views = pd.Series([0.12], index=["AAPL"])  # Example view

bl = BlackLittermanModel(S, pi=mean_historical_returns, absolute_views=views)

posterior_expected_returns = bl.bl_returns()
print(posterior_expected_returns)

ef = EfficientFrontier(posterior_expected_returns, S)

ef.add_constraint(lambda w: w >= 0)

# Find the optimal portfolio
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)

