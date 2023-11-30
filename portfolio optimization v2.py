import pandas as pd
import numpy as np
from pypfopt import BlackLittermanModel, risk_models, expected_returns, EfficientFrontier
import yfinance as yf

tickers = ["AAPL","MSFT", "GE", "LMT", "PFE", "NVO", "AMZN", "TSLA", "WFC", "BLK"]
# start_date = '2018-01-01'
# end_date = '2023-12-31'

# Load your data (as per previous examples)
# tickers = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")
prices = data['Adj Close']

# Calculate mean historical returns (as a proxy for the market prior)
mean_historical_returns = expected_returns.mean_historical_return(prices)

# Calculate the covariance matrix
S = risk_models.sample_cov(prices)

# Define your views on the assets
views = pd.Series([0.12], index=["AAPL"])  # Example view

# Create the Black-Litterman model using the mean historical returns as prior
bl = BlackLittermanModel(S, pi=mean_historical_returns, absolute_views=views)

# Get the posterior expected returns
posterior_expected_returns = bl.bl_returns()
print(posterior_expected_returns)

# Optimize the portfolio
ef = EfficientFrontier(posterior_expected_returns, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)

# Performance metrics
ef.portfolio_performance(verbose=True)
