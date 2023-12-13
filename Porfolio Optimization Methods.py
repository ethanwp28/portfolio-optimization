# Porfolio Optimization Methods Comparison

from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

# Get the stock data to build a portfolio
def fetch_stock_data(tickers, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data = {}
    for ticker in tickers:
        ticker_data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
        data[ticker] = ticker_data['4. close']
    combined_data = pd.concat(data.values(), axis=1, keys=data.keys())
    return combined_data

# Build the stock portfolio
def create_portfolio(tickers, weights, api_key, start_date, end_date):
    stock_data = fetch_stock_data(tickers, api_key)
    # Filter data for the specified date range if necessary
    stock_data = stock_data.loc[start_date:end_date]

    if sum(weights) != 1:
        raise ValueError("Sum of weights must be 1")

    # Create a DataFrame for portfolio
    portfolio = pd.DataFrame({'Weights': weights}, index=tickers)
    
    # Adjust the stock data to match the portfolio structure
    stock_data.columns = pd.MultiIndex.from_product([['Stock Prices'], stock_data.columns])
    
    # Combine portfolio weights with stock data
    portfolio_data = pd.concat([portfolio, stock_data], axis=1)

    # Additional calculations (if needed)
    # ...

    return portfolio_data


# VaR Optimization
def var_optimization(portfolio_data, weights, confidence_level=0.05, days=1):
    """
    Calculate the VaR (Value at Risk) of a portfolio.
    
    :param portfolio_data: DataFrame containing historical stock prices.
    :param weights: Array representing the weights of the stocks in the portfolio.
    :param confidence_level: Confidence level for VaR calculation (default is 5%).
    :param days: The time frame in days for VaR calculation (default is 1 day).
    :return: Value at Risk for the specified confidence level and time frame.
    """
    # Calculate daily returns
    daily_returns = portfolio_data.pct_change().dropna()

    # Calculate portfolio returns
    portfolio_returns = daily_returns.dot(weights)

    # Calculate VaR
    if days > 1:
        scaling_factor = np.sqrt(days)
    else:
        scaling_factor = 1

    VaR = -np.percentile(portfolio_returns, confidence_level * 100) * scaling_factor
    return VaR

# C-VaR Optimization
def cvar_optimization(portfolio_data, weights, confidence_level=0.05):
    """
    Calculate the CVaR (Conditional Value at Risk) of a portfolio.
    
    :param portfolio_data: DataFrame containing historical stock prices.
    :param weights: Array representing the weights of the stocks in the portfolio.
    :param confidence_level: Confidence level for CVaR calculation (default is 5%).
    :return: Conditional Value at Risk for the specified confidence level.
    """
    # Calculate daily returns
    daily_returns = portfolio_data.pct_change().dropna()

    # Calculate portfolio returns
    portfolio_returns = daily_returns.dot(weights)

    # Calculate VaR
    VaR = -np.percentile(portfolio_returns, confidence_level * 100)

    # Calculate CVaR
    CVaR = -portfolio_returns[portfolio_returns < -VaR].mean()
    return CVaR

# Mean-Variance Optimization Function
def mean_variance_optimization(portfolio_data, target_return=None):
    """
    Perform mean-variance optimization for the given portfolio data.

    :param portfolio_data: DataFrame containing historical stock prices.
    :param target_return: Target return for the portfolio. If None, optimizes for minimum variance.
    :return: Optimal weights for the portfolio.
    """
    # Calculate daily returns
    daily_returns = portfolio_data.pct_change().dropna()

    # Mean returns and covariance
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # Number of assets
    num_assets = len(mean_returns)

    # Objective function
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum of weights is 1

    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target_return})

    # Bounds for weights
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Initial guess
    initial_guess = num_assets * [1. / num_assets,]

    # Optimization
    result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

# CAPM Optimization Function
def capm_evaluation(portfolio_data, market_data, risk_free_rate):
    """
    Evaluate assets in a portfolio using the CAPM model.

    :param portfolio_data: DataFrame containing historical stock prices.
    :param market_data: DataFrame containing historical market index prices.
    :param risk_free_rate: The risk-free rate of return.
    :return: Expected returns of assets based on CAPM.
    """
    # Calculate daily returns
    stock_returns = portfolio_data.pct_change().dropna()
    market_returns = market_data.pct_change().dropna()

    # Aligning market and stock returns
    aligned_returns = pd.concat([stock_returns, market_returns], axis=1).dropna()

    # CAPM calculation for each stock
    capm_returns = {}
    for stock in stock_returns.columns:
        # Regression to find beta
        beta, alpha = np.polyfit(aligned_returns[market_data.name], aligned_returns[stock], 1)

        # CAPM formula: Expected Return = Risk-Free Rate + Beta * (Market Return - Risk-Free Rate)
        expected_return = risk_free_rate + beta * (aligned_returns[market_data.name].mean() * 252 - risk_free_rate)  # Annualizing the return
        capm_returns[stock] = expected_return

    return capm_returns

# Monte Carlo Optimization
def monte_carlo_optimization(portfolio_data, num_portfolios=10000, risk_free_rate=0.02):
    """
    Perform Monte Carlo simulation for portfolio optimization.

    :param portfolio_data: DataFrame containing historical stock prices.
    :param num_portfolios: Number of random portfolios to simulate.
    :param risk_free_rate: The risk-free rate of return.
    :return: DataFrame with simulated portfolio returns and risks.
    """
    # Calculate daily returns
    daily_returns = portfolio_data.pct_change().dropna()

    # Number of assets
    num_assets = len(portfolio_data.columns)

    # Initialize arrays to store returns, volatility, and weights
    portfolio_returns = []
    portfolio_volatility = []
    stock_weights = []

    # Simulate random portfolio weights num_portfolios times
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Normalize to sum to 1

        # Calculate portfolio return and volatility
        annualized_return = np.sum(daily_returns.mean() * weights) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))

        # Store results
        portfolio_returns.append(annualized_return)
        portfolio_volatility.append(volatility)
        stock_weights.append(weights)

    # Create a DataFrame for simulated portfolios
    portfolio_simulations = pd.DataFrame({
        'Returns': portfolio_returns,
        'Volatility': portfolio_volatility
    })

    for i, stock in enumerate(portfolio_data.columns):
        portfolio_simulations[f'Weight_{stock}'] = [weights[i] for weights in stock_weights]

    return portfolio_simulations

# Mean-Mean Absolute Deviation Model
def mean_mad_optimization(portfolio_data, target_return=None):
    """
    Perform Mean-Mean Absolute Deviation optimization on a portfolio.

    :param portfolio_data: DataFrame containing historical stock prices.
    :param target_return: Optional target return for the portfolio.
    :return: Optimal weights for the portfolio.
    """
    # Calculate daily returns
    daily_returns = portfolio_data.pct_change().dropna()

    # Calculate mean returns
    mean_returns = daily_returns.mean()

    # Number of assets
    num_assets = len(portfolio_data.columns)

    # Optimization variables
    weights = cp.Variable(num_assets)
    mad = cp.Variable()

    # Objective: Minimize MAD
    objective = cp.Minimize(mad)

    # Constraints
    constraints = [
        cp.sum(weights) == 1,  # Sum of weights is 1
        weights >= 0,          # No short selling
        mad >= cp.abs(daily_returns - mean_returns) @ weights
    ]

    if target_return is not None:
        constraints.append(mean_returns @ weights >= target_return)

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return weights.value

def main():
    # API Key for Alpha Vantage
    api_key = 'VONWIX9CETIDLBPD'
    # Define stock tickers and date range
    tickers = ['AAPL', 'MSFT', 'GOOG']
    weights = [0.4, 0.3, 0.3]
    start_date = '2023-09-01'
    end_date = '2023-10-01'
    risk_free_rate = 0.02  # 2% risk-free rate

    # Fetch stock data and create portfolio
    portfolio_data = create_portfolio(tickers, weights, api_key, start_date, end_date)

    # Debugging: Print portfolio data structure
    print("Portfolio Data Columns:", portfolio_data.columns)
    print("Number of weights:", len(weights))
    
    # Ensure the portfolio data columns match the tickers
    if len(portfolio_data.columns) != len(weights):
        print("Error: The number of stocks in the portfolio does not match the number of weights.")
        return

    # Perform portfolio optimizations

    # VaR
    print("Calculating Value at Risk...")
    var_result = var_optimization(portfolio_data, weights, confidence_level=0.05, days=1)
    print(f"1-day VaR at 95% confidence level: {var_result}")

    # C-VaR
    print("Calculating Conditional Value at Risk (CVaR)...")
    cvar_result = cvar_optimization(portfolio_data, weights, confidence_level=0.05)
    print(f"CVaR at 95% confidence level: {cvar_result}")

    # Mean-Variance
    print("Performing Mean-Variance Optimization...")
    mv_weights = mean_variance_optimization(portfolio_data, target_return=0.1)
    print("Optimal Weights (Mean-Variance):", mv_weights)

    # CAPM
    print("\nEvaluating with CAPM...")
    capm_returns = capm_evaluation(portfolio_data, risk_free_rate)
    print("Expected Returns (CAPM):", capm_returns)
    
    # Monte Carlo
    print("\nPerforming Monte Carlo Optimization...")
    mc_simulation = monte_carlo_optimization(portfolio_data, num_portfolios=10000)
    print(mc_simulation.head())  # Display first few rows of the simulation results

    # Mean-MAD
    print("\nPerforming Mean-MAD Optimization...")
    mad_weights = mean_mad_optimization(portfolio_data, target_return=0.1)
    print("Optimal Weights (Mean-MAD):", mad_weights)


    # Comparison and Analysis...
    # Visualization...

if __name__ == "__main__":
    main()

