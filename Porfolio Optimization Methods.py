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

# Get the market data
def fetch_market_data(market_ticker, api_key):
    """
    Fetch historical market data from Alpha Vantage.

    :param market_ticker: The ticker symbol for the market index (e.g., '^GSPC' for S&P 500).
    :param api_key: Your Alpha Vantage API key.
    :return: DataFrame containing the historical market data.
    """
    ts = TimeSeries(key=api_key, output_format='pandas')
    
    # Get daily adjusted market data
    market_data, _ = ts.get_daily_adjusted(symbol=market_ticker, outputsize='full')

    # Adjusted close price
    return market_data['5. adjusted close']

# Build the stock portfolio
def create_portfolio(tickers, weights, api_key, start_date, end_date):
    stock_data = fetch_stock_data(tickers, api_key)

    # Sort index
    stock_data.index = pd.to_datetime(stock_data.index)
    stock_data.sort_index(inplace=True)

    # Reindex based on date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days
    stock_data = stock_data.reindex(date_range, method='ffill')  # Forward fill missing data

    return stock_data

def portfolio_volatility(weights, cov_matrix):
    """
    Calculate the portfolio volatility (standard deviation) based on given weights and covariance matrix.

    :param weights: Portfolio weights.
    :param cov_matrix: Covariance matrix of asset returns.
    :return: Portfolio volatility.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

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

    # Calculate VaR - remove?
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


# Markowitz Portfolio Optimization
# Similar to mean-variance optimization, with some modifications

def markowitz_optimization(portfolio_data, target_return=None, risk_free_rate=0.0):
    """
    Perform mean-variance optimization for the given portfolio data, with optional Sharpe ratio maximization.

    :param portfolio_data: DataFrame containing historical stock prices.
    :param target_return: Target return for the portfolio. If None, optimizes for minimum variance.
    :param risk_free_rate: The risk-free rate used in Sharpe ratio calculation.
    :return: Dictionary containing optimal weights and additional optimization info.
    """
    # Data validation
    if not isinstance(portfolio_data, pd.DataFrame) or portfolio_data.empty:
        raise ValueError("portfolio_data must be a non-empty DataFrame.")

    # Calculate daily returns
    daily_returns = portfolio_data.pct_change().dropna()

    # Ensure sufficient data
    if len(daily_returns) < len(portfolio_data.columns):
        raise ValueError("Insufficient data for meaningful optimization.")

    # Mean returns and covariance
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # Number of assets
    num_assets = len(mean_returns)

    # Objective function for Sharpe ratio
    def negative_sharpe_ratio(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Bounds for weights
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Simplify or remove constraints
 #   constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
 #   if target_return is not None:
  #      constraints.append({'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target_return})

    # Different initial guess
    initial_guess = np.random.dirichlet(np.ones(num_assets), size=1)[0]

    # Different optimization method, if applicable
    opts = {'maxiter': 1000, 'ftol': 1e-06}

    try:
        result = minimize(negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, options=opts)
    except Exception as e:
        return {"success": False, "message": str(e)}

    # Adjusted return structure
    return {
        "success": result.success,
        "message": result.message,
        "optimal_weights": result.x if result.success else None
    }

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
    daily_returns = portfolio_data.pct_change().dropna().to_numpy()

    # Calculate mean returns for each stock
    mean_returns = portfolio_data.pct_change().mean().to_numpy()


    # Number of stocks in portfolio
    num_assets = portfolio_data.shape[1]

    # Optimization Variables
    weights = cp.Variable(num_assets)

    # Minimize Mean Absolute Deviation (MAD)
    portfolio_return = mean_returns @ weights
    mad = cp.sum(cp.abs(daily_returns @ weights - portfolio_return)) / len(daily_returns)
    objective = cp.Minimize(mad)

    # Constraints
    constraints = [
        cp.sum(weights) == 1,       # Sum of weights is 1
        weights >= 0,               # No short selling (weights are non-negative)
        portfolio_return >= target_return
    ]

    # Problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()

    # Check
    if prob.status not in ["infeasible", "unbounded"]:
        # Return optimal weights
        return weights.value
    else:
        print("Problem status:", prob.status)
        return None
    
def main():
    # API Key for Alpha Vantage
    api_key = '16SIXTINQJAUF0MZ'
    # Define stock tickers and date range
    tickers = ['AAPL', 'MSFT', 'GOOG']
    weights = [0.4, 0.3, 0.3]
    start_date = '2023-09-01'
    end_date = '2023-10-01'
    risk_free_rate = 0.02  # 2% risk-free rate

    market_ticker = 'VTHR' # Vanguard Russell 3000 Index ETF

    # Get stock data and create portfolio
    portfolio_data = create_portfolio(tickers, weights, api_key, start_date, end_date)

    # Get market data
    market_data = fetch_market_data(market_ticker, api_key) 

    # Debug - print portfolio data structure
    print("Portfolio Data Columns:", portfolio_data.columns)
    print("Number of weights:", len(weights))
    
    # Ensure the portfolio data columns match the tickers
    if len(portfolio_data.columns) != len(weights):
        print("Error: The number of stocks in the portfolio does not match the number of weights.")
        return
    
    if portfolio_data is None or market_data is None:
            print("Error: Unable to fetch data.")
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

    # Markowitz Portfolio Optimization
    print("Performing Markowitz Optimization...")
    markowitz_optimization_result = markowitz_optimization(portfolio_data, target_return=0.1, risk_free_rate=risk_free_rate)
    if markowitz_optimization_result['success']:
        print("Optimal Weights (Markowitz):", markowitz_optimization_result['optimal_weights'])
    else:
        print("Markowitz Optimization failed:", markowitz_optimization_result['message'])

    # CAPM
    print("\nEvaluating with CAPM...")
    capm_returns = capm_evaluation(portfolio_data, market_data, risk_free_rate)
    print("Expected Returns (CAPM):", capm_returns)
    
    if portfolio_data is not None and market_data is not None:
        capm_returns = capm_evaluation(portfolio_data, market_data, risk_free_rate)
        print("CAPM Expected Returns:", capm_returns)
    
    # Monte Carlo
    print("\nPerforming Monte Carlo Optimization...")
    mc_simulation = monte_carlo_optimization(portfolio_data, num_portfolios=10000)
    print(mc_simulation.head())  # Display first few rows of the simulation results

    # Mean-MAD
    print("\nPerforming Mean-MAD Optimization...")
    mad_weights = mean_mad_optimization(portfolio_data, target_return=0.1)
    if mad_weights is not None:
        print("Optimal Weights (Mean-MAD):", mad_weights)


    # Comparison and Analysis...
    # Visualization...

if __name__ == "__main__":
    main()

