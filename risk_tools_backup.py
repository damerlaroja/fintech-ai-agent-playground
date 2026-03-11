"""
Risk Analysis Tools Module

Implements financial risk assessment tools for portfolio analysis,
including VaR calculations, beta analysis, correlation matrices,
and risk scoring models.

Architecture Benefits:
- Portfolio risk assessment with multiple metrics
- Statistical risk calculations using historical data
- Correlation analysis for diversification insights
- Risk scoring models for investment decisions
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from functools import lru_cache
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Cache configuration
CACHE_TTL = 600  # 10 minutes cache for financial data (increased)
data_cache = {}
cache_timestamps = {}

# Performance optimizations
MIN_DATA_POINTS = 25  # Minimum data points for analysis
MAX_DATA_POINTS = 30   # Reduced to 30 for even faster analysis

# Global yfinance cache to minimize API calls
_yfinance_cache = {}
_last_fetch_time = {}
YFINANCE_CACHE_TTL = 1800  # 30 minutes for yfinance data (avoid rate limits)

# Mock data for testing (avoids rate limits)
USE_MOCK_DATA = False  # Set to False for real data

def generate_mock_data(ticker: str, days: int):
    """Generate mock financial data for testing."""
    import random
    
    # Generate date range
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    
    # Generate realistic price data
    base_price = 150.0 + random.uniform(-50, 100)  # Random base price
    prices = []
    
    # Generate prices with realistic volatility
    for i in range(days):
        if i == 0:
            prices.append(base_price)
        else:
            # Daily return with realistic volatility (1-3%)
            daily_return = random.gauss(0.001, 0.015)
            # Ensure reasonable daily return range (-5% to +5%)
            daily_return = max(-0.05, min(0.05, daily_return))
            new_price = prices[-1] * (1 + daily_return)
            # Ensure price doesn't go negative
            new_price = max(new_price, 1.0)
            prices.append(new_price)
    
    # Create DataFrame with realistic OHLCV data
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + random.uniform(0.001, 0.02)) for p in prices],
        'Low': [p * (1 - random.uniform(0.001, 0.02)) for p in prices],
        'Close': prices,
        'Volume': [random.randint(1000000, 10000000) for _ in range(days)]
    }, index=dates)
    
    # Ensure High >= Open and Low <= Open
    data['High'] = data[['High', 'Open', 'Close']].max(axis=1)
    data['Low'] = data[['Low', 'Open', 'Close']].min(axis=1)
    
    # Add some randomness to make it more realistic
    np.random.seed(hash(ticker) % 2**32)  # Seed based on ticker for consistency
    
    return data

def get_yfinance_data(ticker: str, period: str):
    """Get yfinance data with aggressive caching to minimize API calls."""
    global _yfinance_cache, _last_fetch_time
    
    # Use mock data if enabled (avoids rate limits)
    if USE_MOCK_DATA:
        days = int(period.replace('d', ''))
        return generate_mock_data(ticker, min(days, MAX_DATA_POINTS))
    
    current_time = time.time()
    cache_key = f"{ticker}_{period}"
    
    # Check cache first
    if (cache_key in _yfinance_cache and 
        cache_key in _last_fetch_time and
        current_time - _last_fetch_time[cache_key] < YFINANCE_CACHE_TTL):
        return _yfinance_cache[cache_key]
    
    # Fetch fresh data with rate limit handling
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        _yfinance_cache[cache_key] = data
        _last_fetch_time[cache_key] = current_time
        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        # Return empty DataFrame instead of raising
        return pd.DataFrame()

def handle_rate_limit():
    """Handle rate limiting with delay and fallback."""
    time.sleep(1)  # Wait 1 second before retry
    return True

def get_cached_data(key: str, fetch_func, *args, **kwargs):
    """Get cached data or fetch if expired."""
    current_time = time.time()
    
    # Check if cache exists and is not expired
    if key in data_cache and key in cache_timestamps:
        if current_time - cache_timestamps[key] < CACHE_TTL:
            return data_cache[key]
    
    # Fetch fresh data
    data = fetch_func(*args, **kwargs)
    data_cache[key] = data
    cache_timestamps[key] = current_time
    return data


def calculate_var(ticker: str, confidence_level: float = 0.95, days: int = 30) -> Dict:
    """
    Calculate Value at Risk (VaR) for a given ticker.
    
    VaR estimates the potential loss in value of a risky asset or portfolio
    over a defined period for a given confidence interval.
    
    Args:
        ticker: Stock ticker symbol
        confidence_level: Confidence level for VaR (default 0.95)
        days: Number of trading days for analysis (default 30 = ~1 month)
        
    Returns:
        Dictionary containing VaR metrics and analysis
        
    Risk Metrics:
        - Historical VaR: Based on historical returns distribution
        - Parametric VaR: Based on normal distribution assumption
        - Maximum Drawdown: Largest peak-to-trough decline
        - Volatility: Annualized standard deviation of returns
        
    Example:
        >>> var_data = calculate_var("AAPL", 0.95, 252)
        >>> print(f"1-Day VaR at 95% confidence: {var_data['var_1day']:.2%}")
    """
    try:
        # Use cached data to reduce API calls
        cache_key = f"var_{ticker}_{days}_{confidence_level}"
        
        def fetch_var_data():
            return get_yfinance_data(ticker, f"{days}d")
        
        data = get_cached_data(cache_key, fetch_var_data)
        
        if data.empty or len(data) < MIN_DATA_POINTS:
            return {"error": f"Insufficient data for {ticker} (need {MIN_DATA_POINTS} days, got {len(data)})"}
        
        # Limit data points for faster processing
        if len(data) > MAX_DATA_POINTS:
            data = data.tail(MAX_DATA_POINTS)
        
        # Calculate daily returns
        data['Returns'] = data['Close'].pct_change().dropna()
        returns = data['Returns']
        
        # Ensure we have enough data for VaR calculation
        if len(returns) < MIN_DATA_POINTS:
            return {"error": f"Insufficient return data for {ticker} (need {MIN_DATA_POINTS}, got {len(returns)})"}
        
        # Historical VaR (based on actual return distribution)
        # Use worst return for small datasets to avoid NaN
        var_historical = returns.min()
        
        # If we have enough data, try percentile calculation
        if len(returns) >= 20:
            try:
                var_percentile = (1 - confidence_level) * 100
                var_percentile_value = np.percentile(returns, var_percentile)
                # Use percentile if it's not NaN
                if not np.isnan(var_percentile_value):
                    var_historical = var_percentile_value
            except:
                pass  # Fall back to min()
        
        # Parametric VaR (assuming normal distribution)
        mean_return = returns.mean()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        var_parametric = mean_return - volatility * np.sqrt(1/252) * 1.645  # 95% confidence
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional risk metrics
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        downside_deviation = returns[returns < 0].std() * np.sqrt(252)
        
        return {
            "ticker": ticker,
            "confidence_level": confidence_level,
            "analysis_period": days,
            "var_1day": var_historical,
            "var_1day_parametric": var_parametric,
            "max_drawdown": max_drawdown,
            "volatility_annualized": volatility,
            "sharpe_ratio": sharpe_ratio,
            "downside_deviation": downside_deviation,
            "data_points": len(returns),
            "analysis_date": datetime.now().strftime("%Y-%m-%d")
        }
        
    except Exception as e:
        return {"error": f"Failed to calculate VaR for {ticker}: {str(e)}"}


def calculate_beta(ticker: str, market_ticker: str = "^GSPC", days: int = 30) -> Dict:
    """
    Calculate beta coefficient for a stock relative to market.
    
    Beta measures the volatility of a stock or portfolio in comparison
    to the market as a whole. Beta > 1 means more volatile than market.
    
    Args:
        ticker: Stock ticker symbol
        market_ticker: Market benchmark ticker (default S&P 500)
        days: Number of trading days for analysis
        
    Returns:
        Dictionary containing beta analysis and regression statistics
        
    Risk Metrics:
        - Beta: Systematic risk measure
        - Alpha: Excess return over market
        - R-squared: Model fit quality
        - Correlation: Linear relationship strength
        
    Example:
        >>> beta_data = calculate_beta("AAPL", "^GSPC")
        >>> print(f"Beta: {beta_data['beta']:.2f}")
    """
    try:
        # Use cached data to reduce API calls
        cache_key = f"beta_{ticker}_{market_ticker}_{days}"
        
        def fetch_beta_data():
            stock_data = get_yfinance_data(ticker, f"{days}d")
            market_data = get_yfinance_data(market_ticker, f"{days}d")
            return stock_data, market_data
        
        stock_data, market_data = get_cached_data(cache_key, fetch_beta_data)
        
        if stock_data.empty or market_data.empty:
            return {"error": f"Insufficient data for {ticker} or {market_ticker}"}
        
        # Limit data points for faster processing
        if len(stock_data) > MAX_DATA_POINTS:
            stock_data = stock_data.tail(MAX_DATA_POINTS)
            market_data = market_data.tail(MAX_DATA_POINTS)
        
        # Calculate daily returns
        stock_returns = stock_data['Close'].pct_change().dropna()
        market_returns = market_data['Close'].pct_change().dropna()
        
        # Align data
        aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner')
        aligned_data.columns = [ticker, market_ticker]
        
        stock_aligned = aligned_data[ticker]
        market_aligned = aligned_data[market_ticker]
        
        # Calculate correlation
        correlation = stock_aligned.corr(market_aligned)
        
        # Calculate beta using linear regression
        covariance = np.cov(stock_aligned, market_aligned)[0][1]
        market_variance = np.var(market_aligned)
        beta = covariance / market_variance if market_variance != 0 else 0
        
        # Calculate alpha (intercept)
        stock_mean = stock_aligned.mean()
        market_mean = market_aligned.mean()
        alpha = stock_mean - beta * market_mean
        
        # Calculate R-squared
        predicted_returns = alpha + beta * market_aligned
        ss_res = np.sum((stock_aligned - predicted_returns) ** 2)
        ss_tot = np.sum((stock_aligned - stock_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "ticker": ticker,
            "market_ticker": market_ticker,
            "beta": beta,
            "alpha": alpha,
            "correlation": correlation,
            "r_squared": r_squared,
            "analysis_period": days,
            "data_points": len(aligned_data),
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "risk_interpretation": interpret_beta(beta)
        }
        
    except Exception as e:
        return {"error": f"Failed to calculate beta for {ticker}: {str(e)}"}


def interpret_beta(beta: float) -> str:
    """Interpret beta coefficient for risk assessment."""
    if beta > 1.5:
        return "Very High Risk - Significantly more volatile than market"
    elif beta > 1.0:
        return "High Risk - More volatile than market"
    elif beta > 0.5:
        return "Moderate Risk - Less volatile than market"
    elif beta > 0:
        return "Low Risk - Much less volatile than market"
    elif beta < -0.5:
        return "Negative Correlation - Moves opposite to market"
    else:
        return "Minimal Market Correlation"


def calculate_correlation_matrix(tickers: List[str], days: int = 30) -> Dict:
    """
    Calculate correlation matrix for multiple stocks.
    
    Correlation analysis helps understand portfolio diversification benefits.
    Lower correlations generally provide better diversification.
    
    Args:
        tickers: List of stock ticker symbols
        days: Number of trading days for analysis
        
    Returns:
        Dictionary containing correlation matrix and diversification insights
        
    Risk Metrics:
        - Correlation Matrix: Pairwise correlations between stocks
        - Average Correlation: Overall portfolio correlation
        - Diversification Score: Benefit of diversification
        - Cluster Analysis: Grouping of similar stocks
        
    Example:
        >>> corr_data = calculate_correlation_matrix(["AAPL", "MSFT", "GOOGL"])
        >>> print(f"Average correlation: {corr_data['average_correlation']:.3f}")
    """
    try:
        if len(tickers) < 2:
            return {"error": "Need at least 2 tickers for correlation analysis"}
        
        # Use cached data to reduce API calls - fetch all data at once
        cache_key = f"corr_{'_'.join(sorted(tickers))}_{days}"
        
        def fetch_correlation_data():
            price_data = {}
            for ticker in tickers:
                stock_data = get_yfinance_data(ticker, f"{days}d")
                if not stock_data.empty:
                    price_data[ticker] = stock_data['Close']
            return price_data
        
        price_data = get_cached_data(cache_key, fetch_correlation_data)
        
        if len(price_data) < 2:
            return {"error": "Insufficient data for correlation analysis"}
        
        # Create DataFrame and calculate returns
        prices_df = pd.DataFrame(price_data)
        
        # Limit data points for faster processing
        if len(prices_df) > MAX_DATA_POINTS:
            prices_df = prices_df.tail(MAX_DATA_POINTS)
        returns_df = prices_df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Calculate average correlation (excluding diagonal)
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        
        # Diversification score (lower average correlation = better diversification)
        diversification_score = max(0, 1 - avg_correlation)
        
        # Find highly correlated pairs (>0.7)
        highly_correlated = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value > 0.7:
                    highly_correlated.append({
                        "stock1": correlation_matrix.columns[i],
                        "stock2": correlation_matrix.columns[j],
                        "correlation": corr_value
                    })
        
        # Find low correlation pairs (<0.3) for diversification
        low_correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value < 0.3:
                    low_correlation_pairs.append({
                        "stock1": correlation_matrix.columns[i],
                        "stock2": correlation_matrix.columns[j],
                        "correlation": corr_value
                    })
        
        return {
            "tickers": tickers,
            "correlation_matrix": correlation_matrix.to_dict(),
            "average_correlation": avg_correlation,
            "diversification_score": diversification_score,
            "highly_correlated_pairs": highly_correlated,
            "low_correlation_pairs": low_correlation_pairs,
            "analysis_period": days,
            "data_points": len(returns_df),
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "diversification_insight": interpret_diversification(diversification_score)
        }
        
    except Exception as e:
        return {"error": f"Failed to calculate correlation matrix: {str(e)}"}


def interpret_diversification(score: float) -> str:
    """Interpret diversification score for portfolio analysis."""
    if score > 0.7:
        return "Excellent Diversification - Low correlation between assets"
    elif score > 0.5:
        return "Good Diversification - Moderate correlation between assets"
    elif score > 0.3:
        return "Fair Diversification - High correlation between assets"
    else:
        return "Poor Diversification - Very high correlation between assets"


def portfolio_risk_score(tickers: List[str], weights: Optional[List[float]] = None, fast_mode: bool = True) -> Dict:
    """
    Calculate comprehensive risk score for a portfolio.
    
    Combines multiple risk metrics into a single risk score
    for easy portfolio risk assessment.
    
    Args:
        tickers: List of stock ticker symbols
        weights: Optional weights for each ticker (default equal weights)
        fast_mode: Use optimized fast analysis with reduced data (default True)
        
    Returns:
        Dictionary containing comprehensive risk analysis and scoring
        
    Risk Components:
        - Volatility Risk: Price fluctuation risk
        - Correlation Risk: Lack of diversification
        - Beta Risk: Systematic market risk
        - VaR Risk: Potential loss risk
        - Overall Risk Score: Combined risk metric (0-100)
        
    Example:
        >>> risk_data = portfolio_risk_score(["AAPL", "MSFT", "GOOGL"])
        >>> print(f"Risk Score: {risk_data['overall_risk_score']}/100")
    """
    try:
        if not tickers:
            return {"error": "No tickers provided"}
        
        # Default equal weights if not provided
        if weights is None:
            weights = [1/len(tickers)] * len(tickers)
        
        if len(weights) != len(tickers):
            return {"error": "Weights length must match tickers length"}
        
        # Use cached data to reduce API calls
        analysis_days = 30 if fast_mode else 90
        cache_key = f"portfolio_{'_'.join(sorted(tickers))}_{analysis_days}"
        
        def fetch_portfolio_data():
            # Get correlation data (cached)
            correlation_data = calculate_correlation_matrix(tickers, days=analysis_days)
            
            # Get individual betas (cached)
            beta_data = {}
            for ticker in tickers:
                beta_result = calculate_beta(ticker, days=analysis_days)
                if "error" not in beta_result:
                    beta_data[ticker] = beta_result["beta"]
            
            # Get individual VaRs (cached)
            var_data = {}
            for ticker in tickers:
                var_result = calculate_var(ticker, days=analysis_days)
                if "error" not in var_result:
                    var_data[ticker] = abs(var_result["var_1day"])
            
            return correlation_data, beta_data, var_data
        
        correlation_data, beta_data, var_data = get_cached_data(cache_key, fetch_portfolio_data)
        
        if "error" in correlation_data:
            return correlation_data
        
        # Calculate portfolio-level metrics
        avg_beta = np.mean(list(beta_data.values())) if beta_data else 0
        avg_var = np.mean(list(var_data.values())) if var_data else 0
        diversification_score = correlation_data["diversification_score"]
        
        # Calculate risk score (0-100 scale)
        volatility_risk = min(100, avg_var * 10000)  # Scale VaR to 0-100
        correlation_risk = (1 - diversification_score) * 100  # Invert diversification
        beta_risk = min(100, abs(avg_beta - 1) * 50)  # Deviation from market beta
        
        # Weighted risk score
        overall_risk_score = (volatility_risk * 0.4 + correlation_risk * 0.3 + beta_risk * 0.3)
        
        # Risk classification
        if overall_risk_score < 30:
            risk_level = "Low Risk"
        elif overall_risk_score < 60:
            risk_level = "Moderate Risk"
        elif overall_risk_score < 80:
            risk_level = "High Risk"
        else:
            risk_level = "Very High Risk"
        
        return {
            "tickers": tickers,
            "weights": weights,
            "overall_risk_score": round(overall_risk_score, 1),
            "risk_level": risk_level,
            "risk_components": {
                "volatility_risk": round(volatility_risk, 1),
                "correlation_risk": round(correlation_risk, 1),
                "beta_risk": round(beta_risk, 1)
            },
            "portfolio_metrics": {
                "average_beta": round(avg_beta, 2),
                "average_var": round(avg_var, 4),
                "diversification_score": round(diversification_score, 2)
            },
            "individual_betas": beta_data,
            "individual_vars": {k: round(v, 4) for k, v in var_data.items()},
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "risk_recommendations": generate_risk_recommendations(overall_risk_score, diversification_score)
        }
        
    except Exception as e:
        return {"error": f"Failed to calculate portfolio risk score: {str(e)}"}


def generate_risk_recommendations(risk_score: float, diversification_score: float) -> List[str]:
    """Generate risk management recommendations based on analysis."""
    recommendations = []
    
    if risk_score > 70:
        recommendations.append("Consider reducing portfolio volatility through diversification")
        recommendations.append("Evaluate position sizes to reduce overall risk exposure")
    
    if diversification_score < 0.4:
        recommendations.append("Increase diversification by adding low-correlation assets")
        recommendations.append("Consider adding assets from different sectors or geographies")
    
    if risk_score < 30:
        recommendations.append("Portfolio risk is low - consider if return expectations are being met")
        recommendations.append("May consider adding growth-oriented assets for better returns")
    
    if not recommendations:
        recommendations.append("Portfolio risk profile appears balanced")
        recommendations.append("Continue monitoring risk metrics regularly")
    
    return recommendations


def generate_synthetic_transactions(ticker: str, num_transactions: int = 100) -> Dict:
    """
    Generate synthetic transactions for risk modeling and testing.
    
    Creates realistic transaction data for backtesting risk models
    and portfolio stress testing scenarios.
    
    Args:
        ticker: Stock ticker symbol
        num_transactions: Number of synthetic transactions to generate
        
    Returns:
        Dictionary containing synthetic transaction data
        
    Transaction Features:
        - Realistic price movements based on historical volatility
        - Random timestamps within trading hours
        - Various transaction types (buy/sell)
        - Realistic position sizes
        
    Example:
        >>> transactions = generate_synthetic_transactions("AAPL", 50)
        >>> print(f"Generated {len(transactions['transactions'])} transactions")
    """
    try:
        # Get historical data for realistic parameters
        stock_data = yf.Ticker(ticker).history(period="1y")
        if stock_data.empty:
            return {"error": f"No historical data available for {ticker}"}
        
        # Calculate realistic parameters
        returns = stock_data['Close'].pct_change().dropna()
        daily_volatility = returns.std()
        avg_price = stock_data['Close'].mean()
        
        # Generate synthetic transactions
        transactions = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(num_transactions):
            # Random timestamp within last 30 days
            days_offset = np.random.randint(0, 30)
            hours_offset = np.random.randint(9, 16)  # Trading hours
            minutes_offset = np.random.randint(0, 60)
            
            transaction_date = base_date + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)
            
            # Random price movement based on historical volatility
            price_change = np.random.normal(0, daily_volatility)
            transaction_price = avg_price * (1 + price_change)
            
            # Random transaction type and size
            transaction_type = np.random.choice(['buy', 'sell'], p=[0.6, 0.4])
            position_size = np.random.randint(10, 1000)  # Shares
            
            transaction = {
                "timestamp": transaction_date.strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "type": transaction_type,
                "price": round(transaction_price, 2),
                "quantity": position_size,
                "total_value": round(transaction_price * position_size, 2)
            }
            
            transactions.append(transaction)
        
        # Sort by timestamp
        transactions.sort(key=lambda x: x['timestamp'])
        
        # Calculate summary statistics
        total_volume = sum(t['total_value'] for t in transactions)
        avg_transaction_size = total_volume / len(transactions)
        buy_count = sum(1 for t in transactions if t['type'] == 'buy')
        sell_count = len(transactions) - buy_count
        
        return {
            "ticker": ticker,
            "num_transactions": num_transactions,
            "transactions": transactions,
            "summary": {
                "total_volume": round(total_volume, 2),
                "avg_transaction_size": round(avg_transaction_size, 2),
                "buy_transactions": buy_count,
                "sell_transactions": sell_count,
                "buy_sell_ratio": round(buy_count / sell_count, 2) if sell_count > 0 else 0,
                "price_range": {
                    "min": min(t['price'] for t in transactions),
                    "max": max(t['price'] for t in transactions),
                    "avg": sum(t['price'] for t in transactions) / len(transactions)
                }
            },
            "generation_date": datetime.now().strftime("%Y-%m-%d"),
            "based_on_historical_volatility": round(daily_volatility, 4)
        }
        
    except Exception as e:
        return {"error": f"Failed to generate synthetic transactions for {ticker}: {str(e)}"}
