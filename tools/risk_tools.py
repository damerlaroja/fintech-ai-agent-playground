from __future__ import annotations
import json
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from datetime import datetime, timedelta
from langchain_core.tools import tool

# Store function references for internal calls
_check_sanctions_func = None
_calculate_beta_func = None

def _init_func_refs():
    """Initialize function references for internal tool calls."""
    global _check_sanctions_func, _calculate_beta_func
    if _check_sanctions_func is None:
        _check_sanctions_func = check_sanctions.func if hasattr(check_sanctions, 'func') else check_sanctions
    if _calculate_beta_func is None:
        _calculate_beta_func = calculate_beta.func if hasattr(calculate_beta, 'func') else calculate_beta

# Load sanctions data
def _load_sanctions_data() -> Dict:
    """Load sanctions data from JSON file."""
    try:
        with open(r'data\mock_sanctions.json', 'r') as f:
            return json.load(f)
    except:
        return {
            "sanctioned_entities": [],
            "high_risk_countries": [],
            "transaction_thresholds": {
                "large_transaction": 10000,
                "suspicious_pattern": 9000,
                "structuring_limit": 3000
            }
        }

@tool
def generate_synthetic_transactions(ticker: str, num_transactions: int = 20) -> list[dict]:
    """Generate synthetic financial transactions for risk analysis.
    
    Args:
        ticker: Stock ticker symbol
        num_transactions: Number of transactions to generate
    
    Returns:
        List of transaction dictionaries with amount, counterparty, country, type
    """
    sanctions_data = _load_sanctions_data()
    transactions = []
    
    for i in range(num_transactions):
        # Generate realistic transaction data
        amount = np.random.uniform(1000, 50000)
        transaction_type = np.random.choice(['buy', 'sell'])
        
        # Mix of safe and risky counterparties
        if np.random.random() < 0.1:  # 10% risky
            counterparty = np.random.choice([entity["name"] for entity in sanctions_data["sanctioned_entities"]])
            country = np.random.choice(sanctions_data["high_risk_countries"])
        else:
            counterparty = f"Counterparty_{i+1}"
            country = np.random.choice(['US', 'GB', 'DE', 'JP', 'CA'])
        
        transactions.append({
            "amount": round(amount, 2),
            "counterparty": counterparty,
            "country": country,
            "type": transaction_type,
            "date": (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime("%Y-%m-%d")
        })
    
    return transactions

@tool
def score_transaction_risk(transaction: dict) -> dict:
    """Score a single transaction for risk level.
    
    Args:
        transaction: Transaction dict with amount, counterparty, country, type
    
    Returns:
        Dict with risk_score (0-100), risk_level, and flags list
    """
    _init_func_refs()
    sanctions_data = _load_sanctions_data()
    risk_score = 0
    flags = []
    
    # Amount-based risk
    amount = transaction.get("amount", 0)
    if amount > sanctions_data["transaction_thresholds"]["large_transaction"]:
        risk_score += 30
        flags.append("Large transaction")
    elif amount > sanctions_data["transaction_thresholds"]["suspicious_pattern"]:
        risk_score += 20
        flags.append("Suspicious amount")
    
    # Counterparty sanctions check
    counterparty = transaction.get("counterparty", "")
    country = transaction.get("country", "")
    
    sanctions_check = _check_sanctions_func(counterparty, country)
    if sanctions_check["is_sanctioned"]:
        risk_score += 50
        flags.append(f"Sanctioned entity: {sanctions_check['match']}")
    
    # Country risk
    if country in sanctions_data["high_risk_countries"]:
        risk_score += 15
        flags.append("High-risk country")
    
    # Cap at 100
    risk_score = min(risk_score, 100)
    
    # Determine risk level
    if risk_score >= 70:
        risk_level = "High"
    elif risk_score >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "flags": flags
    }

@tool
def check_sanctions(counterparty: str, country: str) -> dict:
    """Check if a counterparty appears on the mock sanctions list.
    
    Args:
        counterparty: Name of the counterparty entity
        country: Two-letter country code
    
    Returns:
        Dict with is_sanctioned bool, risk_level, and match details
    """
    sanctions_data = _load_sanctions_data()
    
    # Check entity sanctions
    for entity in sanctions_data["sanctioned_entities"]:
        if entity["name"].lower() in counterparty.lower():
            return {
                "is_sanctioned": True,
                "risk_level": entity["risk_level"],
                "match": entity["name"]
            }
    
    # Check country sanctions
    if country in sanctions_data["high_risk_countries"]:
        return {
            "is_sanctioned": True,
            "risk_level": "high",
            "match": f"Country: {country}"
        }
    
    return {"is_sanctioned": False, "risk_level": "low", "match": None}

@tool
def calculate_var(ticker: str, confidence: float = 0.95, days: int = 252) -> dict:
    """Calculate Value at Risk for a stock using historical simulation.
    
    Args:
        ticker: Stock ticker symbol
        confidence: Confidence level (default 0.95 for 95%)
        days: Number of trading days of history to use
    
    Returns:
        Dict with var_amount, confidence_level, methodology
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=f"{days}d")
        
        if hist.empty:
            return {"var_amount": 0.0, "confidence_level": confidence, "methodology": "historical"}
        
        # Calculate daily returns
        hist['Returns'] = hist['Close'].pct_change().dropna()
        returns = hist['Returns']
        
        if len(returns) < 30:  # Need at least 30 data points
            return {"var_amount": 0.0, "confidence_level": confidence, "methodology": "insufficient_data"}
        
        # Historical VaR (percentile-based)
        var_percentile = (1 - confidence) * 100
        var_amount = -np.percentile(returns, var_percentile) * hist['Close'].iloc[-1]
        
        return {
            "var_amount": round(var_amount, 2),
            "confidence_level": confidence,
            "methodology": "historical"
        }
        
    except Exception as e:
        return {"var_amount": 0.0, "confidence_level": confidence, "methodology": f"error: {str(e)}"}

@tool
def calculate_beta(ticker: str, benchmark: str = "^GSPC", days: int = 252) -> dict:
    """Calculate beta of a stock relative to a benchmark index.
    
    Args:
        ticker: Stock ticker symbol
        benchmark: Benchmark ticker (default S&P 500)
        days: Number of trading days of history
    
    Returns:
        Dict with beta, correlation, benchmark used
    """
    try:
        stock = yf.Ticker(ticker.upper())
        bench = yf.Ticker(benchmark)
        
        stock_hist = stock.history(period=f"{days}d")
        bench_hist = bench.history(period=f"{days}d")
        
        if stock_hist.empty or bench_hist.empty:
            return {"beta": 0.0, "correlation": 0.0, "benchmark": benchmark}
        
        # Calculate daily returns
        stock_returns = stock_hist['Close'].pct_change().dropna()
        bench_returns = bench_hist['Close'].pct_change().dropna()
        
        # Align data
        aligned_data = pd.concat([stock_returns, bench_returns], axis=1, join='inner').dropna()
        if len(aligned_data) < 30:
            return {"beta": 0.0, "correlation": 0.0, "benchmark": benchmark}
        
        stock_aligned = aligned_data.iloc[:, 0]
        bench_aligned = aligned_data.iloc[:, 1]
        
        # Calculate beta and correlation
        covariance = np.cov(stock_aligned, bench_aligned)[0][1]
        bench_variance = np.var(bench_aligned)
        beta = covariance / bench_variance if bench_variance != 0 else 0
        correlation = np.corrcoef(stock_aligned, bench_aligned)[0][1]
        
        return {
            "beta": round(beta, 3),
            "correlation": round(correlation, 3),
            "benchmark": benchmark
        }
        
    except Exception as e:
        return {"beta": 0.0, "correlation": 0.0, "benchmark": f"error: {str(e)}"}

@tool
def calculate_portfolio_risk(tickers: list[str]) -> dict:
    """Calculate aggregate risk score for a portfolio of stocks.
    
    Args:
        tickers: List of stock ticker symbols
    
    Returns:
        Dict with overall_risk_score, individual_scores, diversification_score
    """
    _init_func_refs()
    if not tickers:
        return {"overall_risk_score": 0.0, "individual_scores": {}, "diversification_score": 0.0}
    
    individual_scores = {}
    betas = []
    
    # Get individual betas
    for ticker in tickers:
        beta_result = _calculate_beta_func(ticker, days=90)  # Use 90 days for faster calculation
        if "error" not in str(beta_result.get("benchmark", "")):
            individual_scores[ticker] = beta_result["beta"]
            betas.append(abs(beta_result["beta"]))
    
    if not betas:
        return {"overall_risk_score": 0.0, "individual_scores": {}, "diversification_score": 0.0}
    
    # Calculate diversification score (lower average absolute beta = better diversification)
    avg_beta = np.mean(betas)
    diversification_score = max(0, 100 - avg_beta * 20)  # Normalize to 0-100
    
    # Overall risk score (weighted combination)
    volatility_component = np.std(betas) * 25  # Volatility of betas
    concentration_component = avg_beta * 30  # Average beta exposure
    overall_risk_score = min(100, volatility_component + concentration_component)
    
    return {
        "overall_risk_score": round(overall_risk_score, 1),
        "individual_scores": {k: round(v, 3) for k, v in individual_scores.items()},
        "diversification_score": round(diversification_score, 1)
    }
