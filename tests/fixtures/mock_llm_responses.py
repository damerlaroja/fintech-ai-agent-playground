"""
Mock LLM Responses for Testing

Provides pre-canned LLM responses to avoid real API calls during development.
Follows WINDSURF_RULES.md requirements for mock-first development.

Usage:
    from tests.fixtures.mock_llm_responses import get_mock_var_response
    response = get_mock_var_response("AAPL", 0.95)
"""

from typing import Dict, Any
import json

def get_mock_var_response(ticker: str, confidence: float) -> Dict[str, Any]:
    """Get mock VaR calculation response."""
    return {
        "ticker": ticker,
        "confidence_level": confidence,
        "analysis_period": 30,
        "data_points": 30,
        "var_1day": -0.0250,
        "var_1day_parametric": -0.0234,
        "max_drawdown": -0.0789,
        "volatility_annualized": 0.2847,
        "sharpe_ratio": 1.156,
        "analysis_date": "2026-03-07",
        "interpretation": f"There is a {confidence:.0%} chance that {ticker} will lose more than 2.50% in one day."
    }

def get_mock_beta_response(ticker: str, benchmark: str = "^GSPC") -> Dict[str, Any]:
    """Get mock beta calculation response."""
    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "beta": 1.23,
        "correlation": 0.67,
        "r_squared": 0.45,
        "analysis_period": 90,
        "data_points": 90,
        "interpretation": f"{ticker} has a beta of 1.23 relative to {benchmark}, indicating higher volatility than the market."
    }

def get_mock_correlation_matrix(tickers: list) -> Dict[str, Any]:
    """Get mock correlation matrix response."""
    correlations = {}
    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i+1:]:
            correlations[f"{ticker1}_{ticker2}"] = 0.45
    
    return {
        "tickers": tickers,
        "correlation_matrix": correlations,
        "average_correlation": 0.45,
        "diversification_score": 1.55,
        "analysis_period": 30,
        "data_points": 30
    }

def get_mock_portfolio_risk_score(tickers: list, weights: list = None) -> Dict[str, Any]:
    """Get mock portfolio risk score response."""
    if weights is None:
        weights = [1.0/len(tickers)] * len(tickers)
    
    return {
        "tickers": tickers,
        "weights": weights,
        "overall_risk_score": 72.3,
        "risk_level": "High Risk",
        "var_1day_portfolio": -0.0345,
        "portfolio_beta": 1.18,
        "diversification_benefit": 0.23,
        "recommendations": [
            "Consider adding uncorrelated assets",
            "Monitor concentration risk",
            "Rebalance quarterly"
        ]
    }

def get_mock_market_research_response(query: str) -> str:
    """Get mock market research response."""
    return f"""
Based on the analysis of {query}, here are the key findings:

📊 **Market Analysis**
- Current market sentiment: Neutral to bullish
- Key technical indicators: Mixed signals
- Volume analysis: Above average

💡 **Key Insights**
- The market is showing resilience despite recent volatility
- Sector rotation is underway with technology leading
- Risk appetite appears to be returning gradually

⚠️ **Risk Factors**
- Inflation concerns remain elevated
- Geopolitical tensions could impact supply chains
- Interest rate policy uncertainty persists

📈 **Recommendations**
- Maintain diversified portfolio allocation
- Consider defensive positions for near-term
- Monitor earnings season closely

*This analysis is for educational purposes only and should not be considered investment advice.*
"""

def get_mock_synthetic_transactions(ticker: str, count: int = 100) -> Dict[str, Any]:
    """Get mock synthetic transaction data."""
    import random
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Generate mock transaction data
    transactions = []
    base_price = 150.0
    
    for i in range(count):
        date = datetime.now() - timedelta(days=count-i)
        price = base_price * (1 + random.uniform(-0.05, 0.05))
        volume = random.randint(1000, 10000)
        
        transactions.append({
            "date": date.strftime("%Y-%m-%d"),
            "price": round(price, 2),
            "volume": volume,
            "action": random.choice(["buy", "sell"])
        })
    
    return {
        "ticker": ticker,
        "transaction_count": count,
        "transactions": transactions,
        "statistics": {
            "avg_price": round(base_price, 2),
            "total_volume": sum(t["volume"] for t in transactions),
            "price_volatility": 0.034
        }
    }

# Mock LLM client for testing
class MockLLMClient:
    """Mock LLM client that returns pre-canned responses."""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Return mock response based on prompt content."""
        prompt_lower = prompt.lower()
        
        if "var" in prompt_lower and "confidence" in prompt_lower:
            return json.dumps(get_mock_var_response("AAPL", 0.95))
        elif "beta" in prompt_lower:
            return json.dumps(get_mock_beta_response("AAPL"))
        elif "correlation" in prompt_lower:
            return json.dumps(get_mock_correlation_matrix(["AAPL", "MSFT", "GOOGL"]))
        elif "portfolio" in prompt_lower and "risk" in prompt_lower:
            return json.dumps(get_mock_portfolio_risk_score(["AAPL", "MSFT", "GOOGL"]))
        elif "synthetic" in prompt_lower or "transaction" in prompt_lower:
            return json.dumps(get_mock_synthetic_transactions("AAPL"))
        else:
            return get_mock_market_research_response(prompt)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Make the mock client callable."""
        return self.invoke(prompt, **kwargs)

# Factory function for creating mock clients
def create_mock_llm(provider: str = "mock", model: str = "mock-model"):
    """Create a mock LLM client for testing."""
    return MockLLMClient(model)
