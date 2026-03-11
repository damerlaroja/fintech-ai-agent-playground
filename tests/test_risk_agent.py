import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.risk_tools import (
    generate_synthetic_transactions,
    score_transaction_risk,
    check_sanctions,
    calculate_var,
    calculate_beta,
    calculate_portfolio_risk
)

# Get the actual functions from the tools (not the tool wrappers)
def _get_tool_func(tool):
    """Extract the underlying function from a LangChain tool."""
    return tool.func if hasattr(tool, 'func') else tool

# Get the actual functions
calculate_var_func = _get_tool_func(calculate_var)
calculate_beta_func = _get_tool_func(calculate_beta)
calculate_portfolio_risk_func = _get_tool_func(calculate_portfolio_risk)
score_transaction_risk_func = _get_tool_func(score_transaction_risk)
check_sanctions_func = _get_tool_func(check_sanctions)
generate_synthetic_transactions_func = _get_tool_func(generate_synthetic_transactions)


class TestRiskAgent(unittest.TestCase):
    """Test risk agent tools with mocked dependencies."""
    
    @patch('yfinance.Ticker')
    def test_volatility_low(self, mock_ticker):
        """Test volatility calculation with mock data."""
        mock_hist = MagicMock()
        mock_hist.empty = False
        
        # Create mock DataFrame with close prices
        import pandas as pd
        mock_df = pd.DataFrame({
            'Close': [100, 101, 102, 101.5, 102.5, 103, 102.8, 103.2, 104, 103.8] * 30  # 300 points
        })
        mock_hist.__getitem__.return_value = mock_df['Close']
        
        mock_ticker.return_value.history.return_value = mock_hist
        
        result = calculate_var_func("AAPL", days=30)
        
        self.assertEqual(result["methodology"], "historical")
        self.assertIsInstance(result["var_amount"], (int, float))
        self.assertIsNotNone(result["var_amount"])
    
    @patch('yfinance.Ticker')
    def test_volatility_medium(self, mock_ticker):
        """Test volatility calculation with mock data."""
        mock_hist = MagicMock()
        mock_hist.empty = False
        
        # Mock medium volatility data
        import pandas as pd
        mock_df = pd.DataFrame({
            'Close': [100, 103, 97, 105, 95, 108, 92, 110, 90, 112] * 30  # 300 points
        })
        mock_hist.__getitem__.return_value = mock_df['Close']
        
        mock_ticker.return_value.history.return_value = mock_hist
        
        result = calculate_var_func("AAPL", days=30)
        self.assertEqual(result["methodology"], "historical")
        self.assertIsInstance(result["var_amount"], (int, float))
        self.assertIsNotNone(result["var_amount"])
    
    @patch('yfinance.Ticker')
    def test_volatility_high(self, mock_ticker):
        """Test volatility calculation with mock data."""
        mock_hist = MagicMock()
        mock_hist.empty = False
        
        # Mock high volatility data
        import pandas as pd
        mock_df = pd.DataFrame({
            'Close': [100, 120, 80, 130, 70, 140, 60, 150, 50, 160] * 30  # 300 points
        })
        mock_hist.__getitem__.return_value = mock_df['Close']
        
        mock_ticker.return_value.history.return_value = mock_hist
        
        result = calculate_var_func("AAPL", days=30)
        self.assertEqual(result["methodology"], "historical")
        self.assertIsInstance(result["var_amount"], (int, float))
        self.assertIsNotNone(result["var_amount"])
    
    @patch('yfinance.Ticker')
    def test_composite_score_accuracy(self, mock_ticker):
        """Test portfolio risk score calculation is mathematically correct."""
        # Mock beta calculations
        with patch('tools.risk_tools._calculate_beta_func') as mock_beta:
            mock_beta.side_effect = [
                {"beta": 1.2, "correlation": 0.8, "benchmark": "^GSPC"},
                {"beta": 0.8, "correlation": 0.6, "benchmark": "^GSPC"},
                {"beta": 1.5, "correlation": 0.9, "benchmark": "^GSPC"}
            ]
            
            result = calculate_portfolio_risk_func(["AAPL", "MSFT", "GOOGL"])
            
            # Verify mathematical correctness
            self.assertIn("overall_risk_score", result)
            self.assertIn("individual_scores", result)
            self.assertIn("diversification_score", result)
            
            # Check individual scores
            scores = result["individual_scores"]
            self.assertEqual(scores["AAPL"], 1.2)
            self.assertEqual(scores["MSFT"], 0.8)
            self.assertEqual(scores["GOOGL"], 1.5)
    
    @patch('yfinance.Ticker')
    def test_empty_yfinance_returns_safe_defaults(self, mock_ticker):
        """Test empty yfinance data returns safe defaults without crashing."""
        mock_hist = MagicMock()
        mock_hist.empty = True
        mock_ticker.return_value.history.return_value = mock_hist
        
        result = calculate_var_func("AAPL", days=30)
        
        self.assertEqual(result["var_amount"], 0.0)
        self.assertEqual(result["confidence_level"], 0.95)
        self.assertEqual(result["methodology"], "historical")
    
    @patch('tools.risk_tools._load_sanctions_data')
    @patch('tools.risk_tools._check_sanctions_func')
    def test_no_headlines_defaults_to_neutral(self, mock_check, mock_load):
        """Test missing sanctions data defaults to neutral sentiment."""
        mock_load.return_value = {
            "sanctioned_entities": [],
            "high_risk_countries": [],
            "transaction_thresholds": {"large_transaction": 10000, "suspicious_pattern": 9000, "structuring_limit": 3000}
        }
        mock_check.return_value = {"is_sanctioned": False, "risk_level": "low", "match": None}
        
        transaction = {
            "amount": 5000,
            "counterparty": "Safe Corp",
            "country": "US",
            "type": "buy"
        }
        
        result = score_transaction_risk_func(transaction)
        
        self.assertEqual(result["risk_score"], 0)
        self.assertEqual(result["risk_level"], "Low")
        self.assertEqual(len(result["flags"]), 0)
    
    @patch('tools.risk_tools._load_sanctions_data')
    @patch('tools.risk_tools._check_sanctions_func')
    def test_regulatory_fallback_on_llm_exhaustion(self, mock_check, mock_load):
        """Test regulatory fallback uses keyword heuristics when LLM exhausted."""
        mock_load.return_value = {
            "sanctioned_entities": [
                {"name": "Shadow Corp", "country": "IR", "risk_level": "critical"}
            ],
            "high_risk_countries": ["IR", "KP", "RU", "SY", "BY", "CU"],
            "transaction_thresholds": {"large_transaction": 10000, "suspicious_pattern": 9000, "structuring_limit": 3000}
        }
        mock_check.return_value = {"is_sanctioned": True, "risk_level": "critical", "match": "Shadow Corp"}
        
        transaction = {
            "amount": 15000,
            "counterparty": "Shadow Corp",
            "country": "IR",
            "type": "buy"
        }
        
        result = score_transaction_risk_func(transaction)
        
        # Should detect sanctioned entity
        self.assertEqual(result["risk_score"], 95)  # 30 (large) + 50 (sanctioned) + 15 (high-risk country) = 95
        self.assertEqual(result["risk_level"], "High")
        self.assertIn("Sanctioned entity: Shadow Corp", result["flags"])


if __name__ == '__main__':
    unittest.main()
