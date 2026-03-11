"""
Test Fixtures Module

Contains mock responses and test utilities for avoiding real API calls
during development and testing, following WINDSURF_RULES.md requirements.
"""

from .mock_llm_responses import (
    get_mock_var_response,
    get_mock_beta_response,
    get_mock_correlation_matrix,
    get_mock_portfolio_risk_score,
    get_mock_market_research_response,
    get_mock_synthetic_transactions,
    MockLLMClient,
    create_mock_llm
)

__all__ = [
    'get_mock_var_response',
    'get_mock_beta_response', 
    'get_mock_correlation_matrix',
    'get_mock_portfolio_risk_score',
    'get_mock_market_research_response',
    'get_mock_synthetic_transactions',
    'MockLLMClient',
    'create_mock_llm'
]
