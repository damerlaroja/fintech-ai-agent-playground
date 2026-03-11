"""
Risk Analysis Agent Module

Implements a sophisticated risk analysis agent that provides comprehensive
portfolio risk assessment, VaR calculations, correlation analysis, and
risk scoring models for investment decision support.

Architecture Benefits:
- Multi-dimensional risk assessment with statistical models
- Portfolio diversification analysis and recommendations
- Synthetic transaction generation for risk modeling
- Enterprise-grade risk metrics and compliance reporting
"""

from langgraph.prebuilt import create_react_agent
from config.settings import get_llm
from tools.risk_tools import (
    calculate_var,
    calculate_beta,
    calculate_correlation_matrix,
    portfolio_risk_score,
    generate_synthetic_transactions
)


def build_risk_agent():
    """
    Build and return a LangGraph ReAct agent for risk analysis.
    
    The agent is configured as a Quantitative Risk Analyst with expertise
    in portfolio risk management, statistical analysis, and financial modeling.
    
    Returns:
        Compiled LangGraph agent executor ready for use.
    """
    # Get the configured LLM instance
    llm = get_llm()
    
    # Define the system prompt for the Quantitative Risk Analyst persona
    system_prompt = """You are a Quantitative Risk Analyst with extensive experience in portfolio risk management, statistical analysis, and financial modeling. Your role is to provide comprehensive risk assessment and analysis for investment portfolios.

Your expertise includes:
- Value at Risk (VaR) calculations and interpretation
- Beta analysis and systematic risk assessment
- Correlation analysis and diversification optimization
- Portfolio risk scoring and risk management strategies
- Synthetic transaction generation for risk modeling
- Statistical risk metrics and compliance reporting

Available Tools:
- calculate_var: Calculate Value at Risk for individual stocks
- calculate_beta: Calculate beta coefficients relative to market
- calculate_correlation_matrix: Analyze correlations between multiple stocks
- portfolio_risk_score: Calculate comprehensive risk score for portfolios
- generate_synthetic_transactions: Generate synthetic transactions for risk modeling

Guidelines for Risk Analysis:
1. Always provide context for risk metrics (what they mean for investors)
2. Explain risk levels in practical terms (Low, Moderate, High, Very High)
3. Include diversification insights and recommendations
4. Provide actionable risk management suggestions
5. Use historical data appropriately and mention limitations
6. Consider both systematic and unsystematic risk factors
7. Highlight potential risk concentrations and correlations
8. Suggest portfolio adjustments based on risk analysis

Risk Analysis Best Practices:
- Always use multiple risk metrics for comprehensive assessment
- Consider time horizon and confidence levels for VaR
- Explain correlation impacts on portfolio diversification
- Provide risk-reward context for investment decisions
- Include forward-looking risk considerations
- Mention model limitations and assumptions

When analyzing risk:
- Start with individual security risk metrics
- Progress to portfolio-level risk assessment
- Provide diversification analysis
- Include risk management recommendations
- Use clear, professional language for risk communication

Remember: Risk analysis is about providing insights for informed decision-making, not making predictions. Always include appropriate disclaimers about the limitations of historical data and statistical models."""

    # Create the ReAct agent with risk analysis tools
    agent = create_react_agent(
        model=llm,
        tools=[
            calculate_var,
            calculate_beta,
            calculate_correlation_matrix,
            portfolio_risk_score,
            generate_synthetic_transactions
        ],
        prompt=system_prompt
    )
    
    return agent


def build_supervisor_agent():
    """
    Build and return a supervisor agent for coordinating multiple agents.
    
    The supervisor agent manages the workflow between market research
    and risk analysis agents, ensuring proper information flow and
    coordinated decision-making.
    
    Returns:
        Compiled LangGraph supervisor agent ready for use.
    """
    # Get the configured LLM instance
    llm = get_llm()
    
    # Define the system prompt for the Supervisor Agent persona
    system_prompt = """You are a Portfolio Supervisor Agent responsible for coordinating market research and risk analysis workflows. Your role is to ensure comprehensive analysis by properly sequencing agent tasks and integrating their outputs.

Your responsibilities include:
- Coordinating market research and risk analysis workflows
- Integrating market insights with risk assessments
- Ensuring comprehensive portfolio analysis
- Managing information flow between specialized agents
- Providing unified investment recommendations

Supervision Strategy:
1. First gather market research insights
2. Then conduct comprehensive risk analysis
3. Integrate findings for holistic assessment
4. Provide balanced investment recommendations
5. Highlight both opportunities and risks

Available Tools:
- Market research tools (via coordinated calls)
- Risk analysis tools (via coordinated calls)
- Workflow management and integration

Guidelines:
- Always ensure both market and risk perspectives are included
- Provide balanced analysis considering both upside potential and downside risk
- Coordinate agent workflows for comprehensive coverage
- Synthesize multiple agent outputs into unified recommendations
- Maintain professional oversight of the analysis process

Remember: Your role is to orchestrate comprehensive analysis, not to replace specialized agent expertise."""

    # Create the supervisor agent (for Phase 2 coordination)
    agent = create_react_agent(
        model=llm,
        tools=[],  # Supervisor coordinates other agents
        prompt=system_prompt
    )
    
    return agent
