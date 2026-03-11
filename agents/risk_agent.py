from __future__ import annotations
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState

def build_risk_agent():
    """Build the Risk Scoring Agent.
    
    Returns a compiled LangGraph agent for risk analysis.
    TODO: Implement in Prompt 2
    """
    # TODO: Implement in Prompt 2
    pass

def risk_agent_node(state: MessagesState) -> dict:
    """LangGraph node for risk agent processing.
    
    Args:
        state: Current conversation state
    
    Returns:
        Updated state with risk analysis response
    """
    # TODO: Implement in Prompt 2
    return {"messages": [AIMessage(content=(
        "Risk Analysis Agent is under construction. "
        "Coming soon: VaR calculations, beta analysis, "
        "portfolio risk scoring, and transaction compliance checks."
    ))]}
