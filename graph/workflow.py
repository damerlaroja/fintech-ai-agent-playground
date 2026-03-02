from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from agents.market_agent import build_market_agent


def market_research_node(state: MessagesState):
    """
    Market research node that processes user queries using the market agent.
    
    Args:
        state: Current conversation state containing messages
        
    Returns:
        Updated state with the agent's response
    """
    # Get the market agent
    agent = build_market_agent()
    
    # Process the current message through the agent
    messages = state["messages"]
    response = agent.invoke({"messages": messages})
    
    return {"messages": response["messages"]}


def create_workflow():
    """
    Create and compile the LangGraph workflow for market research.
    
    This function builds a linear workflow that:
    1. Starts with user input
    2. Processes through market research node
    3. Ends with agent response
    
    The workflow includes MemorySaver for persistent conversation history.
    
    Returns:
        Compiled LangGraph workflow ready for execution
    """
    # Initialize the StateGraph
    workflow = StateGraph(MessagesState)
    
    # Add the market research node
    workflow.add_node("market_research_node", market_research_node)
    
    # Define the linear flow: START → market_research_node → END
    workflow.add_edge(START, "market_research_node")
    workflow.add_edge("market_research_node", END)
    
    # ── PHASE 2: Register risk_agent_node here ──────────────────────────────
    # Example for Phase 2:
    # workflow.add_node("risk_agent_node", risk_agent_node)
    # workflow.add_edge("market_research_node", "risk_agent_node")
    # workflow.add_edge("risk_agent_node", END)
    
    # ── PHASE 3: Register supervisor_node and report_agent_node here ─────────
    # Example for Phase 3:
    # workflow.add_node("supervisor_node", supervisor_node)
    # workflow.add_node("report_agent_node", report_agent_node)
    # workflow.add_conditional_edges(
    #     "supervisor_node",
    #     route_to_agents,
    #     {
    #         "market": "market_research_node",
    #         "risk": "risk_agent_node",
    #         "report": "report_agent_node",
    #         "end": END
    #     }
    # )
    
    # Add memory for conversation persistence
    memory = MemorySaver()
    
    # Compile the workflow with memory
    compiled_workflow = workflow.compile(checkpointer=memory)
    
    return compiled_workflow
