from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from agents.market_agent import build_market_agent
from tools.query_preprocessor import preprocess_query
from config.settings import get_llm


def query_preprocessor_node(state: MessagesState):
    """
    Query preprocessor node that normalizes company names and reframes investment questions.
    
    Args:
        state: Current conversation state containing messages
        
    Returns:
        Updated state with preprocessed query or off-topic response
    """
    last_message = state["messages"][-1]
    raw_query = last_message.content
    
    # Remove the "User asks: " prefix if present
    if raw_query.startswith("User asks: "):
        raw_query = raw_query[11:]
    
    # Simple optimization check: skip preprocessor for short ticker-based queries
    words = raw_query.strip().split()
    has_ticker_pattern = any(
        word.isupper() and 2 <= len(word) <= 5 and word.replace('-', '').replace('.', '').isalpha()
        for word in words
    )
    
    if len(words) <= 10 and has_ticker_pattern:
        # Already optimized query - skip preprocessor
        return {"messages": state["messages"]}
    
    llm = get_llm()
    refined_query = preprocess_query(raw_query, llm)
    
    if refined_query == "OFF_TOPIC":
        return {"messages": [AIMessage(content=(
            "I'm specialized in stock market research and financial analysis. "
            "I can help you with:\n\n"
            "- 📈 **Stock prices and fundamentals** (e.g., 'Show me AAPL fundamentals')\n"
            "- 📊 **Compare stocks** (e.g., 'Compare MSFT, GOOGL, and AMZN')\n"
            "- 💰 **Earnings history** (e.g., 'NVDA last 4 earnings surprises')\n"
            "- 📉 **Price history** (e.g., 'TSLA 6-month performance')\n"
            "- 🔍 **Ticker lookup** (e.g., 'What is Berkshire Hathaway's ticker?')\n\n"
            "What would you like to research?"
        ))]}
    
    # Replace the last message with the refined query
    updated_messages = list(state["messages"][:-1]) + [
        HumanMessage(content=refined_query)
    ]
    return {"messages": updated_messages}


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
    2. Preprocesses query (company name resolution, question reframing)
    3. Processes through market research node
    4. Ends with agent response
    
    The workflow includes MemorySaver for persistent conversation history.
    
    Returns:
        Compiled LangGraph workflow ready for execution
    """
    # Initialize the StateGraph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("query_preprocessor", query_preprocessor_node)
    workflow.add_node("market_research", market_research_node)
    
    # Define the flow: START → query_preprocessor → market_research → END
    workflow.add_edge(START, "query_preprocessor")
    workflow.add_edge("query_preprocessor", "market_research")
    workflow.add_edge("market_research", END)
    
    # ── PHASE 2: Register risk_agent_node here ──────────────────────────────
    # Example for Phase 2:
    # workflow.add_node("risk_agent", risk_agent_node)
    # workflow.add_edge("market_research", "risk_agent")
    # workflow.add_edge("risk_agent", END)
    
    # ── PHASE 3: Register supervisor_node and report_agent_node here ─────────
    # Example for Phase 3:
    # workflow.add_node("supervisor", supervisor_node)
    # workflow.add_node("report_agent", report_agent_node)
    # workflow.add_edge("risk_agent", "supervisor")
    # workflow.add_edge("supervisor", "report_agent")
    # workflow.add_edge("report_agent", END)
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
