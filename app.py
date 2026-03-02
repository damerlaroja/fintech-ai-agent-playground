import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from graph.workflow import create_workflow
from config.settings import APP_TITLE, AGENT_VERSION

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

@st.cache_resource
def get_compiled_workflow():
    return create_workflow()

def get_provider_display():
    from config.settings import get_active_provider
    active = get_active_provider()
    if active == "gemini":
        return f"🟢 Gemini 2.5 Flash · Active\nLangGraph · yfinance"
    else:
        return f"🟡 Groq · Llama 3.3 70B · Active\nLangGraph · yfinance"

# Sidebar
with st.sidebar:
    st.title(f"📈 {APP_TITLE}")
    st.caption(AGENT_VERSION)
    
    st.markdown("---")
    st.markdown(f"""
    **Powered by**
    {get_provider_display()}
    """)
    
    with st.expander("💡 Try asking..."):
        st.markdown("""
        • "What is the current price and fundamentals of AAPL?"
        • "Compare MSFT, GOOGL, and AMZN by P/E ratio and market cap"
        • "Show me the 6-month price history for TSLA with key stats"
        • "What were NVDA's last 4 earnings surprises?"
        • "Find the ticker for Berkshire Hathaway and show its fundamentals"
        """)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Main chat interface
st.title("Market Research Agent")
st.caption("Ask me anything about stocks, market data, and financial analysis")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about stocks, market data, or financial analysis..."):
    # Input validation and security checks
    if not prompt or not prompt.strip():
        st.warning("Please enter a valid question.")
        st.stop()
    
    # Length validation
    if len(prompt) > 2000:
        st.error("Query too long. Please keep questions under 2000 characters.")
        st.stop()
    
    # Prompt injection detection
    injection_patterns = [
        "ignore previous instructions", "you are now", "disregard",
        "system:", "assistant:", "forget your instructions",
        "act as", "pretend to be", "roleplay as",
        "jailbreak", "bypass", "override"
    ]
    
    prompt_lower = prompt.lower()
    if any(pattern in prompt_lower for pattern in injection_patterns):
        st.error("Invalid query format. Please ask about stock market data in a clear, direct way.")
        st.stop()
    
    # Add user message to chat (wrapped in human message template)
    user_message = f"User asks: {prompt}"
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing..."):
            try:
                workflow = get_compiled_workflow()
                
                # Convert messages to LangChain format
                langchain_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                # Invoke workflow
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                response = workflow.invoke(
                    {"messages": langchain_messages},
                    config=config
                )
                
                # Check if query was refined (compare original vs preprocessed)
                original_query = prompt
                if response["messages"] and len(response["messages"]) > 0:
                    last_user_msg = None
                    for msg in reversed(response["messages"]):
                        if hasattr(msg, 'type') and msg.type == 'human':
                            last_user_msg = msg.content
                            break
                        elif isinstance(msg, HumanMessage):
                            last_user_msg = msg.content
                            break
                    
                    # Show refinement indicator if query was actually processed
                    if last_user_msg and last_user_msg != f"User asks: {original_query}" and last_user_msg != original_query:
                        refined_query = last_user_msg
                        st.caption(f"🔍 Analyzed as: *{refined_query}*")
                
                # Extract and display response
                if response["messages"]:
                    raw = response["messages"][-1].content
                    if isinstance(raw, str):
                        assistant_message = raw
                    elif isinstance(raw, list):
                        assistant_message = " ".join(
                            block.get("text", "")
                            for block in raw
                            if isinstance(block, dict) and block.get("type") == "text"
                        ).strip()
                    else:
                        assistant_message = str(raw)
                    
                    st.markdown(assistant_message)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                else:
                    error_msg = "I apologize, but I couldn't process your request. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_str = str(e).lower()
                if any(term in error_str for term in
                       ["quota", "rate limit", "429", "resource exhausted",
                        "toomanyrequests"]):
                    warning_msg = ("⚡ Rate limit reached on current provider. "
                                   "Switching to backup provider automatically — "
                                   "please resend your question.")
                    st.warning(warning_msg)
                    st.cache_resource.clear()  # Clear cache for fallback
                    st.session_state.messages.append({"role": "assistant", "content": warning_msg})
                else:
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.caption("⚠️ For educational and portfolio demonstration purposes only.")
