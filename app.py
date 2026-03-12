import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from graph.workflow import create_workflow
from config.settings import APP_TITLE, AGENT_VERSION
from agents.risk_agent import risk_agent_node
import plotly.graph_objects as go
import plotly.express as px

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "active_phase" not in st.session_state:
    st.session_state.active_phase = "phase1"

@st.cache_resource
def get_compiled_workflow(phase="phase1"):
    return create_workflow(phase)

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
    
    # Phase navigation
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("📈 Phase 1\nMarket Research", 
                             width='stretch',
                             type="primary" if st.session_state.active_phase == "phase1" else "secondary"):
            st.session_state.active_phase = "phase1"
            st.rerun()
    with col2:
        if st.sidebar.button("📊 Phase 2\nRisk Analysis",
                             width='stretch',
                             type="primary" if st.session_state.active_phase == "phase2" else "secondary"):
            st.session_state.active_phase = "phase2"
            st.rerun()
    
    st.sidebar.caption("⚠️ Demo uses simulated sentiment & sanctions data")
    
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

# Main interface - conditional based on active phase
if st.session_state.active_phase == "phase1":
    # Phase 1: Market Research Agent (existing UI)
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
                    workflow = get_compiled_workflow("phase1")
                    
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
                            st.caption(f"🔍 Analyzed as: _{refined_query}_")
                    
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

else:
    # Phase 2: Risk Analysis Dashboard
    st.title("🔍 Risk Analysis Dashboard")
    st.caption("Portfolio risk assessment and transaction compliance analysis")
    
    # DEBUG: Test imports at page load
    try:
        from agents.risk_agent import RiskAgent
        import plotly.graph_objects as go
        import plotly.express as px
    except Exception as e:
        st.error(f"DEBUG: Import failed: {e}")
        st.stop()
    
    # Initialize session state for risk results
    if 'risk_results' not in st.session_state:
        st.session_state.risk_results = {}
    
    # Form inputs
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input("Ticker Symbol", placeholder="AAPL", key="risk_ticker")
    with col2:
        st.write("")  # Spacer
        analyze_btn = st.button("🔬 Analyze Risk", type="primary")
    
    # Results area
    if analyze_btn and ticker:
        if not ticker or not ticker.strip():
            st.warning("Please enter a valid ticker symbol.")
            st.stop()
        
        with st.spinner("Analyzing risk..."):
            try:
                # Direct RiskAgent calls (no workflow, no build_risk_agent)
                from agents.risk_agent import RiskAgent
                agent = RiskAgent()
                
                # Get structured data using exact dict keys
                vol = agent.analyze_volatility(ticker)
                sentiment = agent.analyze_sentiment_risk(ticker)
                regulatory = agent.analyze_regulatory_risk(ticker, sentiment.get("headlines", []))
                composite = agent.compute_composite_score(vol, sentiment, regulatory)
                narrative = agent.generate_risk_narrative(ticker, composite, vol, sentiment, regulatory)
                
                # Cache results
                st.session_state.risk_results[ticker.upper()] = {
                    'vol': vol, 'sentiment': sentiment, 
                    'regulatory': regulatory, 'composite': composite, 'narrative': narrative
                }
                
                # Display results using exact dict keys
                st.success(f"✅ Risk analysis complete for {ticker.upper()}")
                
                # Gauge Chart for Composite Score
                col1, col2, col3 = st.columns(3)
                with col1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = composite.get("composite", 0),
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Overall Risk Score"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgray"},
                                {'range': [40, 70], 'color': "gray"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, width='stretch')
                
                with col2:
                    st.metric("Risk Level", composite.get("label", "Unknown"))
                    st.metric("Volatility", vol.get("label", "Unknown"))
                    st.metric("Sentiment", sentiment.get("label", "Unknown"))
                
                with col3:
                    st.metric("Regulatory", regulatory.get("label", "Unknown"))
                    st.metric("News Items", sentiment.get("total_count", 0))
                    st.metric("Flags", len(regulatory.get("flags", [])))
                
                # Rolling Volatility Chart
                st.subheader("📈 Rolling Volatility Analysis")
                if vol.get("series") and vol.get("dates"):
                    fig_vol = go.Figure()
                    
                    # Add volatility line
                    fig_vol.add_trace(go.Scatter(
                        x=vol.get("dates", []),
                        y=vol.get("series", []),
                        mode='lines',
                        name='21-Day Rolling Volatility',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add threshold lines
                    fig_vol.add_hline(y=vol.get("threshold_low", 20), line_dash="dash", line_color="green", 
                                     annotation_text=f"Low Threshold: {vol.get('threshold_low', 20)}%")
                    fig_vol.add_hline(y=vol.get("threshold_high", 40), line_dash="dash", line_color="red", 
                                     annotation_text=f"High Threshold: {vol.get('threshold_high', 40)}%")
                    
                    fig_vol.update_layout(
                        title=f"{ticker.upper()} Volatility Trend",
                        xaxis_title="Date",
                        yaxis_title="Volatility (%)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_vol, width='stretch')
                else:
                    st.info("No volatility data available.")
                
                # Sentiment Analysis Chart
                st.subheader("💭 Sentiment Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    sentiment_data = {
                        'Positive': sentiment.get("positive_count", 0),
                        'Negative': sentiment.get("negative_count", 0),
                        'Neutral': sentiment.get("total_count", 0) - sentiment.get("positive_count", 0) - sentiment.get("negative_count", 0)
                    }
                    fig_sentiment = go.Figure(data=[
                        go.Bar(x=list(sentiment_data.keys()), y=list(sentiment_data.values()),
                              marker_color=['green', 'red', 'gray'])
                    ])
                    fig_sentiment.update_layout(title="News Sentiment Distribution", yaxis_title="Count")
                    st.plotly_chart(fig_sentiment, width='stretch')
                
                with col2:
                    st.metric("Sentiment Score", f"{sentiment.get('score', 0)}%")
                    st.metric("Positive News", sentiment.get("positive_count", 0))
                    st.metric("Negative News", sentiment.get("negative_count", 0))
                
                # Regulatory Risk Pills
                st.subheader("⚖️ Regulatory Compliance")
                if regulatory.get("flags"):
                    st.write("**Detected Risk Flags:**")
                    for flag in regulatory.get("flags", []):
                        st.pill(flag, icon="⚠️")
                else:
                    st.success("✅ No regulatory flags detected")
                
                st.info(regulatory.get("narrative", ""))
                
                # Risk Narrative
                st.subheader("📋 Risk Analysis Summary")
                st.info(narrative)
                
            except Exception as e:
                st.error(f"Error analyzing risk: {str(e)}")
                st.stop()
    
    # Display cached results if available
    elif ticker and ticker.upper() in st.session_state.risk_results:
        cached = st.session_state.risk_results[ticker.upper()]
        st.info(f"Showing cached results for {ticker.upper()}. Click 'Analyze Risk' to refresh.")
        
        # Brief summary of cached results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Score", f"{cached['composite'].get('composite', 0)}/100")
        with col2:
            st.metric("Risk Level", cached['composite'].get('label', 'Unknown'))
        with col3:
            st.metric("Last Updated", "Cached")

# Footer
st.markdown("---")
st.caption("⚠️ For educational and portfolio demonstration purposes only.")
