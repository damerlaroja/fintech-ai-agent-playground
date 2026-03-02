import os
import streamlit as st

# ── LLM Configuration ──────────────────────────────────────────────────────
LLM_PROVIDER = "gemini"            # Switch to "groq" to change providers
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL   = "llama-3.3-70b-versatile"
TEMPERATURE  = 0
MAX_TOKENS   = 4096

# ── App Configuration ───────────────────────────────────────────────────────
APP_TITLE     = "Fintech AI Agent Playground"
AGENT_VERSION = "v0.1.0 — Market Research Agent"

def get_llm():
    """
    Returns the configured LLM instance.
    To switch providers, change LLM_PROVIDER above.
    No other files need to be modified.
    """
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = st.secrets.get("GOOGLE_API_KEY",
                  os.environ.get("GOOGLE_API_KEY", ""))
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=TEMPERATURE
        )
    elif LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        api_key = st.secrets.get("GROQ_API_KEY",
                  os.environ.get("GROQ_API_KEY", ""))
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=api_key,
            temperature=TEMPERATURE
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
