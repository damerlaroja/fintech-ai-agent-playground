import os
import streamlit as st
import logging

logger = logging.getLogger(__name__)

# ── LLM Configuration ──────────────────────────────────────────────────────
LLM_PROVIDER   = "gemini"               # Primary provider
GEMINI_MODEL   = "gemini-2.5-flash"
GROQ_MODEL     = "llama-3.3-70b-versatile"
TEMPERATURE    = 0
MAX_TOKENS     = 4096

# ── App Configuration ───────────────────────────────────────────────────────
APP_TITLE      = "Fintech AI Agent Playground"
AGENT_VERSION  = "v0.1.0 — Market Research Agent"

# ── Provider Status Tracking ────────────────────────────────────────────────
# Tracks which provider is currently active so the UI can reflect it
_active_provider = LLM_PROVIDER


def _get_api_key(key_name: str) -> str:
    """Safely retrieves an API key from Streamlit secrets or environment."""
    try:
        return st.secrets.get(key_name, os.environ.get(key_name, ""))
    except Exception:
        return os.environ.get(key_name, "")


def _build_gemini():
    """Instantiates and returns a Gemini LLM instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = _get_api_key("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in secrets or environment")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=TEMPERATURE
    )


def _build_groq():
    """Instantiates and returns a Groq LLM instance."""
    from langchain_groq import ChatGroq
    api_key = _get_api_key("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in secrets or environment")
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=TEMPERATURE
    )


def get_llm(force_provider: str = None):
    """
    Returns the best available LLM instance using automatic fallback.

    Priority order:
    1. force_provider parameter (if explicitly passed)
    2. LLM_PROVIDER setting (default: gemini)
    3. Groq fallback (if primary provider fails)

    Never raises an exception — always returns a working LLM or raises
    a clear error only when ALL providers are unavailable.
    """
    global _active_provider

    primary = force_provider or LLM_PROVIDER

    if primary == "gemini":
        build_primary = _build_gemini
        build_fallback = _build_groq
        primary_name = "gemini"
        fallback_name = "groq"
    else:
        build_primary = _build_groq
        build_fallback = _build_gemini
        primary_name = "groq"
        fallback_name = "gemini"

    # Attempt primary provider
    try:
        llm = build_primary()
        _active_provider = primary_name
        logger.info(f"[LLM] Using primary provider: {primary_name}")
        return llm
    except Exception as e:
        logger.warning(
            f"[LLM] Primary provider '{primary_name}' failed: {e}. "
            f"Falling back to '{fallback_name}'..."
        )

    # Attempt fallback provider
    try:
        llm = build_fallback()
        _active_provider = fallback_name
        logger.warning(f"[LLM] Using fallback provider: {fallback_name}")
        return llm
    except Exception as e:
        logger.error(f"[LLM] Fallback provider '{fallback_name}' also failed: {e}")

    # Both providers failed
    raise RuntimeError(
        "All LLM providers are unavailable. "
        "Please check your GOOGLE_API_KEY and GROQ_API_KEY in "
        ".streamlit/secrets.toml or environment variables."
    )


def get_active_provider() -> str:
    """Returns the name of the currently active LLM provider."""
    return _active_provider
