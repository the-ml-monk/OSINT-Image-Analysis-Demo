"""
Which AI model answers the image questions — Gemini if GEMINI_API_KEY is set,
otherwise Groq (GROQ_API_KEY). Both are called through the same
OpenAI-compatible chat endpoint, so the rest of the code does not change.

Gemini Flash "thinks" before it answers, and that thinking counts against
max_tokens — with the short limits used here a briefing could come back cut off
or empty. reasoning_effort "none" switches thinking off for these calls.
"""

import os
import requests

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if GEMINI_API_KEY:
    PROVIDER = "Gemini"
    LLM_KEY = GEMINI_API_KEY
    LLM_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    LLM_EXTRA = {"reasoning_effort": "none"}
    _MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/openai/models"
elif GROQ_API_KEY:
    PROVIDER = "Groq"
    LLM_KEY = GROQ_API_KEY
    LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
    LLM_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    LLM_EXTRA = {}
    _MODELS_URL = "https://api.groq.com/openai/v1/models"
else:
    PROVIDER = None
    LLM_KEY = ""
    LLM_URL = ""
    LLM_MODEL = ""
    LLM_EXTRA = {}
    _MODELS_URL = ""


def check_llm() -> tuple:
    """(ok, message) — asks the provider whether the key works and the model is offered."""
    if not PROVIDER:
        return False, "not configured"
    try:
        r = requests.get(_MODELS_URL, headers={"Authorization": f"Bearer {LLM_KEY}"}, timeout=15)
        if r.status_code != 200:
            return False, f"key rejected ({r.status_code})"
        ids = {m.get("id", "").split("/")[-1] for m in r.json().get("data", [])}
        if LLM_MODEL.split("/")[-1] not in ids:
            return False, f"model {LLM_MODEL} not offered"
        return True, f"{LLM_MODEL}"
    except Exception as e:
        return False, f"unreachable ({type(e).__name__})"
