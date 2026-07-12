"""
Loads secrets from .env locally, or from Streamlit's secrets manager
when deployed on Streamlit Community Cloud — so the rest of the code
can just use os.getenv() either way without caring which environment
it's running in.
"""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    for key in ["GEMINI_API_KEY", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"]:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # st.secrets not available (e.g. running outside Streamlit) — fine, .env already loaded