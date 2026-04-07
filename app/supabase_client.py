import os
from dotenv import load_dotenv

load_dotenv()

# ── Read secrets: works locally (.env) AND on Streamlit Cloud (st.secrets) ──────
def _get_secret(key: str) -> str | None:
    """Try st.secrets first (Streamlit Cloud), then env vars (local)."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key)

supabase = None

try:
    from supabase import create_client
    
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_KEY")
    
    if url and key:
        supabase = create_client(url, key)
        print("✅ Supabase connected")
    else:
        print("ℹ️ Supabase credentials not found - using app without session persistence")
        
except ImportError:
    print("ℹ️ Supabase library not installed - session features disabled")
except Exception as e:
    print(f"ℹ️ Supabase connection unavailable: {e}")