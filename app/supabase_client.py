import os
from dotenv import load_dotenv

load_dotenv()

supabase = None

try:
    from supabase import create_client
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if url and key:
        supabase = create_client(url, key)
        print("✅ Supabase connected")
    else:
        print("ℹ️ Supabase credentials not found - using app without session persistence")
        
except ImportError:
    print("ℹ️ Supabase library not installed - session features disabled")
except Exception as e:
    print(f"ℹ️ Supabase connection unavailable: {e}")