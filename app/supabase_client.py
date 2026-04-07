import os
from dotenv import load_dotenv

load_dotenv()

try:
    from supabase import create_client
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        print("⚠️ WARNING: SUPABASE_URL or SUPABASE_KEY is missing!")
        supabase = None
    else:
        supabase = create_client(url, key)
        print("✅ Supabase connected")
except Exception as e:
    print(f"⚠️ Supabase connection failed: {e}")
    supabase = None