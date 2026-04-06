from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    print("⚠️ WARNING: SUPABASE_URL or SUPABASE_KEY is missing from .env!")
    url = "https://placeholder-url.supabase.co"
    key = "placeholder-key"

supabase = create_client(url, key)