from app.supabase_client import supabase

def get_session(name: str):
    if not supabase:
        return {"name": name, "history": [], "chat_history": []}
    
    try:
        response = supabase.table("sessions").select("*").eq("name", name).execute()
        
        if response.data:
            return response.data[0]
        else:
            # create new session if not exists
            supabase.table("sessions").insert({
                "name": name,
                "history": [],
                "chat_history": []
            }).execute()
            
            return {
                "name": name,
                "history": [],
                "chat_history": []
            }
    except Exception as e:
        print(f"⚠️ Session get failed: {e}")
        return {"name": name, "history": [], "chat_history": []}

def update_session(name: str, history: list, chat_history: list):
    """Create or update session in database."""
    if not supabase:
        return
    
    try:
        # Try to update first
        response = supabase.table("sessions").select("name").eq("name", name).execute()
        
        if response.data:
            # Session exists, update it
            supabase.table("sessions").update({
                "history": history,
                "chat_history": chat_history
            }).eq("name", name).execute()
        else:
            # Session doesn't exist, create it
            supabase.table("sessions").insert({
                "name": name,
                "history": history,
                "chat_history": chat_history
            }).execute()
    except Exception as e:
        print(f"⚠️ Session update failed: {e}")

def load_all_sessions() -> list[dict]:
    if not supabase:
        return []
    
    try:
        response = supabase.table("sessions").select("name, history, chat_history").order("name", desc=False).execute()
        return response.data
    except Exception as e:
        print(f"Error loading sessions: {e}")
        return []

def delete_session(name: str):
    if not supabase:
        return
    
    try:
        supabase.table("sessions").delete().eq("name", name).execute()
    except Exception as e:
        print(f"⚠️ Session delete failed: {e}")

def clear_all_sessions():
    if not supabase:
        return
    
    try:
        supabase.table("sessions").delete().neq("name", "nothing_in_particular").execute()
    except Exception as e:
        print(f"⚠️ Clear sessions failed: {e}")
