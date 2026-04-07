import os
from pathlib import Path

BUCKET_NAME = "rag-data"

def upload_indexes(index_dir: Path):
    """Upload all FAISS and BM25 index files to Supabase bucket."""
    try:
        from app.supabase_client import supabase
        
        if not supabase:
            print("ℹ️ Supabase not available - skipping cloud sync")
            return
        
        for file_name in ["vector_store.faiss", "vector_store.json", "bm25_store.bm25.json"]:
            file_path = index_dir / file_name
            if file_path.exists():
                with open(file_path, "rb") as f:
                    supabase.storage.from_(BUCKET_NAME).upload(
                        file_name,
                        f,
                        file_options={"upsert": "true"}
                    )
        print("☁️ Cloud sync complete")
    except Exception as e:
        print(f"ℹ️ Cloud sync skipped: {e}")

def download_indexes(index_dir: Path) -> bool:
    """Download index files from Supabase if they exist."""
    try:
        from app.supabase_client import supabase
        
        if not supabase:
            print("ℹ️ Supabase not available")
            return False
        
        index_dir.mkdir(parents=True, exist_ok=True)
        success = False
        
        for file_name in ["vector_store.faiss", "vector_store.json", "bm25_store.bm25.json"]:
            try:
                response = supabase.storage.from_(BUCKET_NAME).download(file_name)
                if response:
                    with open(index_dir / file_name, "wb") as f:
                        f.write(response)
                    if file_name == "vector_store.faiss":
                        success = True
            except Exception:
                pass
        
        return success
    except Exception as e:
        print(f"ℹ️ Cloud download unavailable: {e}")
        return False
