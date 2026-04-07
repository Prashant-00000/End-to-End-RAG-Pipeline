import os
from pathlib import Path

BUCKET_NAME = "rag-data"

def upload_indexes(index_dir: Path):
    """Upload all FAISS and BM25 index files to Supabase bucket."""
    try:
        from app.supabase_client import supabase
        
        for file_name in ["vector_store.faiss", "vector_store.json", "bm25_store.bm25.json"]:
            file_path = index_dir / file_name
            if file_path.exists():
                with open(file_path, "rb") as f:
                    supabase.storage.from_(BUCKET_NAME).upload(
                        file_name,
                        f,
                        file_options={"upsert": "true"}
                    )
        print("☁️ Successfully synced FAISS indices to Supabase Storage.")
    except Exception as e:
        print(f"⚠️ Cloud sync failed: {e}")

def download_indexes(index_dir: Path) -> bool:
    """Download index files from Supabase if they exist. Returns True if FAISS was found."""
    try:
        from app.supabase_client import supabase
        
        index_dir.mkdir(parents=True, exist_ok=True)
        success = False
        
        for file_name in ["vector_store.faiss", "vector_store.json", "bm25_store.bm25.json"]:
            try:
                response = supabase.storage.from_(BUCKET_NAME).download(file_name)
                if response:
                    with open(index_dir / file_name, "wb") as f:
                        f.write(response)
                    print(f"⬇️ Downloaded {file_name} from cloud.")
                    if file_name == "vector_store.faiss":
                        success = True
            except Exception:
                # File doesn't exist or error downloading
                pass
        
        return success
    except Exception as e:
        print(f"ℹ️ Cloud indices unavailable: {e}")
        return False
