import os
from pathlib import Path
from app.supabase_client import supabase

BUCKET_NAME = "rag-data"

def upload_indexes(index_dir: Path):
    """Upload all FAISS and BM25 index files to Supabase bucket."""
    try:
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
        print(f"⚠️ Cloud sync failed (does bucket '{BUCKET_NAME}' exist?): {e}")

def download_indexes(index_dir: Path) -> bool:
    """Download index files from Supabase if they exist. Returns True if FAISS was found."""
    index_dir.mkdir(parents=True, exist_ok=True)
    success = False
    try:
        for file_name in ["vector_store.faiss", "vector_store.json", "bm25_store.bm25.json"]:
            from postgrest.exceptions import APIError
            try:
                # supabase python SDK storage download returns bytes if found
                # It evaluates to an error if not found. We will catch it.
                response = supabase.storage.from_(BUCKET_NAME).download(file_name)
                if response:
                    with open(index_dir / file_name, "wb") as f:
                        f.write(response)
                    print(f"⬇️ Downloaded {file_name} from cloud.")
                    if file_name == "vector_store.faiss":
                        success = True
            except Exception as inner_e:
                # typically happens if the file doesn't exist in the bucket yet
                pass
        return success
    except Exception as e:
        print(f"ℹ️ Cloud indices bucket fetch failed: {e}")
        return False
