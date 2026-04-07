import numpy as np
import re
from typing import Optional

# ── Lazy-load model for semantic chunking ──────────────────────────────────────

_model: Optional[object] = None
_model_available: bool = False

def get_chunking_model(model_name: str = "BAAI/bge-small-en"):
    """
    Lazy-load model only when semantic chunking is needed.
    Falls back to fixed chunking if model unavailable.
    """
    global _model, _model_available
    
    if _model_available:
        return _model
    
    try:
        from sentence_transformers import SentenceTransformer
        print(f"🔧 Loading chunking model...")
        _model = SentenceTransformer(model_name)
        _model_available = True
        return _model
    except Exception as e:
        print(f"⚠️ Semantic chunking unavailable: {e}")
        print("📊 Using fixed chunking instead")
        _model_available = False
        return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# ─────────────────────────────────────────────────────────────────────────────

def fixed_chunking(text, chunk_size=300, overlap=50):
    # Clean up excessive whitespace and newlines
    text = " ".join(text.split())
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def semantic_chunking(text, threshold=0.6):
    """Chunk text based on semantic similarity between sentences."""
    # Try to load model
    model = get_chunking_model()
    
    # If model load fails, fall back to fixed chunking
    if model is None:
        print("⚠️ Falling back to fixed chunking...")
        return fixed_chunking(text, chunk_size=300, overlap=50)
    
    # Clean text before chunking
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    print(f"📊 Total sentences: {len(sentences)}")
    
    if not sentences or len(sentences) < 2:
        return [text]
    
    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = sentences[0]
    
    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])
        
        if sim < threshold:
            chunks.append(current_chunk)
            current_chunk = sentences[i]
        else:
            current_chunk += " " + sentences[i]
    
    chunks.append(current_chunk)
    
    print(f"📦 Semantic chunks created: {len(chunks)}")
    return chunks


def hybrid_chunking(text, max_words=300, overlap=50, semantic_threshold=0.6):
    """
    Combines fixed + semantic chunking:
    1. First splits into fixed chunks (fast, reliable)
    2. Then applies semantic chunking within each fixed chunk
    3. Result: semantically coherent chunks that never exceed size limits
    """
    print("🔀 Running hybrid chunking...")
    
    # Step 1: fixed chunking to get manageable pieces
    fixed_chunks = fixed_chunking(text, chunk_size=max_words, overlap=overlap)
    
    final_chunks = []
    
    # Step 2: semantic chunking within each fixed chunk
    for fixed_chunk in fixed_chunks:
        if len(fixed_chunk.split()) > 30:
            semantic_sub_chunks = semantic_chunking(fixed_chunk, threshold=semantic_threshold)
            final_chunks.extend(semantic_sub_chunks)
        else:
            final_chunks.append(fixed_chunk)
    
    print(f"✅ Hybrid chunking complete: {len(final_chunks)} final chunks")
    return final_chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))