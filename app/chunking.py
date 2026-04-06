from sentence_transformers import SentenceTransformer
import numpy as np
import re

model = SentenceTransformer("BAAI/bge-small-en")

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