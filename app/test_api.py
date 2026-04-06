import os
from groq import Groq, APIConnectionError, AuthenticationError, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# ── Validation ─────────────────────────────────────────────────────────────────

def _get_client() -> Groq:
    """Build Groq client, failing fast if the key is missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file: GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=api_key)


# ── Core generate function ──────────────────────────────────────────────────────

def generate(
    query: str,
    context_chunks: list[dict],
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """
    Generate a grounded answer from retrieved chunks using Groq.

    Args:
        query:          The user's question.
        context_chunks: List of dicts with 'text', 'source', 'page_number'.
                        Pass your RankedResult list from the RAG pipeline.
        model:          Groq model to use.
        max_tokens:     Max tokens in the response.
        temperature:    Lower = more factual. Keep at 0.1 for RAG.

    Returns:
        Answer string grounded in the provided context.
    """
    # Build numbered context block with citations
    context = ""
    for i, chunk in enumerate(context_chunks, 1):
        src  = chunk.get("source", "unknown")
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "")
        context += f"[{i}] ({src}, p.{page}):\n{text}\n\n"

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information, say "I don't have enough information to answer that."
Always cite which source(s) you used, e.g. [1] or [2].

Context:
{context}
Question: {query}

Answer:"""

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    except AuthenticationError:
        raise EnvironmentError("Invalid GROQ_API_KEY — check your .env file.")
    except RateLimitError:
        raise RuntimeError("Groq rate limit hit — wait a moment and try again.")
    except APIConnectionError:
        raise RuntimeError("Could not reach Groq API — check your internet connection.")