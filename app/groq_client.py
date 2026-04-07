from __future__ import annotations

import os
from typing import Any
from groq import Groq, APIConnectionError, AuthenticationError, RateLimitError
from dotenv import load_dotenv

load_dotenv()


# ── Client ─────────────────────────────────────────────────────────────────────

def _get_groq_key() -> str | None:
    """Try st.secrets first (Streamlit Cloud), then env vars (local .env)."""
    try:
        import streamlit as st
        val = st.secrets.get("GROQ_API_KEY")
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


def _get_client() -> Groq:
    """Build Groq client, failing fast if the key is missing."""
    api_key = _get_groq_key()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file (local) or Streamlit Cloud Secrets."
        )
    timeout_secs = float(os.getenv("GROQ_TIMEOUT_SECS", "120"))
    max_retries = int(os.getenv("GROQ_MAX_RETRIES", "5"))
    return Groq(api_key=api_key, timeout=timeout_secs, max_retries=max_retries)


def _chunk_fields(chunk: Any) -> tuple[str, str, Any, dict]:
    """
    Accept either:
    - RankedResult-like objects: have .text and .metadata
    - dict chunks: {'text','source','page_number',...}
    Returns: (text, source, page_number, metadata)
    """
    if isinstance(chunk, dict):
        text = str(chunk.get("text", ""))
        source = str(chunk.get("source", "unknown"))
        page = chunk.get("page_number", "?")
        meta = {k: v for k, v in chunk.items() if k not in {"text", "source", "page_number"}}
        return text, source, page, meta

    # RankedResult path (or anything with these attrs)
    text = str(getattr(chunk, "text", ""))
    meta = getattr(chunk, "metadata", {}) or {}
    source = str(meta.get("source", "unknown"))
    page = meta.get("page_number", "?")
    return text, source, page, meta


# ── Query rewriting ────────────────────────────────────────────────────────────

def detect_intent(query: str) -> str:
    """
    Rapidly classify if the user wants a broad document summary vs a specific search.
    Returns: "summary" or "search"
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"""Classify the intent of the following user query for a document retrieval system.
If the query asks for a broad overview, summary, or what the document is about, reply exactly with: summary
If the query asks a specific question, asks for facts, or isn't a clear summary request, reply exactly with: search

Query: "{query}"
Reply ONLY with "summary" or "search" (no punctuation, no quotes)."""
                }
            ],
            max_tokens=5,
            temperature=0.0,
        )
        ans = response.choices[0].message.content.strip().lower()
        if "summary" in ans:
            return "summary"
        return "search"
    except Exception:
        return "search"


def rewrite_query(query: str) -> str:
    """
    Rewrite vague or conversational queries into specific searchable ones.

    Handles follow-ups like:
      - "explain more"          → "What are the key concepts covered in these documents?"
      - "tell me more about it" → "What additional details are provided about AI?"
      - "can you elaborate"     → "What detailed information is available on this topic?"

    Returns the original query unchanged if it's already specific enough.
    """
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a query rewriting assistant for a document retrieval system.

Your job is to rewrite queries to be more specific and searchable.

Rules:
- If the query is vague (e.g. "explain more", "tell me more", "elaborate", "what else"), 
  expand it into a specific, detailed question suitable for document search.
- If the query is already specific and clear, return it unchanged.
- Return ONLY the rewritten query — no explanation, no preamble, no quotes.

Examples:
  "explain more"              → "What are the main topics and key concepts covered in these documents?"
  "can you elaborate"         → "What detailed information is provided about the main subject?"
  "tell me more about that"   → "What additional details and context are available on this topic?"
  "what is generative AI"     → "what is generative AI"
  "what are the risks of AI"  → "what are the risks of AI"

Original query: {query}
Rewritten query:"""
                }
            ],
            max_tokens=150,
            temperature=0.3,
        )
        rewritten = response.choices[0].message.content.strip()

        # Safety check — if rewriting returns something too long or weird, use original
        if not rewritten or len(rewritten) > 300:
            return query

        return rewritten

    except Exception:
        # If rewriting fails for any reason, fall back to original query silently
        return query


# ── Query expansion ────────────────────────────────────────────────────────────

def expand_query(query: str) -> list[str]:
    """
    Generate multiple variations of a query to improve retrieval recall.

    A single query may miss relevant chunks due to wording differences.
    Running 3 variations and merging results significantly improves recall.

    Returns a list starting with the original query, followed by variations.
    """
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"""Generate 3 different ways to search for the answer to this question.
Each variation should use different wording but ask for the same information.
Return only the 3 queries, one per line, no numbering, no explanation.

Question: {query}"""
                }
            ],
            max_tokens=200,
            temperature=0.7,
        )

        raw = response.choices[0].message.content.strip()
        variants = [v.strip() for v in raw.split("\n") if v.strip()][:3]

        return [query] + variants

    except Exception:
        # If expansion fails, just use the original
        return [query]


# ── Core generation ────────────────────────────────────────────────────────────

def generate(
    query: str,
    context_chunks: list[Any],
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 2048,
    temperature: float = 0.1,
    chat_history: list[dict] | None = None,
    *,
    stream: bool = False,
    is_summary_mode: bool = False,
) -> str:
    """
    Generate one precise grounded answer from retrieved chunks using Groq.

    Args:
        query:          The user's question.
        context_chunks: List of retrieved chunks. Accepts either RankedResult objects
                        (from reranker) or dict chunks with 'text'/'source'/'page_number'.
        model:          Groq model to use.
        max_tokens:     Max tokens in the response.
        temperature:    Lower = more factual (recommended for RAG).
        chat_history:   Optional past conversation turns for memory.
                        Format: [{"role": "user/assistant", "content": "..."}]

    Returns:
        Detailed answer string grounded in the provided context.
    """
    if not context_chunks:
        return "I don't have enough information to answer that question."

    # Build numbered context block with source citations
    context = ""
    for i, chunk in enumerate(context_chunks, 1):
        text, src, page, _ = _chunk_fields(chunk)
        context += f"[{i}] ({src}, p.{page}):\n{text}\n\n"

    if is_summary_mode:
        system_prompt = """You are an expert document synthesizer. Your job is to summarize the core topics, themes, and goals of these extracted document chunks into one comprehensive overview.

Your rules:
- Provide a clear, well-structured, and comprehensive summary.
- Focus on the high-level themes, not minute facts.
- Do NOT say "I don't have enough information" — generate the best possible summary from the provided text.
- Optionally use bullet points for key topics."""
    else:
        system_prompt = """You are a helpful assistant answering questions using only the provided context chunks.

Your rules:
- Answer ONLY using the context provided — never use outside knowledge
- Write ONE precise answer (no bullet points, no numbered lists), ideally 1 short paragraph (max 3 sentences)
- If multiple sources mention the topic, synthesize them into a single cohesive answer
- Always cite which source(s) you used inline, e.g. [1] or [2] or [1][2]
- If the context does not contain enough information, say exactly: "I don't have enough information to answer that."
- Never make up information not present in the context"""

    user_message = f"""Context:
{context}

Question: {query}

Answer:"""

    # Build messages — system + optional history + current question
    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        # Only keep last 6 turns to avoid hitting context limits
        messages.extend(chat_history[-6:])

    messages.append({"role": "user", "content": user_message})

    if stream:
        collected: list[str] = []
        for tok in generate_stream(
            query,
            context_chunks,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            chat_history=chat_history,
        ):
            collected.append(tok)
            print(tok, end="", flush=True)
        print()
        return "".join(collected).strip()

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    except AuthenticationError:
        raise EnvironmentError(
            "Invalid GROQ_API_KEY — check your .env file."
        )
    except RateLimitError:
        raise RuntimeError(
            "Groq rate limit hit — wait a moment and try again."
        )
    except APIConnectionError:
        raise RuntimeError(
            "Could not reach Groq API — check your internet connection."
        )


# ── Streaming generation ───────────────────────────────────────────────────────

def generate_stream(
    query: str,
    context_chunks: list[Any],
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 2048,
    temperature: float = 0.1,
    chat_history: list[dict] | None = None,
    is_summary_mode: bool = False,
):
    """
    Streaming version of generate() — yields tokens as they arrive.

    Use this in Streamlit with st.write_stream() for a live typing effect.

    Usage in Streamlit:
        with st.chat_message("assistant"):
            answer = st.write_stream(
                generate_stream(query, chunks, chat_history=history)
            )

    Usage in terminal:
        for token in generate_stream(query, chunks):
            print(token, end="", flush=True)
        print()
    """
    if not context_chunks:
        yield "I don't have enough information to answer that question."
        return

    context = ""
    for i, chunk in enumerate(context_chunks, 1):
        text, src, page, _ = _chunk_fields(chunk)
        context += f"[{i}] ({src}, p.{page}):\n{text}\n\n"

    if is_summary_mode:
        system_prompt = """You are an expert document synthesizer. Your job is to summarize the core topics, themes, and goals of these extracted document chunks into one comprehensive overview.

Your rules:
- Provide a clear, well-structured, and comprehensive summary.
- Focus on the high-level themes, not minute facts.
- Do NOT say "I don't have enough information" — generate the best possible summary from the provided text.
- Optionally use bullet points for key topics."""
    else:
        system_prompt = """You are a helpful assistant answering questions using only the provided context chunks.

Your rules:
- Answer ONLY using the context provided — never use outside knowledge
- Write ONE precise answer (no bullet points, no numbered lists), ideally 1 short paragraph (max 3 sentences)
- If multiple sources mention the topic, synthesize them into a single cohesive answer
- Always cite which source(s) you used inline, e.g. [1] or [2] or [1][2]
- If the context does not contain enough information, say exactly: "I don't have enough information to answer that."
- Never make up information not present in the context"""

    user_message = f"""Context:
{context}

Question: {query}

Answer:"""

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": user_message})

    client = _get_client()

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    except AuthenticationError:
        yield "\n\n⚠️ Invalid GROQ_API_KEY — check your .env file."
    except RateLimitError:
        yield "\n\n⚠️ Groq rate limit hit — wait a moment and try again."
    except APIConnectionError:
        yield "\n\n⚠️ Could not reach Groq API — check your internet connection."
    except Exception as e:
        yield f"\n\n⚠️ Groq Error: {e}"