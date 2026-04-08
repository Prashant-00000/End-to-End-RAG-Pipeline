from __future__ import annotations

# Configure logging FIRST - before any imports
import logging
import os

logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.models').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Initialize session state FIRST before any imports that might use it
import streamlit as st
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_name" not in st.session_state:
    st.session_state.session_name = None

from pathlib import Path

from app.bm25_store import BM25Store
from app.groq_client import generate_stream, rewrite_query, detect_intent
from app.reranker import RankedResult
from app.vector_store import VectorStore
from main import (
    INDEX_DIR, DATA_DIR, build_indexes, load_indexes, UPLOAD_TASKS,
    async_add_document, remove_document_fast, query_pipeline,
)

# ── Persistent session storage (Supabase) ─────────────────────────────────────

from app.db import get_session, update_session, load_all_sessions, delete_session, clear_all_sessions

def save_current_session() -> None:
    """Save the active session to Supabase."""
    if not st.session_state.history:
        return
    name = st.session_state.session_name or "Untitled chat"
    update_session(name, st.session_state.history, st.session_state.chat_history)


from app.cloud_storage import download_indexes

# ── Index loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def check_and_get_indexes() -> tuple[VectorStore, BM25Store]:
    faiss_marker = INDEX_DIR / "vector_store.faiss"

    if faiss_marker.exists():
        return load_indexes()

    with st.spinner("⬇️ Checking cloud storage for indexes..."):
        if download_indexes(INDEX_DIR):
            return load_indexes()

    with st.spinner("🔨 Building indexes from PDFs... (this may take a while)"):
        return build_indexes()


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="centered",
)

st.markdown("""
<style>
    .source-pill {
        display: inline-block;
        background: rgba(255,107,53,0.15);
        border: 1px solid rgba(255,107,53,0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        color: #FF6B35;
        margin: 2px 4px 2px 0;
    }
</style>
""", unsafe_allow_html=True)

vs, bm25 = check_and_get_indexes()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 RAG System")
    st.caption("Ask questions about your documents")
    st.markdown("---")

    @st.fragment(run_every="1s")
    def render_background_tasks():
        active   = {k: v for k, v in UPLOAD_TASKS.items() if v["status"] != "Done" and not v["status"].startswith("Error")}
        finished = {k: v for k, v in UPLOAD_TASKS.items() if v["status"] == "Done"}
        errors   = {k: v for k, v in UPLOAD_TASKS.items() if v["status"].startswith("Error")}

        for name in list(finished.keys()):
            st.toast(f"✅ Finished indexing {name}!")
            check_and_get_indexes.clear()
            del UPLOAD_TASKS[name]

        for name, info in list(errors.items()):
            st.toast(f"❌ Failed to index {name}: {info['status']}")
            del UPLOAD_TASKS[name]

        if active:
            st.markdown("### ⏳ Background Tasks")
            for name, info in active.items():
                st.progress(info["progress"], text=f"{name[:20]}: {info['status']}")
            st.markdown("---")

    render_background_tasks()

    if st.button("➕  New Chat", use_container_width=True):
        save_current_session()
        st.session_state.history      = []
        st.session_state.chat_history = []
        st.session_state.session_name = None
        st.rerun()

    st.markdown("---")

    st.markdown("### 📚 Documents")
    DATA_DIR.mkdir(exist_ok=True)
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        st.caption("No PDFs found.")
    else:
        for pdf in pdf_files:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.caption(pdf.name)
            with col2:
                if st.button("🗑", key=f"del_pdf_{pdf.name}", help="Remove this document"):
                    pdf.unlink()
                    with st.spinner("Removing document..."):
                        remove_document_fast(pdf, vs, bm25)
                        check_and_get_indexes.clear()
                    st.rerun()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        file_path = DATA_DIR / uploaded_file.name
        if not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            async_add_document(file_path, vs, bm25)
            st.rerun()

    st.markdown("---")

    st.markdown("### 🎯 Targeting")
    doc_options = ["All Documents"] + [p.name for p in pdf_files] if pdf_files else ["All Documents"]
    selected_doc = st.selectbox("Search in:", doc_options, label_visibility="collapsed")

    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    show_sources = st.toggle("Show retrieved sources", value=False)
    show_rewrite = st.toggle("Show query rewrites",    value=False)

    st.markdown("---")

    st.markdown("### 🕘 Chat history")

    all_sessions = load_all_sessions()

    if not all_sessions:
        st.caption("No saved chats yet.\nHistory saves automatically.")
    else:
        for session in all_sessions:
            col1, col2 = st.columns([5, 1])
            label = session["name"][:28] + ("..." if len(session["name"]) > 28 else "")

            with col1:
                if st.button(
                    f"💬 {label}",
                    key=f"load_{session['name']}",
                    use_container_width=True,
                ):
                    save_current_session()
                    st.session_state.history      = session["history"]
                    st.session_state.chat_history = session["chat_history"]
                    st.session_state.session_name = session["name"]
                    st.rerun()

            with col2:
                if st.button("🗑", key=f"del_{session['name']}"):
                    delete_session(session["name"])
                    if st.session_state.session_name == session["name"]:
                        st.session_state.history      = []
                        st.session_state.chat_history = []
                        st.session_state.session_name = None
                    st.rerun()

    st.markdown("---")

    if st.button("🚨  Clear all history", use_container_width=True, help="WARNING: This will permanently delete all saved chats."):
        clear_all_sessions()
        st.session_state.history      = []
        st.session_state.chat_history = []
        st.session_state.session_name = None
        st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────────

st.title("💬 Ask your documents")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if (
            show_sources
            and msg["role"] == "assistant"
            and msg.get("sources")
        ):
            with st.expander("📎 Retrieved sources", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<span class="source-pill">📄 [{i}] '
                        f'{src["source"]} — p.{src["page"]} '
                        f'(score: {src["score"]:.2f})</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(src["preview"])


# ── Handle new query ───────────────────────────────────────────────────────────

query = st.chat_input("Ask a question about your documents...")

if query:
    with st.chat_message("user"):
        st.write(query)
    st.session_state.history.append({"role": "user", "content": query})

    if st.session_state.session_name is None:
        import uuid
        uid = uuid.uuid4().hex[:4]
        st.session_state.session_name = f"{query[:30]} - {uid}"

    try:
        intent = detect_intent(query)
        is_summary_mode = (intent == "summary")
    except (EnvironmentError, Exception):
        # Fallback to search mode if Groq API is not available
        is_summary_mode = False

    if is_summary_mode:
        if show_rewrite:
            st.caption("📑 Document Summary Mode activated")
        rewritten = query
    else:
        try:
            rewritten = rewrite_query(query)
            if show_rewrite and rewritten != query:
                st.caption(f"🔄 Searching for: *{rewritten}*")
        except (EnvironmentError, Exception):
            # Fallback to original query if rewriting fails
            rewritten = query

    metadata_filter = None
    if selected_doc != "All Documents":
        metadata_filter = {"source": selected_doc}

    with st.spinner("🔍 Extracting documents..." if is_summary_mode else "🔍 Searching documents..."):
        results: list[RankedResult] = query_pipeline(
            rewritten,
            vs,
            bm25,
            is_summary_mode=is_summary_mode,
            metadata_filter=metadata_filter,
        )

    chunks = [
        {
            "text":        r.text,
            "source":      r.metadata.get("source", "unknown"),
            "page_number": r.metadata.get("page_number", "?"),
        }
        for r in results
    ]

    sources = [
        {
            "source":  r.metadata.get("source", "unknown"),
            "page":    r.metadata.get("page_number", "?"),
            "score":   r.score,
            "preview": r.text[:200] + "...",
        }
        for r in results
    ]

    collected: list[str] = []

    def token_gen():
        try:
            for tok in generate_stream(
                query=query,
                context_chunks=chunks,
                chat_history=st.session_state.chat_history,
                is_summary_mode=is_summary_mode,
            ):
                collected.append(tok)
                yield tok
        except (EnvironmentError, Exception) as e:
            # Fallback response when Groq API is not available
            fallback_msg = "⚠️ Groq API not configured. Please add GROQ_API_KEY to Streamlit Cloud secrets to enable AI features. For now, here are the retrieved documents."
            yield fallback_msg
            collected.append(fallback_msg)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(token_gen())

            if show_sources and sources:
                with st.expander("📎 Retrieved sources", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f'<span class="source-pill">📄 [{i}] '
                            f'{src["source"]} — p.{src["page"]} '
                            f'(score: {src["score"]:.2f})</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(src["preview"])
            answer = "".join(collected).strip()

        except Exception as e:
            error_msg = str(e).lower()
            if "groq_api_key" in error_msg or "not found" in error_msg:
                st.warning("⚠️ **GROQ_API_KEY not configured**\n\nTo enable AI features on Streamlit Cloud:\n1. Go to your app settings\n2. Click 'Secrets (beta)'\n3. Add: `GROQ_API_KEY=your_key_here`")
                answer = "API key not configured. Retrieved documents are shown above, but AI features are disabled. Please configure GROQ_API_KEY to enable AI responses."
            else:
                st.error(f"Error: {e}")
                answer = "Sorry — I couldn't reach the language model right now."

            if show_sources and sources:
                with st.expander("📎 Retrieved sources", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f'<span class="source-pill">📄 [{i}] '
                            f'{src["source"]} — p.{src["page"]} '
                            f'(score: {src["score"]:.2f})</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(src["preview"])

    st.session_state.history.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })

    st.session_state.chat_history.append({"role": "user",      "content": query})
    st.session_state.chat_history.append({"role": "assistant",  "content": answer})

    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]

    save_current_session()

    st.rerun()