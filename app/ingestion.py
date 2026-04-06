from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pypdf import PdfReader


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class PageDoc:
    """One page of text with full provenance — feeds directly into chunking."""
    text:      str
    metadata:  dict = field(default_factory=dict)
    # metadata keys populated here:
    #   source, page_number, total_pages, file_size_kb


# ── Main entry point ───────────────────────────────────────────────────────────

def load_pdfs(
    file_paths: str | list[str],
    min_chars:  int = 50,           # skip pages with almost no text
    ocr_fallback: bool = False,     # enable for scanned docs (needs pytesseract)
) -> list[PageDoc]:
    """
    Load one or more PDFs into a list of PageDoc objects.

    Each PageDoc = one page, with source filename + page number in metadata.
    This feeds directly into chunking.py — pass the list of PageDoc texts
    and their metadata separately.

    Args:
        file_paths:   Single path string or list of paths.
        min_chars:    Pages with fewer characters than this are skipped.
                      Catches blank pages and decoration-only pages.
        ocr_fallback: If True, runs Tesseract OCR on pages where pypdf
                      returns no text. Requires: pip install pytesseract pillow
                      and a Tesseract installation.

    Returns:
        List of PageDoc, one per non-empty page, across all files.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_docs: list[PageDoc] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        # ── File validation ────────────────────────────────────────────────────
        if not path.exists():
            print(f"⚠️  File not found, skipping: {path}")
            continue
        if path.suffix.lower() != ".pdf":
            print(f"⚠️  Not a PDF, skipping: {path}")
            continue

        docs = _load_single_pdf(path, min_chars, ocr_fallback)
        all_docs.extend(docs)
        print(f"📄 {path.name}: {len(docs)} pages loaded")

    print(f"\n✅ Total pages loaded: {len(all_docs)} across {len(file_paths)} file(s)")
    return all_docs


# ── Single-file loader ─────────────────────────────────────────────────────────

def _load_single_pdf(
    path: Path,
    min_chars: int,
    ocr_fallback: bool,
) -> list[PageDoc]:
    """Load and clean all pages from one PDF file."""
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        print(f"⚠️  Could not open {path.name}: {e}")
        return []

    if reader.is_encrypted:
        print(f"⚠️  {path.name} is password-protected, skipping")
        return []

    total_pages = len(reader.pages)
    file_size_kb = round(path.stat().st_size / 1024, 1)
    docs: list[PageDoc] = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as e:
            print(f"  ⚠️  Page {page_num} extraction failed: {e}")
            raw = ""

        # OCR fallback for scanned pages
        if not raw.strip() and ocr_fallback:
            raw = _ocr_page(page, page_num, path.name)

        text = _clean_text(raw)

        if len(text) < min_chars:
            continue                    # skip blank / decoration-only pages

        docs.append(PageDoc(
            text=text,
            metadata={
                "source":       path.name,
                "page_number":  page_num,
                "total_pages":  total_pages,
                "file_size_kb": file_size_kb,
            },
        ))

    return docs


# ── Text cleaning ──────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Fix common pypdf extraction artifacts.

    1. Unicode normalisation  — mojibake like â€™ → '
    2. Dehyphenation          — "computa-\ntional" → "computational"
    3. Collapse whitespace    — multiple spaces/newlines → single space
    4. Strip control chars    — remove non-printable characters
    """
    if not text:
        return ""

    # 1. Normalise unicode (NFC handles most mojibake from PDF encoding)
    text = unicodedata.normalize("NFC", text)

    # 2. Rejoin hyphenated line-breaks  ("computa-\ntional" → "computational")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 3. Replace all whitespace sequences (incl. \n, \t, \r) with single space
    text = re.sub(r"\s+", " ", text)

    # 4. Strip non-printable control characters (keep normal punctuation)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text.strip()


# ── OCR fallback ───────────────────────────────────────────────────────────────

def _ocr_page(page, page_num: int, filename: str) -> str:
    """
    Run Tesseract OCR on a single page that returned no text from pypdf.
    Only called when ocr_fallback=True and the page is blank.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        # pypdf can render a page to an image via its images list
        # For a full rasterize you'd use pdf2image — this is a lightweight fallback
        images = page.images
        if not images:
            return ""

        texts = []
        for img_obj in images:
            img = Image.open(io.BytesIO(img_obj.data))
            texts.append(pytesseract.image_to_string(img))

        return " ".join(texts)

    except ImportError:
        print(
            f"  ⚠️  OCR requested for {filename} p.{page_num} "
            "but pytesseract/pillow not installed. "
            "Run: pip install pytesseract pillow"
        )
        return ""
    except Exception as e:
        print(f"  ⚠️  OCR failed for {filename} p.{page_num}: {e}")
        return ""


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def docs_to_texts_and_metadata(
    docs: list[PageDoc],
) -> tuple[list[str], list[dict]]:
    """
    Unzip PageDocs into parallel lists for chunking.py and embedding.py.

    Usage:
        docs = load_pdfs(["a.pdf", "b.pdf"])
        texts, metadata = docs_to_texts_and_metadata(docs)
        chunks = semantic_chunking(texts)           # chunking.py
        embedded = embed_documents(chunks, metadata=metadata)  # embedding.py
    """
    texts    = [doc.text     for doc in docs]
    metadata = [doc.metadata for doc in docs]
    return texts, metadata