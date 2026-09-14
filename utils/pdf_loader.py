import io
from typing import TypedDict

from pypdf import PdfReader


class PageText(TypedDict):
    page: int
    text: str


def load_pdf_pages(pdf_bytes: bytes) -> list[PageText]:
    """Extract text from a PDF byte stream, returning per-page text with page numbers."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})
    return pages


def load_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF byte stream (backward-compatible)."""
    pages = load_pdf_pages(pdf_bytes)
    return "\n\n".join(p["text"] for p in pages)
