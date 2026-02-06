import io

from pypdf import PdfReader


def load_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF byte stream."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n\n".join(texts)
