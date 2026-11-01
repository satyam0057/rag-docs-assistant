"""Ingestion: load -> split (headers, then recursive) -> embed -> store in Chroma.

Run all of data/docs with:  python -m app.ingestion
"""
import requests
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app import config
from app.vectorstore import get_vectorstore, delete_source

_md_split = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)
_rec_split = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n```\n", "\n\n", "\n", ". ", " "],
)


def ingest_text(text: str, source: str) -> int:
    if not text.strip():
        raise ValueError("Document is empty")
    sections = _md_split.split_text(text)
    chunks = _rec_split.split_documents(sections)
    chunks = [c for c in chunks if c.page_content.strip()]
    for i, c in enumerate(chunks):
        c.metadata.update({"source": source, "chunk_id": f"{source}-{i}"})
    if not chunks:
        raise ValueError("Document contains no indexable text")
    delete_source(source)  # dedupe
    if chunks:
        get_vectorstore().add_documents(chunks)
    return len(chunks)


def ingest_url(url: str) -> int:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return ingest_text(r.text, source=url)


def ingest_directory() -> None:
    files = sorted(config.DOCS_DIR.glob("*.md")) + sorted(config.DOCS_DIR.glob("*.txt"))
    if not files:
        print("No files in data/docs. Run: python scripts/fetch_corpus.py")
        return
    for f in files:
        n = ingest_text(f.read_text(encoding="utf-8"), source=f.name)
        print(f"Ingested {f.name}: {n} chunks")


if __name__ == "__main__":
    ingest_directory()
