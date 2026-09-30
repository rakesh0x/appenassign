"""Download the PDF, chunk it, embed it, upload it to Pinecone."""

import sys
from pathlib import Path

import pymupdf
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_PATH, PDF_URL, validate
from app.embeddings import embed_texts
from app.vector_store import delete_index, upsert_chunks

BATCH_SIZE = 96


def download_pdf() -> Path:
    if PDF_PATH.exists():
        print(f"PDF already present: {PDF_PATH}")
        return PDF_PATH

    print(f"Downloading {PDF_URL} ...")
    response = requests.get(PDF_URL, timeout=120)
    response.raise_for_status()
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    PDF_PATH.write_bytes(response.content)
    print(f"Saved {len(response.content) / 1_000_000:.1f} MB to {PDF_PATH}")
    return PDF_PATH


def extract_pages(pdf_path: Path) -> list[dict]:
    document = pymupdf.open(pdf_path)
    return [
        {"page": number, "text": page.get_text().strip()}
        for number, page in enumerate(document, start=1)
        if page.get_text().strip()
    ]


def split_into_chunks(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for start in range(0, len(text), step):
            piece = text[start : start + CHUNK_SIZE].strip()
            if len(piece) < 50:
                continue
            chunks.append(
                {
                    "id": f"page{page['page']}-chunk{len(chunks):04d}",
                    "page": page["page"],
                    "text": piece,
                }
            )
            if start + CHUNK_SIZE >= len(text):
                break
    return chunks


def main() -> None:
    validate()
    if "--rebuild" in sys.argv:
        print("Deleting index ...")
        delete_index()

    download_pdf()
    pages = extract_pages(PDF_PATH)
    chunks = split_into_chunks(pages)
    print(f"Extracted {len(pages)} pages -> {len(chunks)} chunks")

    uploaded = 0
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        uploaded += upsert_chunks(batch, embed_texts([chunk["text"] for chunk in batch]))
        print(f"  uploaded {uploaded}/{len(chunks)}")

    print(f"Done. {uploaded} chunks are in Pinecone.")


if __name__ == "__main__":
    main()
