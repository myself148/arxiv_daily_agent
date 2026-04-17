import re
from typing import List, Optional


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_text_into_chunks(
    text: str,
    *,
    chunk_size: int,
    overlap: int = 0,
    max_chunks: Optional[int] = None,
) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]
    chunks: List[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current_chunk else f"{current_chunk}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)
            if max_chunks is not None and len(chunks) >= max_chunks:
                return chunks

            prefix = current_chunk[-overlap:] if overlap else ""
            current_chunk = f"{prefix}\n\n{paragraph}".strip() if prefix else paragraph
            if len(current_chunk) <= chunk_size:
                continue

        start = 0
        step = chunk_size - overlap if overlap else chunk_size
        while start < len(paragraph):
            piece = paragraph[start:start + chunk_size].strip()
            if piece:
                chunks.append(piece)
                if max_chunks is not None and len(chunks) >= max_chunks:
                    return chunks
            start += step
        current_chunk = ""

    if current_chunk and (max_chunks is None or len(chunks) < max_chunks):
        chunks.append(current_chunk)

    return chunks[:max_chunks] if max_chunks is not None else chunks
