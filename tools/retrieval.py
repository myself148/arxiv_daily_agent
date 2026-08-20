import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from tools.text_utils import normalize_text, split_text_into_chunks


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_\-+./]*", re.IGNORECASE)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


@dataclass(frozen=True)
class RetrievalDocument:
    paper_index: int
    title: str
    source: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    document: RetrievalDocument
    score: float


def tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS and len(token) > 1
    ]


def build_retrieval_query(paper: Dict) -> str:
    return normalize_text(
        " ".join(
            [
                str(paper.get("title", "")),
                str(paper.get("summary", "")),
            ]
        )
    )


def build_retrieval_corpus(
    papers: Sequence[Dict],
    full_texts: Sequence[str],
    *,
    chunk_size: int,
    overlap: int,
    max_chunks_per_paper: Optional[int],
) -> List[RetrievalDocument]:
    corpus: List[RetrievalDocument] = []

    for paper_index, paper in enumerate(papers):
        full_text = full_texts[paper_index] if paper_index < len(full_texts) else ""
        source = "PDF" if full_text else "Abstract"
        source_text = full_text or paper.get("summary", "")
        chunks = split_text_into_chunks(
            source_text,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks=max_chunks_per_paper,
        )

        for chunk_index, chunk in enumerate(chunks, start=1):
            corpus.append(
                RetrievalDocument(
                    paper_index=paper_index,
                    title=str(paper.get("title", "")),
                    source=source,
                    chunk_index=chunk_index,
                    text=chunk,
                )
            )

    return corpus


def _document_frequencies(tokenized_documents: Iterable[Sequence[str]]) -> Counter:
    frequencies: Counter = Counter()
    for tokens in tokenized_documents:
        frequencies.update(set(tokens))
    return frequencies


def retrieve_relevant_chunks(
    corpus: Sequence[RetrievalDocument],
    query: str,
    *,
    top_k: int,
    paper_index: Optional[int] = None,
) -> List[RetrievedChunk]:
    if top_k <= 0:
        return []

    candidates = [
        document
        for document in corpus
        if paper_index is None or document.paper_index == paper_index
    ]
    query_tokens = tokenize(query)
    if not candidates or not query_tokens:
        return []

    tokenized_documents = [tokenize(document.text) for document in candidates]
    document_frequencies = _document_frequencies(tokenized_documents)
    document_count = len(candidates)
    average_length = sum(len(tokens) for tokens in tokenized_documents) / max(document_count, 1)
    average_length = average_length or 1.0
    query_counter = Counter(query_tokens)

    scored_chunks: List[RetrievedChunk] = []
    for document, document_tokens in zip(candidates, tokenized_documents):
        if not document_tokens:
            continue

        token_counts = Counter(document_tokens)
        document_length = len(document_tokens)
        score = 0.0

        for token, query_weight in query_counter.items():
            term_frequency = token_counts.get(token, 0)
            if term_frequency == 0:
                continue

            frequency = document_frequencies.get(token, 0)
            inverse_document_frequency = math.log(
                1 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            denominator = term_frequency + 1.5 * (1 - 0.75 + 0.75 * document_length / average_length)
            score += query_weight * inverse_document_frequency * (term_frequency * 2.5 / denominator)

        if score > 0:
            scored_chunks.append(RetrievedChunk(document=document, score=score))

    return sorted(scored_chunks, key=lambda chunk: chunk.score, reverse=True)[:top_k]


def format_retrieved_context(
    chunks: Sequence[RetrievedChunk],
    *,
    max_chars: int,
) -> str:
    if not chunks or max_chars <= 0:
        return ""

    sections: List[str] = []
    used_chars = 0
    for index, chunk in enumerate(chunks, start=1):
        document = chunk.document
        header = (
            f"[{index}] {document.source} chunk {document.chunk_index} "
            f"(score: {chunk.score:.2f}, paper: {document.title})"
        )
        remaining = max_chars - used_chars - len(header) - 2
        if remaining <= 0:
            break

        text = normalize_text(document.text)
        if len(text) > remaining:
            text = f"{text[: max(remaining - 3, 0)].rstrip()}..."

        section = f"{header}\n{text}"
        sections.append(section)
        used_chars += len(section) + 2

    return "\n\n".join(sections)
