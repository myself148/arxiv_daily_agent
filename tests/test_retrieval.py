from tools.retrieval import (
    build_retrieval_corpus,
    build_retrieval_query,
    format_retrieved_context,
    retrieve_relevant_chunks,
)


def test_retrieves_relevant_pdf_chunk_for_paper():
    papers = [
        {
            "title": "Adaptive DETR for Small Object Detection",
            "summary": "This paper improves DETR with adaptive query refinement for small objects.",
        }
    ]
    full_texts = [
        "\n\n".join(
            [
                "The method introduces adaptive query refinement for DETR. "
                "It improves small object recall by updating object queries across decoder layers.",
                "The appendix discusses training schedules, dataset splits, and implementation details.",
            ]
        )
    ]

    corpus = build_retrieval_corpus(
        papers,
        full_texts,
        chunk_size=140,
        overlap=0,
        max_chunks_per_paper=4,
    )
    retrieved = retrieve_relevant_chunks(
        corpus,
        build_retrieval_query(papers[0]),
        top_k=1,
        paper_index=0,
    )

    assert len(retrieved) == 1
    assert retrieved[0].document.source == "PDF"
    assert "adaptive query refinement" in retrieved[0].document.text


def test_formats_retrieved_context_with_budget():
    papers = [
        {
            "title": "Vision Transformer Detector",
            "summary": "Transformer detector with attention-based feature fusion.",
        }
    ]
    full_texts = [
        "Transformer detector uses attention-based feature fusion for robust detection. "
        "The ablation study shows consistent gains on crowded scenes."
    ]
    corpus = build_retrieval_corpus(
        papers,
        full_texts,
        chunk_size=200,
        overlap=0,
        max_chunks_per_paper=2,
    )
    retrieved = retrieve_relevant_chunks(
        corpus,
        "attention-based feature fusion ablation",
        top_k=1,
        paper_index=0,
    )

    context = format_retrieved_context(retrieved, max_chars=500)

    assert "PDF chunk" in context
    assert "attention-based feature fusion" in context
    assert len(context) <= 500
