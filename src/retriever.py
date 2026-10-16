
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_pages_into_chunks(
    pages: list[dict],
    max_chars: int = 1200,
    overlap: int = 180,
) -> list[dict]:
    """
    Split extracted PDF pages into overlapping text chunks.

    Every chunk retains its original PDF page number.

    Args:
        pages: List of dictionaries with page and text fields.
        max_chars: Maximum approximate characters per chunk.
        overlap: Number of characters shared between chunks.

    Returns:
        A list of dictionaries containing page and text.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")

    if overlap < 0 or overlap >= max_chars:
        raise ValueError(
            "overlap must be non-negative and smaller than max_chars."
        )

    chunks = []

    for page in pages:
        page_number = page.get("page")
        text = page.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        # Normalize whitespace while retaining source page numbers.
        text = re.sub(r"\s+", " ", text).strip()

        start = 0

        while start < len(text):
            end = min(start + max_chars, len(text))

            # Prefer ending at a word boundary when practical.
            if end < len(text):
                space = text.rfind(" ", start, end)

                if space > start + int(max_chars * 0.6):
                    end = space

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "page": page_number,
                        "text": chunk_text,
                    }
                )

            if end >= len(text):
                break

            # Overlap retains context around chunk boundaries.
            start = max(end - overlap, start + 1)

    return chunks


def retrieve_relevant_passages(
    question: str,
    pages: list[dict],
    top_k: int = 5,
    min_similarity: float = 0.08,
) -> list[dict]:
    """
    Retrieve text passages that match a user's question.

    Args:
        question: User's current question.
        pages: Extracted report pages.
        top_k: Maximum number of passages to return.
        min_similarity: Minimum cosine similarity required.

    Returns:
        A ranked list of dictionaries containing:
        page, text, and similarity.

    Similarity is a lexical text-matching score, NOT a measure
    of factual correctness or probability of relevance.
    """

    if not question or not question.strip():
        return []

    if top_k <= 0:
        return []

    if not 0 <= min_similarity <= 1:
        raise ValueError(
            "min_similarity must be between 0 and 1."
        )

    # Prepare source chunks from all extracted report pages.
    chunks = split_pages_into_chunks(pages)

    if not chunks:
        return []

    documents = [chunk["text"] for chunk in chunks]

    # TF-IDF uses individual words and two-word phrases.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
    )

    try:
        document_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        # For example, text may be empty after tokenization.
        return []

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        document_matrix,
    ).flatten()

    # Examine the most similar passages first.
    ranked_indices = similarities.argsort()[::-1]

    results = []
    chunks_per_page = {}

    for index in ranked_indices:
        similarity = float(similarities[index])

        # Exclude weak matches and passages with no lexical overlap.
        # The remaining results are ranked no higher than this score.
        if similarity < min_similarity or similarity <= 0:
            break

        chunk = chunks[index]
        page_number = chunk["page"]

        # Avoid returning too many chunks from the same page.
        if chunks_per_page.get(page_number, 0) >= 2:
            continue

        results.append(
            {
                "page": page_number,
                "text": chunk["text"],
                "similarity": round(similarity, 4),
            }
        )

        chunks_per_page[page_number] = (
            chunks_per_page.get(page_number, 0) + 1
        )

        if len(results) >= top_k:
            break

    return results
