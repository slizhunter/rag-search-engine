from .hybrid_search import HybridSearch
from .search_utils import (
    load_golden_dataset,
    load_movies,
    RRF_K,
)
from .semantic_search import SemanticSearch

def evaluate_command(limit: int = 5) -> dict:
    movies = load_movies()
    golden_dataset = load_golden_dataset()
    test_cases = golden_dataset["test_cases"]

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    results_dict = {}
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        results = hybrid_search.rrf_search(query=query, k=RRF_K, limit=limit)
        retrieved_docs = [result["document"].get("title", "") for result in results]

        precision = precision_at_k(relevant_docs=relevant_docs, retrieved_docs=retrieved_docs, k=limit)
        recall = recall_at_k(relevant_docs=relevant_docs, retrieved_docs=retrieved_docs, k=limit)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results_dict[query] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "retrieved": retrieved_docs[:limit],
            "relevant": list(relevant_docs),
        }
    return results_dict

def precision_at_k(relevant_docs: list[str], retrieved_docs: list[str], k: int = RRF_K) -> float:
    top_k = retrieved_docs[:k]
    relevant_retrieved_docs = [doc for doc in top_k if doc in relevant_docs]
    return len(relevant_retrieved_docs) / len(top_k) if len(top_k) > 0 else 0

def recall_at_k(relevant_docs: list[str], retrieved_docs: list[str], k: int = RRF_K) -> float:
    top_k = retrieved_docs[:k]
    relevant_retrieved_docs = [doc for doc in top_k if doc in relevant_docs]
    return len(relevant_retrieved_docs) / len(relevant_docs) if relevant_docs else 0