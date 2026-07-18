import os

from .search_utils import load_movies

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        # Normalize the BM25 and semantic search scores
        bm25_scores = normalize([result["score"] for result in bm25_results])
        semantic_scores = normalize([result["score"] for result in semantic_results])

        # Combine results using the weighted sum of BM25 and semantic search scores
        # Create a dictionary to store the combined results with this structure: 
        # {     
        #   doc_id: {
        #       "document": <doc object>, 
        #       "bm25_score": float, 
        #       "semantic_score": float,
        #       "hybrid_score": float,
        #   }
        # }
        combined_results = {}
        for i, result in enumerate(bm25_results):
            doc_id = result["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "document": result,
                    "bm25_score": bm25_scores[i],
                    "semantic_score": 0.0,
                }

        for i, result in enumerate(semantic_results):
            doc_id = result["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "document": result,
                    "bm25_score": 0.0,
                    "semantic_score": semantic_scores[i],
                }
            else:
                combined_results[doc_id]["semantic_score"] = semantic_scores[i]

        # Calculate hybrid scores using the weighted sum of BM25 and semantic search scores
        for doc_id, result in combined_results.items():
            result["hybrid_score"] = hybrid_score(
                result["bm25_score"], result["semantic_score"], alpha
            )

        # Sort the combined results by the hybrid score in descending order and return the top `limit` results
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["hybrid_score"], reverse=True
        )
        return sorted_results

    def rrf_search(self, query: str, k: int = 60, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        # Combine results using the weighted sum of BM25 and semantic search scores
        # Create a dictionary to store the combined results with this structure: 
        # {     
        #   doc_id: {
        #       "document": <doc object>, 
        #       "bm25_rank": int, 
        #       "semantic_rank": int,
        #   }
        # }
        combined_results = {}
        for i, result in enumerate(bm25_results):
            doc_id = result["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "document": result,
                    "bm25_rank": i + 1,
                    "semantic_rank": 0,
                    "rrf_score": rrf_score(i + 1, k),
                }

        for i, result in enumerate(semantic_results):
            doc_id = result["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "document": result,
                    "bm25_rank": 0,
                    "semantic_rank": i + 1,
                    "rrf_score": rrf_score(i + 1, k),
                }
            else:
                combined_results[doc_id]["semantic_rank"] = i + 1
                combined_results[doc_id]["rrf_score"] += rrf_score(i + 1, k)

        # Sort the combined results by the RRF score in descending order and return the top `limit` results
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["rrf_score"], reverse=True
        )
        return sorted_results
    
def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score == 0:
        normalized_scores = [1.0 for _ in scores]
    else:
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized_scores

def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

def handle_weighted_search(query: str, alpha: float, limit: int = 5) -> list[dict]:
    search_engine = HybridSearch(load_movies())
    results = search_engine.weighted_search(query, alpha, limit)
    return results[:limit]

def handle_rrf_search(query: str, k: int = 60, limit: int = 5) -> list[dict]:
    search_engine = HybridSearch(load_movies())
    results = search_engine.rrf_search(query, k, limit)
    return results[:limit]