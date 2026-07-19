import os, time
from typing import Literal

from .search_utils import load_movies, RRF_K, DEFAULT_SEARCH_LIMIT
from .query_enhancement import llm_rerank_batch, llm_rerank_individual, llm_spellcheck, llm_rewrite, llm_expand

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

from sentence_transformers import CrossEncoder

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

    def rrf_search(
            self, 
            query: str, 
            k: int = RRF_K, 
            limit: int = DEFAULT_SEARCH_LIMIT, 
            enhance: Literal["spell", "rewrite", "expand"] | None = None, 
            rerank_method: Literal["individual", "batch", "cross_encoder"] | None = None
        ) -> list[dict]:
        if enhance:
            original_query = query
            if enhance == "spell":
                query = llm_spellcheck(original_query)
            elif enhance == "rewrite":
                query = llm_rewrite(original_query)
            elif enhance == "expand":
                query = llm_expand(original_query)
            if query != original_query:
                print(f"Enhanced query ({enhance}): '{original_query}' -> '{query}'\n")

        mult = 5 if rerank_method else 1 # If rerank_method is specified, we multiply the limit by 5 to get more results for re-ranking. Otherwise, we use the original limit.

        bm25_results = self._bm25_search(query, limit * mult)
        semantic_results = self.semantic_search.search_chunks(query, limit * mult)

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

        # Sort the combined results by the RRF score in descending order
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        if rerank_method == "individual":
            print(f"Re-ranking top {limit} results using {rerank_method} method...")
            print(f"Making {len(sorted_results)} calls to the LLM for re-ranking...")
            for result in sorted_results:
                rerank_score = llm_rerank_individual(query, result["document"])
                # Check if rerank_score can be converted to an integer, if not, set it to 0 and print a warning
                try:
                    rerank_score = int(rerank_score)
                except ValueError:
                    print(f"Warning: Rerank score '{rerank_score}' is not an integer for document '{result['document']['title']}'. Setting rerank score to 0.")
                    rerank_score = 0
                result["rerank_score"] = rerank_score
                time.sleep(3)  # Sleep for 3 seconds to avoid rate limiting
            sorted_results = sorted(
                sorted_results, key=lambda x: x["rerank_score"], reverse=True
            )
        
        if rerank_method == "batch":
            print(f"Re-ranking top {limit} results using {rerank_method} method...")
            rerank_batch = []
            retries = 3
            for attempt in range(1, retries):
                print(f"Attempt {attempt} to rerank batch...")
                try:
                    rerank_batch = llm_rerank_batch(query, sorted_results)
                    print(f"Rerank batch results: {rerank_batch}")
                    if len(rerank_batch) == len(sorted_results):
                        break  # If successful, exit the retry loop
                    raise ValueError(f"Rerank batch length mismatch: expected {len(sorted_results)}, got {len(rerank_batch)}")
                except Exception as e:
                    print(f"Attempt {attempt} failed with error: {e}")
                    if attempt < retries - 1:
                        print("Retrying...")
                    else:
                        print("All retry attempts failed.")
                        rerank_batch = []
            if rerank_batch == []:
                print("Rerank batch is empty after all retry attempts.")
            else:
                rank_by_id = {doc_id: rank for rank, doc_id in enumerate(rerank_batch, 1)} # Create a mapping of document IDs to their new ranks based on the rerank_batch

                for result in sorted_results:
                    doc_id = result["document"]["id"]
                    result["rerank_rank"] = rank_by_id.get(doc_id, len(rerank_batch) + 1)

                sorted_results = sorted(sorted_results, key=lambda x: x["rerank_rank"])

        if rerank_method == "cross_encoder":
            print(f"Re-ranking top {limit} results using {rerank_method} method...")
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            pairs = []
            for doc in sorted_results:
                pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
            scores = cross_encoder.predict(pairs)
            for i, result in enumerate(sorted_results):
                result["cross_encoder_score"] = scores[i]
            sorted_results = sorted(sorted_results, key=lambda x: x["cross_encoder_score"], reverse=True)

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

def handle_rrf_search(
        query: str, 
        k: int = RRF_K, 
        limit: int = DEFAULT_SEARCH_LIMIT, 
        enhance: Literal["spell", "rewrite", "expand"] | None = None, 
        rerank_method: Literal["individual", "batch", "cross_encoder"] | None = None,
    ) -> list[dict]:
    search_engine = HybridSearch(load_movies())
    results = search_engine.rrf_search(query, k, limit, enhance, rerank_method)
    return results[:limit]