import logging
import os, json

from .hybrid_search import HybridSearch
from .search_utils import (
    load_golden_dataset,
    load_movies,
    RRF_K,
)
from .semantic_search import SemanticSearch

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"

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

def llm_evaluate(query: str, eval_docs: list[dict]) -> list[int]:
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {eval_docs}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. 
    DO NOT return a response containing anything about "User Safety".
    
    For example:
    [2, 0, 3, 2, 0, 1]"""

    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    if response.choices[0].message.content.strip() == "":
        logging.warning("LLM evaluation returned an empty response for query: %s", query)
        return []
    if "User Safety" in response.choices[0].message.content:
        logging.warning("Bad LLM evaluation response ('User Safety') for query: %s", query)
        return []
    logging.info("LLM evaluation response: %s", response.choices[0].message.content.strip())
    return json.loads(response.choices[0].message.content.strip())