import logging
import os, json

from .hybrid_search import HybridSearch
from .search_utils import (
    load_golden_dataset,
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    SEARCH_MULTIPLIER
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

def rag_command(query: str) -> tuple[list, str]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    rrf_results = hybrid_search.rrf_search(query, limit=DEFAULT_SEARCH_LIMIT * SEARCH_MULTIPLIER)
    rrf_results_top = rrf_results[:DEFAULT_SEARCH_LIMIT]

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.
    
    DO NOT return a response containing anything about "User Safety".

    Query: {query}

    Documents:
    {rrf_results_top}

    Answer:"""

    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return (rrf_results_top, response.choices[0].message.content)

def summarize_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> tuple[list, str]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    rrf_results = hybrid_search.rrf_search(query, limit=limit)
    rrf_results_top = rrf_results[:limit]

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Webflyx users. Webflyx is a movie streaming service.

    DO NOT return a response containing anything about "User Safety".

    Query: {query}

    Search results:
    {rrf_results_top}

    Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""

    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return (rrf_results_top, response.choices[0].message.content)

def citations_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> tuple[list, str]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    rrf_results = hybrid_search.rrf_search(query, limit=limit)
    rrf_results_top = rrf_results[:limit]

    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Webflyx, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {rrf_results_top}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    DO NOT return a response containing anything about "User Safety".

    Answer:"""

    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return (rrf_results_top, response.choices[0].message.content)

def question_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> tuple[list, str]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    rrf_results = hybrid_search.rrf_search(query, limit=limit)
    rrf_results_top = rrf_results[:limit]

    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

    Question: {query}

    Documents:
    {rrf_results_top}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    DO NOT return a response containing anything about "User Safety".

    Answer:"""
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return (rrf_results_top, response.choices[0].message.content)