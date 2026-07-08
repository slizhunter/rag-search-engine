
import math

from lib.search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT, SearchResult

from .keyword_search import InvertedIndex
from .token_utils import single_token

def term_frequency_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, single_token(term)) # Get the term frequency of the specified term in the document with the given ID using the inverted index

def inverse_document_frequency_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(single_token(term)) # Get the inverse document frequency of the specified term using the inverted index

def bm25_idf_command(term: str) -> float: # Get the BM25 inverse document frequency for a specific term using the inverted index
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(single_token(term))

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float: # Get the BM25 term frequency for a specific term in a given document ID using the inverted index
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, single_token(term), k1, b)

def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchResult]: # Perform a BM25 search for the given query and return a list of matching documents sorted by their BM25 scores
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)