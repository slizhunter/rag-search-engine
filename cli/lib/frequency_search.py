
import math

from .keyword_search import InvertedIndex
from .token_utils import single_token

def term_frequency_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, single_token(term)) # Get the term frequency of the specified term in the document with the given ID using the inverted index

def inverse_document_frequency_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    total_doc_count = len(idx.docmap) # Get the total number of documents in the inverted index
    term_match_doc_count = len(idx.get_documents(single_token(term))) # Get the number of documents that contain the specified term using the inverted index
    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1)) # Get the inverse document frequency of the specified term using the inverted index
    return idf