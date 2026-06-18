import pickle, os, math
from .search_utils import BM25_K1, CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies
from .token_utils import tokenize, single_token
from collections import Counter, defaultdict

# Inverted Index Implementation
# Takes the shape: {
#   "token1": {doc_id1, doc_id2, ...},
#   "token2": {doc_id3, doc_id4, ...},
#   ...
# } 
# For example: {
#   "space": {1, 3, 5},
#   "adventure": {2, 3},
#   ...
# }
# where each token maps to a set of document IDs that contain that token. 
# The document IDs correspond to the "id" field in the movie data, 
# and the actual movie information can be retrieved from the docmap using these IDs.
class InvertedIndex:
    def __init__(self) -> None:
         self.index = defaultdict(set)
         self.docmap: dict[int, dict] = {}
         self.term_frequencies: defaultdict[int, Counter] = defaultdict(Counter)         
         self.index_path = os.path.join(CACHE_DIR, "index.pkl")
         self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
         self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str) -> None: # Add a document to the inverted index with the document ID and the text content (title + description) to be indexed
        tokens = tokenize(text) # Preprocess and tokenize the text content to extract meaningful tokens for indexing
        for token in tokens:
            self.index[token].add(doc_id) # Add the document ID to the set of document IDs for the given token in the inverted index
        self.term_frequencies[doc_id].update(tokens) # Update the term frequencies for the document ID with the tokens from the text content

    def get_tf(self, doc_id: int, term: str) -> int: # Get the term frequency of a specific term in a given document ID
        return self.term_frequencies[doc_id][term] # Return the count of how many times the term appears in the document with the specified ID
    
    def get_bm25_idf(self, term: str) -> float: # Get the BM25 inverse document frequency of a specific term
        total_doc_count = len(self.docmap) # Get the total number of documents in the inverted index
        term_match_doc_count = len(self.get_documents(term)) # Get the number of documents that contain the specified term using the inverted index
        idf = math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1) # Calculate the BM25 IDF using the formula: log((N - n + 0.5) / (n + 0.5) + 1), where N is the total number of documents and n is the number of documents containing the term
        return idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1) -> float: # Get the BM25 term frequency of a specific term in a given document ID
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1) # Calculate the BM25 term frequency saturation using the formula: (tf * (k1 + 1)) / (tf + k1), where tf is the term frequency and k1 is a tuning parameter that controls the saturation level

    def get_documents(self, term: str) -> list[int]: # Return a list of document IDs that contain the given term sorted in ascending order
        return sorted(list(self.index.get(term, set())))
    
    def build(self) -> None:
        movies = load_movies() # Load movie data from JSON file as a list of dictionaries
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}") # Add the movie title and description to the inverted index using the document ID as the identifier

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            print("Inverted index files not found. Please run the 'build' command first.")
            raise

def build_command() -> None: # Build the inverted index and save it to disk
        idx = InvertedIndex()
        idx.build()
        idx.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize(query) # Preprocess and tokenize the search query
    results = []
    for query_token in query_tokens:
        doc_ids = idx.get_documents(query_token) # Get the list of document IDs that contain the query token from the inverted index
        for doc_id in doc_ids:
            if doc_id not in results:
                results.append(doc_id)
                if len(results) >= limit:
                    break
        if len(results) >= limit:
            break
    return [idx.docmap[doc_id] for doc_id in results] # Retrieve the actual movie information from the docmap using the document IDs 
                                                      # and return a list of movie dictionaries that match the search query

def bm25_idf_command(term: str) -> float: # Get the BM25 inverse document frequency for a specific term using the inverted index
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(single_token(term))

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float: # Get the BM25 term frequency for a specific term in a given document ID using the inverted index
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, single_token(term), k1)