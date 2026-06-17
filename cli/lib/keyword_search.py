import string, pickle, os
from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
from nltk.stem import PorterStemmer
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
        
STOPWORDS = load_stopwords()

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

def term_frequency_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, single_token(term)) # Get the term frequency of the specified term in the document with the given ID using the inverted index

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token: # Check if the query token matches the title token
                return True
    return False

def preprocess_text(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation)).lower() # Remove punctuation and convert to lowercase for case-insensitive matching

def tokenize(text: str) -> list[str]:
    all_tokens = preprocess_text(text).split() # Split the preprocessed text into tokens based on whitespace
    valid_tokens = []
    for token in all_tokens:
        if token: # Check if the token is not an empty string (which can happen if there are multiple spaces or punctuation removal)
            valid_tokens.append(token) # Add it to the list of valid tokens
    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS: # Filter out stopwords from the list of valid tokens
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for word in filtered_words:
        stemmed_tokens.append(stemmer.stem(word)) # Apply stemming to the filtered tokens to reduce them to their root form
    return stemmed_tokens

def single_token(token: str) -> str:
    tokens = tokenize(token)
    if len(tokens) == 1:
        return tokens[0] # Tokenize a single token using the same preprocessing and tokenization steps as the main tokenize function
    raise Exception("Input is not a single token")