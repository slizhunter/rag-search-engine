import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies() # Load movie data from JSON file as a list of dictionaries
    stopword_tokens = tokenize(load_stopwords()) # Load stopwords from file, preprocess and tokenize them into a list of tokens
    query_tokens = remove_stopwords(tokenize(query), stopword_tokens) # Preprocess and tokenize the search query, then remove stopwords from the query tokens
    results = []
    for movie in movies:
        title_tokens = remove_stopwords(tokenize(movie["title"]), stopword_tokens) # Preprocess and tokenize the movie title, then remove stopwords from the title tokens
        if has_matching_token(query_tokens, title_tokens):
                results.append(movie)
                if len(results) >= limit:
                    break
    return results

def remove_stopwords(tokens: list[str], stopwords: list[str]) -> list[str]:
    return [token for token in tokens if token not in stopwords] # Filter out tokens that are present in the stopwords list

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if stemmer.stem(query_token) in stemmer.stem(title_token): # Check if the stemmed version of the query token matches the stemmed version of any title token
                return True
    return False

def preprocess_text(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation)).lower() # Remove punctuation and convert to lowercase for case-insensitive matching

def tokenize(text: str) -> list[str]:
    all_tokens = preprocess_text(text).split() # Split the preprocessed text into tokens based on whitespace
    return [token for token in all_tokens if token] # Filter out any empty tokens that may result from multiple spaces or punctuation removal