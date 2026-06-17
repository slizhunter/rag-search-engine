
import string

from nltk.stem import PorterStemmer

from .search_utils import load_stopwords

STOPWORDS = load_stopwords()

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