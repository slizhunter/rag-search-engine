import os

import numpy as np

from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies

MOVIE_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

class SemanticSearch:
    def __init__(self, model_name = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        movieData = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            movieData.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movieData, show_progress_bar=True)
        np.save(MOVIE_PATH, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(MOVIE_PATH):
            self.embeddings = np.load(MOVIE_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def generate_embedding(self, text: str):
        if not text.strip(): # Checks if the text string is empty
            raise ValueError("Input text cannot be empty")
        embedding = self.model.encode([text])
        return embedding[0]

def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Query Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")