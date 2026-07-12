import os

import numpy as np

from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE, load_movies

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
    
    def search(self, query: str, limit: int = 5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scores = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, doc_embedding)
            # Embeddings are ordered to match self.documents.
            scores.append((similarity_score, self.documents[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scores[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"]
                }
            )
        return results

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

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(load_movies())
    results = semantic_search.search(query, limit)
    for i, (score, doc) in enumerate(results):
        print(f"{i + 1}. {doc['title']} (score: {score:.4f}) \n {doc['description']}\n")

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = 0):
    if not text.strip():
        raise ValueError("Input text cannot be empty")
    words = text.split()
    print(f"Chunking {len(text)} characters")
    for i in range(0, len(words), chunk_size - overlap):
        if len(words[i:i + chunk_size]) > overlap:
            chunk = " ".join(words[i:i + chunk_size])
            print(f"{i // (chunk_size - overlap) + 1}. {chunk}")