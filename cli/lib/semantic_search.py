import os, re, json

import numpy as np

from sentence_transformers import SentenceTransformer
from typing import TypedDict

from .search_utils import (
    SearchResult,
    CACHE_DIR, 
    DEFAULT_SEARCH_LIMIT, 
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    SCORE_PRECISION,
    load_movies,
    format_search_result
)
MOVIE_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int

# Example self.documents:
# [
#     {"id": 1, "title": "Movie 1", "description": "Description 1"},
#     {"id": 2, "title": "Movie 2", "description": "Description 2"},
#     ...
# ]
#
# Examples self.document_map:
# {
#     1: {"id": 1, "title": "Movie 1", "description": "Description 1"},
#     2: {"id": 2, "title": "Movie 2", "description": "Description 2"},
#     ...
# }

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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
         super().__init__(model_name)
         self.chunk_embeddings = None
         self.chunk_metadata = None
        
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        chunks = []
        chunk_metadata: list[ChunkMetadata] = []
        for doc in documents:
            if len(doc['description']) > 0:
                doc_chunks = semantic_chunk(doc['description'], 4, 1)
                total_doc_chunks = len(doc_chunks)
                for chunk_idx, chunk in enumerate(doc_chunks):
                    chunks.append(chunk)
                    chunk_metadata.append(
                        {
                            "movie_idx": doc["id"],
                            "chunk_idx": chunk_idx,
                            "total_chunks": total_doc_chunks,
                        }
                    )
        if not chunks:
            raise ValueError("No text chunks were generated from the provided documents")
        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = chunk_metadata
        np.save(CHUNK_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(CHUNK_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                chunk_metadata = json.load(f)
                self.chunk_metadata = chunk_metadata["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        if self.chunk_metadata is None:
            raise ValueError("No chunk metadata loaded. Call `load_or_create_chunk_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": i,
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": similarity_score
                 }
            )
        best_chunk_scores = {}
        for current_score in chunk_scores:
            if current_score["movie_idx"] not in best_chunk_scores or current_score["score"] > best_chunk_scores[current_score["movie_idx"]]["score"]:
                best_chunk_scores[current_score["movie_idx"]] = current_score
        sorted_scores = sorted(best_chunk_scores.values(), key=lambda x: x["score"], reverse=True)
        formatted_results = []
        for result in sorted_scores[:limit]:
            movie = self.document_map[result["movie_idx"]]
            formatted_results.append(format_search_result(
                doc_id=result["movie_idx"],
                title=movie["title"],
                document=movie["description"][:100],
                score=round(result["score"], SCORE_PRECISION),
                metadata={}
            ))
        return formatted_results

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
    words = text.split()
    print(f"Chunking {len(text)} characters")
    for i in range(0, len(words), chunk_size - overlap):
        if len(words[i:i + chunk_size]) > overlap:
            chunk = " ".join(words[i:i + chunk_size])
            print(f"{i // (chunk_size - overlap) + 1}. {chunk}")

def semantic_chunk(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = text.strip()  # Strip leading/trailing whitespace from the input text
    if len(text) == 0:  # Check if the input text is empty after stripping
        print("Input text is empty")
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text) # Strip out leading/trailing whitespace and split text into sentences based on punctuation followed by whitespace
    if len(sentences) == 1 and not sentences[0].endswith((('.', '!', '?'))): # Check if the single sentence does not end with punctuation
        print("Only found: one sentence with no punctuation in the input text")
        return sentences
    chunks = []
    for i in range(0, len(sentences), max_chunk_size - overlap): # Iterate over sentences with a step size of max_chunk_size - overlap
        if len(sentences[i:i + max_chunk_size]) > overlap: # Check if the number of sentences in the current chunk is greater than the overlap to avoid creating chunks that are already included in the previous chunk
            chunk = " ".join( # Strip leading/trailing whitespace from the sentences in the current chunk and join them into a single string
                s                                               # Use the walrus operator to assign the stripped sentence to 's' and include it in the join if it's not empty
                for sentence in sentences[i:i + max_chunk_size] # Iterate over the sentences in the current chunk
                if (s := sentence.strip())                      # Include the sentence in the chunk if it's not empty
            )
            chunks.append(chunk) # Append the chunk to the list of chunks
    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP):
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def embed_chunks():
    search_instance = ChunkedSemanticSearch()
    embeddings = search_instance.load_or_create_chunk_embeddings(load_movies())
    print(f"Generated {len(embeddings)} chunked embeddings")

def semantic_search_chunked(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    search_instance = ChunkedSemanticSearch()
    search_instance.load_or_create_chunk_embeddings(load_movies())
    results = search_instance.search_chunks(query, limit)
    for i, result in enumerate(results):
        print(f"\n{i + 1}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
