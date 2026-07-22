from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .search_utils import load_movies, DEFAULT_SEARCH_LIMIT

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents=None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents] if self.documents else []
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True) if self.texts else []

    def embed_image(self, image_path: str):
        loaded_image = Image.open(image_path)
        encoded_image = self.model.encode([loaded_image])
        return encoded_image[0]
    
    def search_with_image(self, image_path: str) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        similarity_scores = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity_score = cosine_similarity(image_embedding, text_embedding)
            similarity_scores.append(
                {
                    "doc_id": self.documents[i]["id"] if self.documents else None,
                    "title": self.documents[i]["title"] if self.documents else None,
                    "description": self.documents[i]["description"][:100] if self.documents else None,
                    "similarity_score": similarity_score
                }
            )
        similarity_scores.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarity_scores

    
def verify_image_embedding(image_path: str) -> bool:
    multimodal_search = MultimodalSearch()
    image_embedding = multimodal_search.embed_image(image_path)
    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")
    return image_embedding is not None

def image_search_command(image_path: str, limit: int = DEFAULT_SEARCH_LIMIT):
    multimodal_search = MultimodalSearch(documents=load_movies())
    results = multimodal_search.search_with_image(image_path)
    return results[:limit]