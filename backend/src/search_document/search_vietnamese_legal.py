"""
Simple search module for vietnamese_legal_docs collection
Uses: minhquan6203/paraphrase-vietnamese-law embedding model
"""
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class VietnameseLegalSearch:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        host: str = "http://194.93.48.55:6333",
        collection_name: str = "vietnamese_legal_docs",
        model_name: str = "minhquan6203/paraphrase-vietnamese-law",
        device: str = "cuda:0"
    ):
        if self._initialized:
            return

        self.host = host
        self.collection_name = collection_name

        logger.info(f"Initializing VietnameseLegalSearch...")
        logger.info(f"Loading embedding model: {model_name}")

        self.client = QdrantClient(host, timeout=60)
        self.model = SentenceTransformer(model_name, device=device)

        self._initialized = True
        logger.info("VietnameseLegalSearch initialized!")

    def encode_query(self, query_text: str):
        """Encode query text to vector"""
        embedding = self.model.encode(
            query_text,
            normalize_embeddings=True
        )
        return embedding.tolist()

    def search(self, query_text: str, limit: int = 20):
        """Search for similar documents"""
        query_vector = self.encode_query(query_text)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

        return results


# Singleton instance
_search_instance = None

def get_legal_search() -> VietnameseLegalSearch:
    global _search_instance
    if _search_instance is None:
        _search_instance = VietnameseLegalSearch()
    return _search_instance


def search_vietnamese_legal(query: str, top_k: int = 20) -> list:
    """
    Search Vietnamese legal documents

    Args:
        query: Search query
        top_k: Number of results to return

    Returns:
        List of text results
    """
    searcher = get_legal_search()
    results = searcher.search(query, limit=top_k)

    texts = []
    for point in results.points:
        text = point.payload.get("text", "")
        if text:
            texts.append(text)

    return texts
