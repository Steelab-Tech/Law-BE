"""
Simple search module for vietnamese_legal_docs collection
Uses: minhquan6203/paraphrase-vietnamese-law embedding model
Supports both local Qdrant and Qdrant Cloud
"""
import os
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Default configurations from environment
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "localhost:6333")
DEFAULT_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
DEFAULT_DEVICE = os.getenv("LLM_DEVICE", "cpu")


class VietnameseLegalSearch:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        host: str = None,
        collection_name: str = "vietnamese_legal_docs",
        model_name: str = "minhquan6203/paraphrase-vietnamese-law",
        device: str = None,
        api_key: str = None
    ):
        if self._initialized:
            return

        device = device or DEFAULT_DEVICE
        api_key = api_key or DEFAULT_QDRANT_API_KEY

        self.collection_name = collection_name

        logger.info(f"Initializing VietnameseLegalSearch...")
        logger.info(f"Loading embedding model: {model_name}")

        # Connect to Qdrant (Cloud or local)
        if api_key:
            # Qdrant Cloud
            qdrant_url = host or DEFAULT_QDRANT_URL
            if not qdrant_url.startswith("http"):
                qdrant_url = f"https://{qdrant_url}"
            logger.info(f"Connecting to Qdrant Cloud: {qdrant_url}")
            self.client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=60)
        else:
            # Local Qdrant
            qdrant_url = host or f"http://{DEFAULT_QDRANT_URL}"
            logger.info(f"Connecting to local Qdrant: {qdrant_url}")
            self.client = QdrantClient(qdrant_url, timeout=60)

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
