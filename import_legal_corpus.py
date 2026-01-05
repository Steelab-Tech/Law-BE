"""
Import vietnamese-legal-corpus-20k-raw into Qdrant
Dataset: https://huggingface.co/datasets/52100303-TranPhuocSang/vietnamese-legal-corpus-20k-raw
"""
import torch
import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from tqdm import tqdm

# ===== CONFIG =====
MODEL_NAME = "minhquan6203/paraphrase-vietnamese-law"
COLLECTION_NAME = "vietnamese_legal_docs"
VECTOR_SIZE = 768
BATCH_SIZE = 32

# Chunking config
CHUNK_SIZE = 512  # Characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks

# ===== STEP 0: Check GPU =====
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# ===== STEP 1: Load dataset =====
print("\nLoading dataset: vietnamese-legal-corpus-20k-raw...")
print("(This may take a few minutes for 2.6GB download)")
dataset = load_dataset("52100303-TranPhuocSang/vietnamese-legal-corpus-20k-raw")
print(f"Dataset loaded: {len(dataset['train'])} documents")

# ===== STEP 2: Load model =====
print(f"\nLoading model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=device)
print("Model loaded!")

# ===== STEP 3: Chunking functions =====
def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    if not text or len(text) < 100:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to cut at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > chunk_size // 2:
                chunk = chunk[:cut_point + 1]
                end = start + cut_point + 1

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap
        if start >= len(text):
            break

    return chunks

# ===== STEP 4: Connect to Qdrant =====
print("\nConnecting to Qdrant Docker...")
client = QdrantClient(host="localhost", port=6333, timeout=300)

# Delete empty collection if exists, or create new one
if client.collection_exists(COLLECTION_NAME):
    count = client.count(COLLECTION_NAME).count
    if count == 0:
        print(f"Collection exists but empty, recreating: {COLLECTION_NAME}...")
        client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
    else:
        print(f"Collection already has {count} points, will add more...")
else:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )
print(f"Collection '{COLLECTION_NAME}' ready!")

# ===== STEP 5: Process and upload data =====
print("\nProcessing and uploading data...")
print("Step 1: Chunking all documents...")

all_chunks = []
for idx, doc in enumerate(tqdm(dataset['train'], desc="Chunking")):
    # Get full_text (preferred) or title
    full_text = clean_text(doc.get('full_text', ''))
    title = clean_text(doc.get('title', ''))

    # Extract metadata
    metadata = {
        'title': title,
        'official_number': doc.get('official_number', ''),
        'document_type': doc.get('document_type', ''),
        'document_field': doc.get('document_field', ''),
        'issued_date': doc.get('issued_date', ''),
        'effective_date': doc.get('effective_date', ''),
        'place_issue': doc.get('place_issue', ''),
        'signer': doc.get('signer', ''),
        'url': doc.get('url', ''),
        'source_id': doc.get('source_id', idx),
    }

    # Chunk full_text
    if full_text:
        chunks = chunk_text(full_text)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'chunk_idx': chunk_idx,
                'total_chunks': len(chunks),
                **metadata
            })
    elif title:
        # If no full_text, use title
        all_chunks.append({
            'text': title,
            'chunk_idx': 0,
            'total_chunks': 1,
            **metadata
        })

print(f"Total chunks created: {len(all_chunks)}")

# ===== STEP 6: Embedding and upload =====
print("\nStep 2: Embedding and uploading to Qdrant...")
total_uploaded = 0

for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Uploading"):
    batch = all_chunks[i:i+BATCH_SIZE]

    # Create embeddings
    texts = [chunk['text'] for chunk in batch]
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Create points
    points = []
    for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
        points.append(
            models.PointStruct(
                id=i + j,
                vector=embedding.tolist(),
                payload=chunk
            )
        )

    # Upload batch
    client.upsert(COLLECTION_NAME, points)
    total_uploaded += len(points)

    # Clear GPU memory periodically
    if torch.cuda.is_available() and (i // BATCH_SIZE) % 50 == 0:
        torch.cuda.empty_cache()

# ===== DONE =====
final_count = client.count(COLLECTION_NAME).count
print(f"\n{'='*50}")
print(f"DONE!")
print(f"Collection: {COLLECTION_NAME}")
print(f"Total chunks indexed: {final_count}")
print(f"Model: {MODEL_NAME}")
print(f"Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars")
print(f"{'='*50}")
