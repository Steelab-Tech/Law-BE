"""
Import vietnamese-legal-corpus-20k-raw into Qdrant
Dataset: https://huggingface.co/datasets/52100303-TranPhuocSang/vietnamese-legal-corpus-20k-raw

Overnight run script - includes:
- Progress checkpointing (resume if interrupted)
- Logging to file
- Error handling
"""
import torch
import re
import os
import json
import logging
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from tqdm import tqdm

# ===== LOGGING SETUP =====
log_file = f"import_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Checkpoint file for resume capability
CHECKPOINT_FILE = "import_checkpoint.json"

# ===== CONFIG =====
MODEL_NAME = "minhquan6203/paraphrase-vietnamese-law"
COLLECTION_NAME = "vietnamese_legal_docs"
VECTOR_SIZE = 768
BATCH_SIZE = 128

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

# ===== STEP 6: Embedding + Upload with checkpointing =====
logger.info("Step 2: Embedding and uploading to Qdrant...")
EMBED_BATCH = 256
UPLOAD_BATCH = 5000
total_chunks = len(all_chunks)

# Load checkpoint if exists
start_idx = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoint = json.load(f)
        start_idx = checkpoint.get('last_uploaded', 0)
        if start_idx > 0:
            logger.info(f"Resuming from checkpoint: {start_idx}/{total_chunks}")

# Process in batches with checkpoint saves
all_embeddings = []
last_checkpoint_save = start_idx

try:
    # Embedding phase
    logger.info(f"Embedding {total_chunks - start_idx} remaining chunks...")
    for i in tqdm(range(start_idx, total_chunks, EMBED_BATCH), desc="Embedding", initial=start_idx//EMBED_BATCH, total=total_chunks//EMBED_BATCH):
        batch_texts = [chunk['text'] for chunk in all_chunks[i:i+EMBED_BATCH]]
        batch_emb = model.encode(
            batch_texts,
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        all_embeddings.extend(batch_emb)

    logger.info(f"Embeddings done: {len(all_embeddings)}")

    # Upload phase
    logger.info("Uploading to Qdrant...")
    for i in tqdm(range(0, len(all_embeddings), UPLOAD_BATCH), desc="Uploading"):
        actual_idx = start_idx + i
        end = min(i + UPLOAD_BATCH, len(all_embeddings))

        points = [
            models.PointStruct(
                id=actual_idx + j,
                vector=all_embeddings[i + j].tolist(),
                payload=all_chunks[actual_idx + j]
            )
            for j in range(end - i)
        ]

        client.upsert(COLLECTION_NAME, points)

        # Save checkpoint every 50k uploads
        if (actual_idx + end - i) - last_checkpoint_save >= 50000:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump({'last_uploaded': actual_idx + end - i}, f)
            last_checkpoint_save = actual_idx + end - i
            logger.info(f"Checkpoint saved: {last_checkpoint_save}/{total_chunks}")

    # Clean up checkpoint on success
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

except KeyboardInterrupt:
    logger.warning("Interrupted! Saving checkpoint...")
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'last_uploaded': last_checkpoint_save}, f)
    logger.info(f"Checkpoint saved at {last_checkpoint_save}. Run script again to resume.")
    exit(1)
except Exception as e:
    logger.error(f"Error: {e}")
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'last_uploaded': last_checkpoint_save}, f)
    raise

# ===== DONE =====
final_count = client.count(COLLECTION_NAME).count
logger.info("="*50)
logger.info("DONE!")
logger.info(f"Collection: {COLLECTION_NAME}")
logger.info(f"Total chunks indexed: {final_count}")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars")
logger.info(f"Log file: {log_file}")
logger.info("="*50)
