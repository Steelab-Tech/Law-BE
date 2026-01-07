#!/usr/bin/env python3
"""
Import pre-generated embeddings into local Qdrant.
No GPU required - just uploads vectors from JSONL file.

Usage:
    python import_embeddings_local.py [--file embeddings.jsonl] [--qdrant-url localhost:6333]
"""

import json
import argparse
from tqdm import tqdm
from qdrant_client import QdrantClient, models

# Configuration
COLLECTION_NAME = "vietnamese_legal_docs"
VECTOR_SIZE = 768
BATCH_SIZE = 100


def main():
    parser = argparse.ArgumentParser(description="Import embeddings to local Qdrant")
    parser.add_argument("--file", default="embeddings.jsonl", help="Input JSONL file")
    parser.add_argument("--qdrant-url", default="localhost:6333", help="Qdrant URL")
    args = parser.parse_args()

    print(f"Connecting to Qdrant at {args.qdrant_url}...")
    client = QdrantClient(f"http://{args.qdrant_url}", timeout=120)

    # Test connection
    try:
        collections = client.get_collections()
        print(f"Connected! Existing collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        print("Make sure Qdrant is running: docker-compose up -d qdrant")
        return

    # Create or recreate collection
    if client.collection_exists(COLLECTION_NAME):
        count = client.count(COLLECTION_NAME).count
        print(f"Collection '{COLLECTION_NAME}' exists with {count} points")
        response = input("Recreate collection? (yes/no): ").lower().strip()
        if response == "yes":
            client.delete_collection(COLLECTION_NAME)
            print("Collection deleted")
        else:
            print("Aborting import")
            return

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )
    print("Collection created!")

    # Count lines in file
    print(f"Counting vectors in {args.file}...")
    total_lines = sum(1 for _ in open(args.file, "r", encoding="utf-8"))
    print(f"Total vectors to import: {total_lines}")

    # Import in batches
    print(f"Importing vectors in batches of {BATCH_SIZE}...")
    points = []
    imported = 0
    errors = 0

    with open(args.file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Importing"):
            try:
                data = json.loads(line)
                point = models.PointStruct(
                    id=data["id"],
                    vector=data["vector"],
                    payload=data["payload"]
                )
                points.append(point)

                if len(points) >= BATCH_SIZE:
                    client.upsert(COLLECTION_NAME, points)
                    imported += len(points)
                    points = []

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"Error parsing line: {e}")

    # Upload remaining points
    if points:
        client.upsert(COLLECTION_NAME, points)
        imported += len(points)

    # Verify
    final_count = client.count(COLLECTION_NAME).count
    print(f"\n{'='*50}")
    print("IMPORT COMPLETE!")
    print(f"{'='*50}")
    print(f"Total imported: {imported}")
    print(f"Errors: {errors}")
    print(f"Collection count: {final_count}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
