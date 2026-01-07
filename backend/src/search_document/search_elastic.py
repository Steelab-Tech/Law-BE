import os
import json
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

# Get Elasticsearch URL from environment
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "localhost:9200")

# Connect to Elasticsearch
try:
    es = Elasticsearch(
        [f"http://{ELASTICSEARCH_URL}"],
    )

    # Check connection
    if es.ping():
        print("Connected to Elasticsearch!")
    else:
        print("Could not connect to Elasticsearch.")
except ConnectionError as e:
    print(f"Error connecting to Elasticsearch: {e}")

# Search function for Elasticsearch
def search_data(index_name, query, top_k=10):
    # Perform search with top_k limit
    response = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "text": query  # Search by text content
                }
            },
            "sort": [
                {"_score": {"order": "desc"}}  # Sort by score descending
            ],
            "size": top_k  # Limit number of results
        }
    )

    # Extract results from response
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "text": hit["_source"]["text"],
        })

    return results
