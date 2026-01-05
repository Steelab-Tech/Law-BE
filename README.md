# Vietnamese Legal Q&A System

A RAG-based Question-Answering chatbot for Vietnamese legal documents using local LLM.

**Dataset**: [Google Drive](https://drive.google.com/drive/folders/1HyF8-EfL4w0G3spBbhcc0jTOqdc4XUhB)

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [RAG Pipeline](#rag-retrieval-augmented-generation)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [API Usage](#api-usage)

## Project Structure

```
├── backend/                    # FastAPI backend with Celery async processing
│   ├── requirements.txt        # Python dependencies
│   ├── entrypoint.sh           # Script to run backend
│   └── src/
│       ├── app.py              # FastAPI entry point
│       ├── brain.py            # LLM logic (routing, rewriting, generation)
│       ├── local_llm.py        # Local LLM wrapper (4-bit quantized)
│       ├── tasks.py            # Celery task definitions
│       ├── database.py         # MongoDB connection
│       ├── models.py           # Database models
│       ├── tavily_search.py    # Web search tool
│       └── search_document/
│           ├── search_vietnamese_legal.py  # Main retrieval (Vietnamese law embeddings)
│           ├── rerank.py                   # BGE Reranker
│           ├── combine_search.py           # Multi-model search (BGE + E5 + ES)
│           ├── search_with_bge.py          # BGE-M3 search
│           ├── search_with_e5.py           # E5 search
│           └── search_elastic.py           # Elasticsearch search
├── finetune_llm/               # LLM fine-tuning (QLoRA + SFTTrainer)
│   ├── gen_data.py             # Generate training data
│   ├── finetune.py             # Fine-tune LLM
│   └── merge_with_base.py      # Merge LoRA weights
├── retrieval/                  # Reranker fine-tuning
│   ├── create_data_rerank.py   # Create reranking data
│   ├── finetune.sh             # Fine-tune BGE reranker
│   └── setup_env.sh            # Environment setup
└── images/                     # Documentation assets
```
## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (for local LLM)
- Running services: Redis, MongoDB, Qdrant, Elasticsearch

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### Environment Variables

Create `.env` file:
```
TAVILY_API_KEY=tvly-...
CELERY_BROKER_URL=redis://localhost:6379
CELERY_RESULT_BACKEND=redis://localhost:6379
```

### Run Backend

```bash
cd backend/src
sh ../entrypoint.sh
```

This starts FastAPI (port 8002) and Celery worker.

## RAG (Retrieval-Augmented Generation)

### System Overview

![rag_system](images/rag_flow.jpg)

### Pipeline Flow

1. **Intent Routing** (`brain.py:detect_route`) - Local LLM classifies query as "legal" or "chitchat"
2. **Query Rewriting** (`brain.py:detect_user_intent`) - Rewrites query with conversation context
3. **Document Retrieval** (`search_vietnamese_legal.py`) - Vietnamese law embedding search via Qdrant
4. **Reranking** (`rerank.py`) - BGE Reranker scores and filters top documents
5. **Answer Generation** (`tasks.py`) - Local LLM generates answer from retrieved docs
6. **Web Search Fallback** (`brain.py`) - Tavily search if RAG returns "no"

### Models Used

| Component | Model |
|-----------|-------|
| LLM | [1TuanPham/T-VisStar-7B-v0.1](https://huggingface.co/1TuanPham/T-VisStar-7B-v0.1) (4-bit quantized) |
| Embeddings | minhquan6203/paraphrase-vietnamese-law, BAAI/BGE-M3, intfloat/multilingual-e5-large |
| Reranker | BAAI/bge-reranker-v2-m3 |

## Fine-tuning

### Reranker Fine-tuning

```bash
cd retrieval
sh setup_env.sh
CUDA_VISIBLE_DEVICES=0 python create_data_rerank.py
sh finetune.sh
```

Training data format:
```json
{"query": "...", "pos": ["positive text"], "neg": ["negative text"]}
```

Parameters: epochs=6, lr=1e-5, batch_size=2

### LLM Fine-tuning

```bash
cd finetune_llm
python gen_data.py                    # Generate 10k train + 1k test samples
CUDA_VISIBLE_DEVICES=0 python finetune.py
python merge_with_base.py             # Merge LoRA weights
```

Uses [QLoRA](https://arxiv.org/abs/2305.14314) with [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) for memory-efficient fine-tuning.

![Tracking training](images/tracking_finetune_llm.png)

## Evaluation

Golden dataset: 1000 samples (query, related_documents, answer)

### Recall@k

| Model             | K=3    | K=5    | K=10   |
|-------------------|--------|--------|--------|
| BGE-m3            | 55.11% | 63.43% | 72.18% |
| E5                | 54.61% | 63.53% | 72.02% |
| Elasticsearch     | 42.54% | 49.61% | 56.85% |
| Ensemble          | 68.38% | 74.85% | 80.66% |
| Ensemble + rerank | 79.82% | 82.82% | 87.66% |

### Correctness

**4.27/5** (5-point scale)

## API Usage

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/retrieval` | Retrieve and rerank documents |
| POST | `/chat/complete` | Submit chat message (async) |
| GET | `/chat/complete_v2/{task_id}` | Poll async response |

### Example

```bash
# Retrieval
curl -X POST http://localhost:8002/retrieval \
  -H "Content-Type: application/json" \
  -d '{"query": "Xe máy không đội mũ bảo hiểm bị phạt bao nhiêu?", "top_k_search": 30, "top_k_rerank": 5}'

# Chat (async)
curl -X POST http://localhost:8002/chat/complete \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "user_message": "Luật giao thông về đội mũ bảo hiểm?"}'
```

## Infrastructure

| Service | Port |
|---------|------|
| Backend API | 8002 |
| Qdrant | 6333 |
| Elasticsearch | 9200 |
| Redis | 6379 |
| MongoDB | 27017 |

## Demo

![demo](images/demo.png)