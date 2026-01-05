# Vietnamese Legal Q&A System

A production-ready RAG (Retrieval-Augmented Generation) system for Vietnamese legal document question answering. Built with FastAPI, Celery, and local LLM inference.

## Features

- Multi-stage RAG pipeline with intent routing and query rewriting
- Hybrid retrieval: dense embeddings (Qdrant) + lexical search (Elasticsearch)
- BGE Reranker for improved relevance scoring
- Async processing with Celery for scalable workloads
- Web search fallback via Tavily API
- 4-bit quantized LLM for efficient GPU inference

## Tech Stack

| Component | Technology |
|-----------|------------|
| API | FastAPI |
| Task Queue | Celery + Redis |
| Vector DB | Qdrant |
| Search | Elasticsearch |
| Database | MongoDB |
| LLM | T-VisStar-7B (4-bit quantized) |
| Embeddings | BGE-M3, E5-large, Vietnamese-law |
| Reranker | BGE-Reranker-v2-m3 |

## Requirements

- Python 3.10+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- Docker (for infrastructure services)

## Quick Start

### 1. Start Infrastructure

```bash
# Start required services
docker-compose up -d redis mongodb qdrant elasticsearch
```

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

Required environment variables:
```
TAVILY_API_KEY=tvly-xxx
CELERY_BROKER_URL=redis://localhost:6379
CELERY_RESULT_BACKEND=redis://localhost:6379
```

### 4. Run Application

```bash
cd backend/src
sh ../entrypoint.sh
```

## API Reference

### Chat Completion

```bash
# Async request
curl -X POST http://localhost:8002/chat/complete \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "user_message": "Mức phạt khi không đội mũ bảo hiểm?"}'

# Response: {"task_id": "abc-123"}

# Poll result
curl http://localhost:8002/chat/complete_v2/abc-123
```

### Document Retrieval

```bash
curl -X POST http://localhost:8002/retrieval \
  -H "Content-Type: application/json" \
  -d '{"query": "Luật giao thông đường bộ", "top_k_search": 30, "top_k_rerank": 5}'
```

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ Intent Router   │ ──► Chitchat ──► Direct LLM Response
└────────┬────────┘
         │ Legal
         ▼
┌─────────────────┐
│ Query Rewriter  │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Hybrid Retrieval│ (Qdrant + Elasticsearch)
└────────┬────────┘
         ▼
┌─────────────────┐
│ BGE Reranker    │
└────────┬────────┘
         ▼
┌─────────────────┐
│ LLM Generation  │ ──► No answer? ──► Web Search ──► LLM
└────────┬────────┘
         ▼
    Final Answer
```

## Performance

Evaluated on 1,000 legal Q&A samples:

| Metric | Score |
|--------|-------|
| Recall@5 (Ensemble + Rerank) | 82.82% |
| Recall@10 (Ensemble + Rerank) | 87.66% |
| Answer Correctness | 4.27/5 |

## Infrastructure Ports

| Service | Port |
|---------|------|
| API | 8002 |
| Qdrant | 6333 |
| Elasticsearch | 9200 |
| Redis | 6379 |
| MongoDB | 27017 |

## License

MIT
