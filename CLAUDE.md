# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Vietnamese Law Question-Answering system implementing a multi-stage RAG (Retrieval-Augmented Generation) pipeline. The system handles legal queries by retrieving relevant documents from vector databases, reranking them, and generating answers using LLMs. Non-legal queries (chitchat) are routed directly to OpenAI.

## Build & Run Commands

**Install dependencies:**
```bash
pip install -r backend/requirements.txt
```

**Run backend (FastAPI + Celery):**
```bash
sh backend/entrypoint.sh
```

**Fine-tune reranker model:**
```bash
cd retrieval
sh setup_env.sh
CUDA_VISIBLE_DEVICES=0 python create_data_rerank.py
sh finetune.sh
```

**Fine-tune LLM:**
```bash
cd finetune_llm
python gen_data.py
CUDA_VISIBLE_DEVICES=0 python finetune.py
python merge_with_base.py
```

## Architecture

### Project Structure
```
backend/          # FastAPI backend with Celery async processing
finetune_llm/     # LLM fine-tuning scripts (QLoRA with SFTTrainer)
retrieval/        # Retrieval/reranker model training
```

### RAG Pipeline Flow
1. **Intent Detection** (`brain.py:detect_route`) - Classify query as "legal" or "chitchat"
2. **Query Rewriting** (`brain.py:detect_user_intent`) - Rewrite query with conversation history context
3. **Multi-Model Retrieval** (`search_document/combine_search.py`):
   - BGE-M3 dense+sparse search via Qdrant
   - Multilingual-E5 dense search via Qdrant
   - Elasticsearch lexical search
4. **Reranking** (`search_document/rerank.py`) - BGE Reranker scores top documents
5. **LLM Generation** (`tasks.py:bot_answer_message`) - Local LLM generates answer with retrieved docs
6. **Web Search Fallback** (`agent.py`) - Tavily search if RAG cannot answer

### Key Backend Files
- `backend/src/app.py` - FastAPI endpoints (`/retrieval`, `/chat/complete`)
- `backend/src/tasks.py` - Celery tasks for async message processing
- `backend/src/brain.py` - Core LLM logic: intent detection, query rewriting, answer generation
- `backend/src/local_llm.py` - Local LLM wrapper (4-bit quantized T-VisStar-7B)
- `backend/src/agent.py` - ReAct agent with Tavily web search tool
- `backend/src/search_document/` - All retrieval components (BGE, E5, Elasticsearch, reranker)

### API Endpoints
- `POST /retrieval` - Retrieve and rerank documents for a query
- `POST /chat/complete` - Submit chat message (async via Celery)
- `GET /chat/complete_v2/{task_id}` - Poll async response

## Tech Stack

**Backend:** FastAPI, Celery, Redis (broker), MongoDB (chat history), Qdrant (vectors), Elasticsearch

**Models:**
- LLM: GPT-4o-mini (routing/rewriting), 1TuanPham/T-VisStar-7B-v0.1 (generation)
- Embeddings: BAAI/BGE-M3, intfloat/multilingual-e5-large
- Reranker: BGE-Reranker-v2-m3

**Fine-tuning:** Transformers, PEFT (QLoRA), BitsAndBytes (4-bit quantization), TRL (SFTTrainer)

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=sk-...        # Required for GPT-4o-mini
TAVILY_API_KEY=tvly-...      # Required for web search fallback
CELERY_BROKER_URL=redis://194.93.48.55:6379
CELERY_RESULT_BACKEND=redis://194.93.48.55:6379
```

## Infrastructure Defaults
- Backend API: port 8002
- Qdrant: 194.93.48.55:6333
- Elasticsearch: 194.93.48.55:9200
- Redis: 194.93.48.55:6379
- MongoDB: 194.93.48.55:27017

## Database Collections & Indices

**Qdrant collections:**
- `law_with_bge_round1` - BGE-M3 embeddings
- `law_with_e5_emb_not_finetune` - Multilingual-E5 embeddings

**Elasticsearch index:**
- `legal_data_part2` - Lexical search index

## Hardcoded Device Configuration

The codebase has hardcoded GPU device assignments:
- `local_llm.py` - Local LLM runs on `cuda:1`
- `search_vietnamese_legal.py` - Vietnamese legal search runs on `cuda:0`

Modify these if your GPU setup differs.

## Testing

No test suite is currently implemented. The `.gitignore` includes pytest patterns, indicating test infrastructure is planned but not yet in place.
