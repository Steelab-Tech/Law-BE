# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Vietnamese Law Question-Answering system implementing a multi-stage RAG (Retrieval-Augmented Generation) pipeline. The system handles legal queries by retrieving relevant documents from vector databases, reranking them, and generating answers using LLMs. Non-legal queries (chitchat) are routed directly to OpenAI.

## Build & Run Commands

**Install dependencies:**
```bash
pip install -r backend/requirements.txt
pip install -r chatbot-ui/requirements.txt
```

**Run backend (FastAPI + Celery):**
```bash
sh backend/entrypoint.sh
```

**Run frontend (Streamlit):**
```bash
sh chatbot-ui/entrypoint.sh
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
chatbot-ui/       # Streamlit frontend interface
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
5. **LLM Generation** (`brain.py:detect_answer`) - GPT-4o-mini generates answer
6. **Web Search Fallback** (`agent.py`) - Tavily search if RAG cannot answer

### Key Backend Files
- `backend/src/app.py` - FastAPI endpoints (`/retrieval`, `/chat/complete`)
- `backend/src/tasks.py` - Celery tasks for async message processing
- `backend/src/brain.py` - Core LLM logic: intent detection, query rewriting, answer generation
- `backend/src/agent.py` - ReAct agent with Tavily web search tool
- `backend/src/search_document/` - All retrieval components (BGE, E5, Elasticsearch, reranker)

### API Endpoints
- `POST /retrieval` - Retrieve and rerank documents for a query
- `POST /chat/complete` - Submit chat message (async via Celery)
- `GET /chat/complete_v2/{task_id}` - Poll async response

## Tech Stack

**Backend:** FastAPI, Celery, Redis (broker), MongoDB (chat history), Qdrant (vectors), Elasticsearch

**Models:**
- LLM: GPT-4o-mini (routing/generation), 1TuanPham/T-VisStar-7B-v0.1 (fine-tuned)
- Embeddings: BAAI/BGE-M3, intfloat/multilingual-e5-large
- Reranker: BGE-Reranker-v2-m3

**Fine-tuning:** Transformers, PEFT (QLoRA), BitsAndBytes (4-bit quantization), TRL (SFTTrainer)

## Infrastructure Defaults
- Backend API: port 8002
- Streamlit UI: port 8051
- Qdrant: localhost:6333
- Elasticsearch: localhost:9200
- Redis: localhost:6379
- MongoDB: 10.0.0.138:27017
