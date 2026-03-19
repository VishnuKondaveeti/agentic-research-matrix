# 🔬 Agentic Research Matrix

An intelligent research platform that automatically collects research papers, processes them with AI, builds a RAG knowledge base, and provides multi-agent analysis with a conversational interface.

## Architecture

```
┌──────────────────────────────────────────────────┐
│             Professional Dashboard               │
│   (HTML5/CSS3/JS • Vis.js • Plotly • Lucide)     │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│              FastAPI Backend                      │
│   /api/search │ /api/query │ /api/report          │
│   /webhook/new-topic │ /webhook/report            │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│           Orchestrator Agent                      │
│  ┌──────────┬──────────┬──────────┬────────────┐ │
│  │ Research │ Retrieval│ Analysis │  Report     │ │
│  │  Agent   │  Agent   │  Agent   │  Agent     │ │
│  └──────────┴──────────┴──────────┴────────────┘ │
│  ┌──────────┐                                    │
│  │ Advisor  │                                    │
│  │  Agent   │                                    │
│  └──────────┘                                    │
└───────┬──────────────────────┬───────────────────┘
        │                      │
┌───────▼──────────┐  ┌───────▼──────────┐
│   RAG Pipeline   │  │ Knowledge Graph  │
│ ChromaDB Vector  │  │    (Neo4j)       │
│    Database      │  │                  │
└───────┬──────────┘  └──────────────────┘
        │
┌───────▼──────────────────────────────────────────┐
│           Paper Collection                        │
│  arXiv API │ Semantic Scholar │ CORE API          │
└──────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Set LLM_PROVIDER (gemini, ollama, or openai)
# Add GOOGLE_API_KEY, OPENAI_API_KEY, or OLLAMA_HOST
```

### 3. Launch the Matrix (Local)
```bash
python run_all.py
```

### 4. Docker Deployment (Recommended)
For a production-ready setup with Neo4j included:
```bash
docker-compose up --build -d
```
- **Web UI & API**: http://localhost:8000
- **Neo4j Browser**: http://localhost:7474



## Features

| Feature | Description |
|---------|-------------|
| **Paper Collection** | Search arXiv, Semantic Scholar, CORE APIs |
| **PDF Processing** | Extract, clean, chunk text from papers |
| **Vector Database** | ChromaDB semantic search over papers |
| **RAG Chat** | Ask questions, get answers with citations |
| **Literature Reviews** | Auto-generate structured reports |
| **Research Advisor** | Identify gaps and suggest directions |
| **Trend Detection** | Cluster analysis and keyword trending |
| **Knowledge Graph** | Neo4j paper/author/topic relationships |
| **n8n Automation** | Webhook-triggered research workflows |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/search` | POST | Search & collect papers |
| `/api/query` | POST | RAG query |
| `/api/report` | POST | Generate report |
| `/api/papers` | GET | List papers |
| `/api/ingest` | POST | Trigger ingestion |
| `/api/analyze` | POST | Analyze papers |
| `/api/advise` | POST | Get research advice |
| `/api/trends` | GET | Trend analysis |
| `/webhook/new-topic` | POST | n8n: new topic trigger |
| `/webhook/report` | POST | n8n: report trigger |

## Project Structure

```
Agentic Research Matrix/
├── config/settings.py         # Pydantic settings

├── collectors/                # Paper collection clients
│   ├── arxiv_client.py
│   ├── semantic_scholar_client.py
│   ├── core_client.py
│   └── paper_manager.py
├── processing/                # Document processing
│   ├── pdf_extractor.py
│   ├── text_cleaner.py
│   ├── chunker.py
│   └── pipeline.py
├── rag/                       # RAG system
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   └── rag_chain.py
├── agents/                    # AI agents
│   ├── base_agent.py
│   ├── research_agent.py
│   ├── retrieval_agent.py
│   ├── analysis_agent.py
│   ├── report_agent.py
│   ├── advisor_agent.py
│   └── orchestrator_agent.py
├── graph/knowledge_graph.py   # Neo4j integration
├── analytics/trend_detector.py
├── api/                       # FastAPI backend
│   ├── main.py
│   ├── routes.py
│   ├── models.py
│   └── webhooks.py
├── ui/app.py                  # Streamlit UI
├── workflows/n8n_templates/   # n8n workflow templates
├── logs/system_logger.py
├── memory/query_memory.py
└── data/                      # Runtime data (auto-created)
```

## Multi-Agent System

| Agent | Role |
|-------|------|
| **Orchestrator** | Routes tasks, coordinates agents |
| **Research** | Searches databases, downloads papers |
| **Retrieval** | Queries vector DB for relevant chunks |
| **Analysis** | Extracts insights using LLM |
| **Report** | Generates literature reviews |
| **Advisor** | Identifies gaps, suggests directions |

## Technology Stack

- **Backend:** Python, FastAPI, Uvicorn
- **AI/LLM:** Google Gemini, Ollama (Local), OpenAI
- **Vector DB:** ChromaDB
- **Knowledge Graph:** Neo4j (Real-time analytics)
- **Frontend:** Professional Dashboard (HTML5, Vanilla JS, Vis.js, Plotly)
- **Automation:** n8n Webhooks
- **Analytics:** scikit-learn (PCA 3D Embeddings), Plotly
