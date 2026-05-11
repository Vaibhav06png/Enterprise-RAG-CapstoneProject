# Enterprise RAG Pipeline with Multi-AI Agentic Workflow

Capstone Project 2 — A Retrieval-Augmented Generation system enhanced with a CrewAI multi-agent workflow for **Enterprise Customer Support Automation**, served via **FastAPI**, with a simple **Streamlit** UI, **Docker** deployment, and **GitHub Actions CI/CD**.

---

## Architecture

```
Streamlit UI  ─►  FastAPI /ask  ─►  CrewAI Multi-Agent Workflow
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
            Query Understanding   RAG Retrieval        Sentiment
                  Agent           Agent (Tool)          Agent
                                       │                    │
                                       ▼                    ▼
                              Hybrid Search           Escalation Agent
                              (FAISS + BM25)
                                       │
                                       ▼
                              Grounded GPT Answer
```

---

## Tech Stack

- **LangChain** — RAG building blocks
- **FAISS** — vector store (semantic search)
- **BM25** — keyword search (hybrid retrieval)
- **HuggingFaceEmbeddings** — `all-MiniLM-L6-v2` (free, local)
- **OpenAI GPT** — grounded answer generation
- **CrewAI** — multi-agent orchestration
- **FastAPI** — REST API
- **Streamlit** — minimal UI
- **Docker + GitHub Actions** — deployment + CI/CD

---

## Folder Structure

```
enterprise-rag-crewai/
├── app/
│   ├── agents/agents.py        # CrewAI multi-agent workflow
│   ├── rag/rag_pipeline.py     # RAG: load → chunk → embed → hybrid search
│   └── main.py                 # FastAPI backend
├── data/customer_support_data.csv
├── vectorstore/                # FAISS index (auto-created)
├── tests/test_smoke.py
├── .github/workflows/ci-cd.yml
├── streamlit.py                # Streamlit frontend
├── evaluation.py               # ROUGE + relevance comparison
├── download_dataset.py
├── requirements.txt
├── Dockerfile
├── .env
└── README.md
```

---

## Setup

```bash
# 1. Create + activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Download the dataset (one-time)
python download_dataset.py
```

---

## Run

### Run the FastAPI backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check: `http://localhost:8000/`
Docs (Swagger): `http://localhost:8000/docs`

### Run the Streamlit UI

```bash
streamlit run streamlit.py
```

### Test the API directly

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I cancel my order?"}'
```

---

## Run with Docker

```bash
docker build -t enterprise-rag-crewai .
docker run -p 8000:8000 --env-file .env enterprise-rag-crewai
```

---

## Evaluation

```bash
python evaluation.py
```

This runs both **baseline RAG** and the **agentic workflow** on a few sample queries and outputs ROUGE-1, ROUGE-L, and a simple relevance score to `evaluation_results.csv`.

---

## CI/CD

Pushing to `main` triggers `.github/workflows/ci-cd.yml`, which:
1. Installs dependencies
2. Runs tests (`pytest tests/`)
3. Builds the Docker image to verify it builds cleanly

---
