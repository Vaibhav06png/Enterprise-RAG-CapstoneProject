# =====================================================
# main.py  --  FastAPI REST API
# -----------------------------------------------------
# Exposes the multi-agent RAG workflow over HTTP.
#
# Endpoints:
#   GET  /       -> health check
#   POST /ask    -> run the full agentic workflow
# =====================================================

from fastapi import FastAPI
from pydantic import BaseModel

# Our own modules
from app.rag import rag_pipeline
from app.agents.agents import run_agentic_workflow

# Create the FastAPI app
app = FastAPI(title="Enterprise RAG + CrewAI API")

# Build the RAG pipeline ONCE at startup (avoids reloading per request)
print("Initializing RAG pipeline on startup...")
rag_pipeline.build_rag_pipeline()
print("Ready to serve requests.")


# Request body schema
class Query(BaseModel):
    query: str


# -----------------------------------------------------
# Health check
# -----------------------------------------------------
@app.get("/")
def home():
    return {"message": "Enterprise RAG + CrewAI API is running"}


# -----------------------------------------------------
# Main endpoint: run the multi-agent workflow
# -----------------------------------------------------
@app.post("/ask")
def ask(q: Query):
    """
    1. Receive user query
    2. Run CrewAI workflow (4 agents)
    3. Return final answer + sources + escalation status
    """
    result = run_agentic_workflow(q.query)

    return {
        "response": result["final_response"],
        "sources": result["retrieved_sources"],
        "sentiment": result["sentiment"],
        "escalation": result["escalation"],
        "escalate_flag": result["escalate_flag"],
    }
