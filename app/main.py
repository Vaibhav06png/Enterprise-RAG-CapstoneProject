from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import rag_pipeline
from app.agents.agents import run_agentic_workflow

app = FastAPI(title="Enterprise RAG + CrewAI API")


# Load RAG pipeline during startup 
@app.on_event("startup")
async def load_pipeline():
    try:
        print(" Initializing RAG pipeline...")
        rag_pipeline.build_rag_pipeline()
        print(" RAG pipeline ready")
    except Exception as e:
        print("Error initializing pipeline:", e)


class Query(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Enterprise RAG + CrewAI API is running"}


@app.post("/ask")
def ask(q: Query):
    result = run_agentic_workflow(q.query)

    return {
        "response": result["final_response"],
        "sources": result["retrieved_sources"],
        "sentiment": result["sentiment"],
        "escalation": result["escalation"],
        "escalate_flag": result["escalate_flag"],
    }