# =====================================================
# rag_pipeline.py
# -----------------------------------------------------
# Builds the Retrieval-Augmented Generation pipeline:
#   1. Load CSV -> Documents
#   2. Recursive metadata-aware chunking
#   3. HuggingFace embeddings
#   4. FAISS vector store (semantic search)
#   5. BM25 (keyword search)
#   6. Hybrid search = FAISS + BM25
#   7. Grounded response via OpenAI GPT
#
# Style: simple, functional, beginner-friendly.
# =====================================================

import os
import pandas as pd
from dotenv import load_dotenv

# LangChain core building blocks
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load env variables (OPENAI_API_KEY)
load_dotenv()

# -----------------------------------------------------
# Paths and config (kept simple as module-level constants)
# -----------------------------------------------------
DATA_PATH = "data/customer_support_data.csv"
VECTORSTORE_PATH = "vectorstore/faiss_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Globals filled in by build_rag_pipeline()
# (we store them as module-level so agents/tools can reuse them)
faiss_store = None
bm25_retriever = None
llm = None
prompt_template = None


# -----------------------------------------------------
# 1. Load CSV and convert each row into a Document with metadata
# -----------------------------------------------------
def load_documents(csv_path=DATA_PATH):
    """
    Read the CSV and turn each row into a LangChain Document.
    The 'page_content' = customer query + agent response
    so that retrieval can match on either side.
    Metadata holds category/intent — useful for filtering later.
    """
    df = pd.read_csv(csv_path)

    documents = []
    for _, row in df.iterrows():
        # Combine query + response so the chunk is self-contained
        content = (
            f"Customer Query: {row.get('instruction', '')}\n"
            f"Support Response: {row.get('response', '')}"
        )

        # Metadata travels with each chunk -> "metadata-aware chunking"
        metadata = {
            "category": str(row.get("category", "general")),
            "intent": str(row.get("intent", "unknown")),
        }

        documents.append(Document(page_content=content, metadata=metadata))

    print(f"Loaded {len(documents)} documents from {csv_path}")
    return documents


# -----------------------------------------------------
# 2. Recursive chunking with overlap (metadata preserved)
# -----------------------------------------------------
def chunk_documents(documents, chunk_size=400, chunk_overlap=60):
    """
    RecursiveCharacterTextSplitter splits long text smartly
    (tries paragraphs -> sentences -> words).
    Metadata is automatically carried into each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# -----------------------------------------------------
# 3. Build FAISS vector store from chunks
# -----------------------------------------------------
def build_or_load_faiss(chunks, embeddings):
    """
    If a FAISS index already exists on disk, just load it.
    Otherwise build it once and save -> avoids re-embedding on every run.
    """
    if os.path.exists(VECTORSTORE_PATH):
        print("FAISS index found on disk. Loading...")
        store = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        print("Building FAISS index (this takes ~1-2 minutes)...")
        store = FAISS.from_documents(chunks, embeddings)
        os.makedirs("vectorstore", exist_ok=True)
        store.save_local(VECTORSTORE_PATH)
        print(f"Saved FAISS index to {VECTORSTORE_PATH}")
    return store


# -----------------------------------------------------
# 4. Build a BM25 keyword retriever (for hybrid search)
# -----------------------------------------------------
def build_bm25(chunks, k=4):
    """
    BM25 = classic keyword scoring. Good at exact-word matches
    where embeddings sometimes miss (e.g. order IDs, product names).
    """
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


# -----------------------------------------------------
# 5. Hybrid search = semantic + keyword
# -----------------------------------------------------
def hybrid_search(query, k=4):
    """
    Combine FAISS (semantic) results with BM25 (keyword) results.
    De-duplicates by page_content. This is the function agents call.
    """
    # Semantic results
    semantic_docs = faiss_store.similarity_search(query, k=k)
    # Keyword results
    keyword_docs = bm25_retriever.invoke(query)

    # Merge while removing duplicates
    seen = set()
    merged = []
    for doc in semantic_docs + keyword_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)

    # Return top-k merged results
    return merged[:k]


# -----------------------------------------------------
# 6. Prompt template for grounded generation
# -----------------------------------------------------
GROUNDED_PROMPT = """You are a helpful enterprise customer support assistant.
Use ONLY the context below to answer the user's question.
If the answer is not present in the context, say:
"I don't have enough information — please escalate to a human agent."

Context:
{context}

Question: {question}

Answer:"""


# -----------------------------------------------------
# 7. Generate a grounded answer using the LLM
# -----------------------------------------------------
def generate_answer(query):
    """
    1. Retrieve docs via hybrid search
    2. Stuff them into the prompt
    3. Ask GPT to answer grounded ONLY in the retrieved context
    """
    docs = hybrid_search(query, k=4)

    # Combine all retrieved chunks into one context string
    context_text = "\n\n".join([d.page_content for d in docs])

    # Fill in the prompt
    final_prompt = prompt_template.format(context=context_text, question=query)

    # Call GPT
    response = llm.invoke(final_prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    # Return both the answer and the source docs (for transparency)
    return {
        "answer": answer,
        "sources": [
            {"content": d.page_content, "metadata": d.metadata} for d in docs
        ],
    }


# -----------------------------------------------------
# 8. The "build everything once" entry point
# -----------------------------------------------------
def build_rag_pipeline():
    """
    Run this ONCE at app startup. It sets up all the globals
    (faiss_store, bm25_retriever, llm, prompt_template).
    """
    global faiss_store, bm25_retriever, llm, prompt_template

    # Step A: load documents from CSV
    docs = load_documents()

    # Step B: chunk with overlap
    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=60)

    # Step C: embeddings (free, runs locally via HuggingFace)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Step D: FAISS (semantic) + BM25 (keyword)
    faiss_store = build_or_load_faiss(chunks, embeddings)
    bm25_retriever = build_bm25(chunks, k=4)

    # Step E: the LLM that will generate grounded responses
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
    )

    # Step F: prompt template
    prompt_template = PromptTemplate(
        template=GROUNDED_PROMPT,
        input_variables=["context", "question"],
    )

    print("RAG pipeline ready.")
    return {
        "faiss_store": faiss_store,
        "bm25_retriever": bm25_retriever,
        "llm": llm,
    }


# -----------------------------------------------------
# Quick local test (run: python -m app.rag.rag_pipeline)
# -----------------------------------------------------
if __name__ == "__main__":
    build_rag_pipeline()
    result = generate_answer("How do I cancel my order?")
    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- SOURCES ---")
    for s in result["sources"]:
        print("-", s["metadata"], "|", s["content"][:80], "...")
