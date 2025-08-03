from fastapi import FastAPI, Body, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import os
import json


from app.utils import load_text_documents, load_excel_rows
from app.retriever import SimpleRetriever
from app.ollama_client import query_ollama
from app.session import get_session, update_session

app = FastAPI(title="Ollama RAG Assistant")

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None 
    top_k: Optional[int] = 3


def normalize_ollama_response(raw):
    """
    Handles cases where the model wraps the real structured answer as a JSON string
    inside raw["response"], e.g. { ..., "response": "{ \"answer\": ..., ... }", ... }
    """

    result = {
        "answer": None,
        "sources": [],
        "confidence": "unknown",
        "original": raw
    }


    if isinstance(raw, dict) and "response" in raw:
        try:
            inner = json.loads(raw["response"])
            result["answer"] = inner.get("answer")
            result["sources"] = inner.get("sources", [])
            result["confidence"] = inner.get("confidence", "unknown")
        except Exception:

            result["answer"] = raw["response"]
    else:
        from app.ollama_client import extract_text, detect_uncertainty
        text = extract_text(raw)
        result["answer"] = text
        result["confidence"] = "low" if detect_uncertainty(text) else "high"

    return result

@app.on_event("startup")
def startup():

    text_docs = load_text_documents(DATA_FOLDER)
    excel_docs = load_excel_rows(DATA_FOLDER)
    all_docs = text_docs + excel_docs
    if not all_docs:
        print("WARNING: no documents loaded from data/ folder.")
    app.state.retriever = SimpleRetriever(all_docs)

@app.post("/query")
def query(req: QueryRequest):
    retriever: SimpleRetriever = app.state.retriever

    retrieved = retriever.retrieve(req.question, top_k=req.top_k)
    context_blocks = [f"[{r['source']}] {r['content']}" for r in retrieved]
    context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    history_summary = ""
    if req.session_id:
        session = get_session(req.session_id)
        if session.get("last_answer"):
            history_summary = f"Previous interaction: {session['last_answer']}\n"

    system_instruction = (
        "You are a concise technical assistant. Use the provided context to answer the question. "
        "If uncertain, say you don’t have enough info and ask for clarification."
    )
    prompt = f"""{system_instruction}

{history_summary}
Context:
{context_text}

User question:
{req.question}

Answer in a JSON object with fields: answer, sources (list), confidence ('high' or 'low').
"""

    try:
        ollama_resp = query_ollama("llama3.2:latest", prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {e}")


    normalized = normalize_ollama_response(ollama_resp["raw"])
    answer_text = normalized["answer"]


    print("=== raw ollama response ===")
    import pprint
    pprint.pprint(ollama_resp["raw"])
    print("==========================")


    if req.session_id:
        update_session(req.session_id, "last_question", req.question)
        update_session(req.session_id, "last_answer", answer_text)

  

    response = {
        "question": req.question,
        "retrieved": retrieved,
        "ollama": {
            "answer": answer_text,
            "sources": normalized["sources"],
            "confidence": normalized["confidence"],
            "duration_sec": ollama_resp["duration_sec"],
            "uncertain": ollama_resp["uncertain"]
        }
    }
    return response