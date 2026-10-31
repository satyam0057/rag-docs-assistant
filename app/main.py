import json
import datetime
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import requests

from app import config
from app.schemas import QueryRequest, QueryResponse, Source, FeedbackRequest
from app.graph.workflow import graph
from app.graph.nodes import condense_question
from app.ingestion import ingest_text, ingest_url
from app.vectorstore import get_vectorstore

app = FastAPI(title="RAG Technical Documentation Assistant", version="1.0.0")

SESSIONS: Dict[str, List[dict]] = {}  # in-memory chat history per session


def _section(meta: dict) -> Optional[str]:
    parts = [meta[k] for k in ("h1", "h2", "h3") if k in meta]
    return " > ".join(parts) or None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    history = SESSIONS.get(req.session_id, []) if req.session_id else []
    try:
        question = condense_question(req.question, history)
        out = graph.invoke({"question": question, "documents": [], "retries": 0})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow failed: {e}")

    if req.session_id:
        SESSIONS.setdefault(req.session_id, []).append(
            {"question": req.question, "answer": out["generation"]})

    sources = [Source(source=d.metadata.get("source", "unknown"),
                      section=_section(d.metadata),
                      snippet=d.page_content[:200])
               for d in out.get("documents", [])]
    return QueryResponse(
        answer=out["generation"], sources=sources,
        query_type=out.get("query_type", ""),
        rewritten_query=out.get("rewritten_query", ""),
        retries=out.get("retries", 0),
        grounded=out.get("grounded", True),
        used_web_search=out.get("used_web_search", False),
    )


@app.post("/ingest", status_code=201)
async def ingest(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)):
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide a file or a url")
    if file:
        if not file.filename.lower().endswith((".md", ".txt")):
            raise HTTPException(status_code=415, detail="Only .md and .txt files are supported")
        try:
            text = (await file.read()).decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 text")
        return {"source": file.filename, "chunks": ingest_text(text, source=file.filename)}
    try:
        return {"source": url, "chunks": ingest_url(url)}
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")


@app.get("/documents")
def documents():
    data = get_vectorstore().get(include=["metadatas"])
    counts: Dict[str, int] = {}
    for m in data["metadatas"]:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return {"total_chunks": sum(counts.values()),
            "documents": [{"source": s, "chunks": c} for s, c in sorted(counts.items())]}


@app.post("/feedback", status_code=201)
def feedback(fb: FeedbackRequest):
    rec = {**fb.model_dump(), "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(config.FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return {"status": "saved"}
