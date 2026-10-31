from typing import List, Literal
import requests
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app import config
from app.vectorstore import get_vectorstore

llm = ChatGroq(model=config.GROQ_MODEL, temperature=0, max_retries=3)

IDK_MESSAGE = (
    "I don't know based on the indexed documentation. "
    "Try rephrasing your question or ingest more relevant documents."
)


# ---------- structured outputs ----------
class QueryAnalysis(BaseModel):
    rewritten_query: str = Field(description="Expanded, retrieval-friendly search query")
    query_type: Literal["conceptual", "how-to", "troubleshooting", "api-reference"]


class BatchGrade(BaseModel):
    relevant_indices: List[int] = Field(
        description="Indices (as numbered in the input) of chunks that help answer the question"
    )


class Grounded(BaseModel):
    grounded: bool = Field(description="True only if every claim is supported by the context")


# ---------- nodes ----------
def analyze_query(state):
    chain = ChatPromptTemplate.from_messages([
        ("system",
         "You prepare questions for searching technical documentation about "
         + config.CORPUS_DESCRIPTION + ". "
         "Rewrite the question into a short, natural retrieval query: expand abbreviations "
         "and add at most 2-3 synonyms, but stay close to the original wording. Keep the "
         "documentation's own library/framework context, and do NOT introduce other "
         "frameworks, languages or technologies. Also classify its type. "
         "ALWAYS return the structured result, even if the question looks unrelated to the "
         "documentation: in that case just return the question lightly cleaned up. "
         "Never refuse or answer the question yourself."),
        ("human", "{q}"),
    ]) | llm.with_structured_output(QueryAnalysis)
    try:
        r = chain.invoke({"q": state["question"]})
        rewritten, qtype = r.rewritten_query, r.query_type
    except Exception:
        # Safety net: if analysis fails, search with the original question
        rewritten, qtype = state["question"], "conceptual"
    return {"rewritten_query": rewritten, "query_type": qtype,
            "retries": 0, "gen_attempts": 0, "used_web_search": False}

def retrieve(state):
    """Search with the rewritten query AND the original question, then merge and dedupe."""
    store = get_vectorstore()
    seen, docs = set(), []
    for q in (state["rewritten_query"], state["question"]):
        for d in store.similarity_search(q, k=config.TOP_K):
            cid = d.metadata.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                docs.append(d)
    return {"documents": docs}
def grade_documents(state):
    """Self-corrective step: LLM keeps only chunks relevant to the ORIGINAL question.
    All chunks are graded in one call (cheaper and friendlier to rate limits)."""
    docs = state["documents"]
    if not docs:
        return {"documents": []}
    numbered = "\n\n".join(f"[{i}]\n{d.page_content}" for i, d in enumerate(docs))
    chain = ChatPromptTemplate.from_messages([
        ("system",
         "You grade retrieved documentation chunks. Return the indices of chunks that contain "
         "information useful for answering the question. Be strict: a chunk that only shares "
         "keywords but does not help answer is irrelevant. Return an empty list if none help."),
        ("human", "Question: {q}\n\nChunks:\n{chunks}"),
    ]) | llm.with_structured_output(BatchGrade)
    result = chain.invoke({"q": state["question"], "chunks": numbered})
    keep = sorted({i for i in result.relevant_indices if 0 <= i < len(docs)})
    return {"documents": [docs[i] for i in keep]}


def rewrite_query(state):
    chain = ChatPromptTemplate.from_messages([
        ("system",
         "The previous search returned nothing relevant. Write a different search query "
         "(different keywords, broader or more specific). Return ONLY the query text."),
        ("human", "Original question: {q}\nPrevious query: {p}"),
    ]) | llm
    new = chain.invoke({"q": state["question"], "p": state["rewritten_query"]}).content.strip()
    return {"rewritten_query": new, "retries": state["retries"] + 1}


def web_search(state):
    """Optional fallback (Tavily) when the vector store has nothing relevant."""
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": config.TAVILY_API_KEY, "query": state["question"], "max_results": 3},
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception:
        results = []
    docs = [Document(page_content=x.get("content", ""),
                     metadata={"source": x.get("url", "web")}) for x in results if x.get("content")]
    return {"documents": docs, "used_web_search": bool(docs)}


def generate(state):
    docs = state["documents"]
    ctx = "\n\n".join(f"[{i+1}] (source: {d.metadata.get('source')})\n{d.page_content}"
                      for i, d in enumerate(docs))
    chain = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the question using ONLY the context. Cite sources inline using plain ASCII "
         "square brackets like [1], [2] (never 【 】 or other bracket styles). "
         "If the context is insufficient, say so. Include code examples when the context has them. "
         "Do not add facts that are not in the context."),
        ("human", "Context:\n{ctx}\n\nQuestion: {q}"),
    ]) | llm
    answer = chain.invoke({"ctx": ctx, "q": state["question"]}).content
    return {"generation": answer, "gen_attempts": state.get("gen_attempts", 0) + 1}


def check_grounding(state):
    """Self-RAG style hallucination check."""
    ctx = "\n\n".join(d.page_content for d in state["documents"])
    chain = ChatPromptTemplate.from_messages([
                ("system",
         "Decide whether the answer is supported by the context. Paraphrasing, citation "
         "markers like [1], and code examples copied or directly adapted from the context "
         "are fine. Set grounded=false ONLY if the answer states substantive facts that are "
         "clearly absent from the context."),
        ("human", "Context:\n{ctx}\n\nAnswer:\n{a}"),
    ]) | llm.with_structured_output(Grounded)
    r = chain.invoke({"ctx": ctx, "a": state["generation"]})
    return {"grounded": r.grounded}


def fallback(state):
    return {"generation": IDK_MESSAGE, "documents": [], "grounded": True}


# ---------- memory helper (used by the API, outside the graph) ----------
def condense_question(question: str, history: List[dict]) -> str:
    """Turn a follow-up into a standalone question using recent chat history."""
    if not history:
        return question
    hist = "\n".join(f"User: {h['question']}\nAssistant: {h['answer'][:300]}" for h in history[-3:])
    chain = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the follow-up question as a standalone question using the chat history. "
         "If it is already standalone, return it unchanged. Return ONLY the question."),
        ("human", "History:\n{h}\n\nFollow-up: {q}"),
    ]) | llm
    return chain.invoke({"h": hist, "q": question}).content.strip()
