"""Exercise the real graph routing with deterministic node replacements."""
from langchain_core.documents import Document
from app.graph import nodes, workflow


def test_graph_retries_then_answers(monkeypatch):
    doc = Document(page_content="Supported fact", metadata={"source": "doc.md"})
    monkeypatch.setattr(nodes, "analyze_query", lambda s: {"rewritten_query": s["question"], "query_type": "how-to", "retries": 0, "gen_attempts": 0, "used_web_search": False})
    monkeypatch.setattr(nodes, "retrieve", lambda s: {"documents": [] if s["retries"] == 0 else [doc]})
    monkeypatch.setattr(nodes, "grade_documents", lambda s: {"documents": s["documents"]})
    monkeypatch.setattr(nodes, "rewrite_query", lambda s: {"rewritten_query": "new query", "retries": s["retries"] + 1})
    monkeypatch.setattr(nodes, "generate", lambda s: {"generation": "Supported fact [1]", "gen_attempts": s["gen_attempts"] + 1})
    monkeypatch.setattr(nodes, "check_grounding", lambda s: {"grounded": True})
    out = workflow.build_graph().invoke({"question": "question"})
    assert out["retries"] == 1
    assert out["generation"] == "Supported fact [1]"
    assert out["documents"] == [doc]


def test_graph_never_returns_repeatedly_ungrounded_answer(monkeypatch):
    doc = Document(page_content="context")
    monkeypatch.setattr(nodes, "analyze_query", lambda s: {"rewritten_query": s["question"], "query_type": "conceptual", "retries": 0, "gen_attempts": 0, "used_web_search": False})
    monkeypatch.setattr(nodes, "retrieve", lambda s: {"documents": [doc]})
    monkeypatch.setattr(nodes, "grade_documents", lambda s: {"documents": s["documents"]})
    monkeypatch.setattr(nodes, "generate", lambda s: {"generation": "Unsupported", "gen_attempts": s["gen_attempts"] + 1})
    monkeypatch.setattr(nodes, "check_grounding", lambda s: {"grounded": False})
    out = workflow.build_graph().invoke({"question": "question"})
    assert out["generation"] == nodes.IDK_MESSAGE
    assert out["grounded"] is False
    assert out["documents"] == []


def test_normalize_model_citation_brackets():
    assert nodes.normalize_citations("Example 【2】 and [1]") == "Example [2] and [1]"
