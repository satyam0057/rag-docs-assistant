from fastapi.testclient import TestClient
import app.main as main

client = TestClient(main.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_query_validation_empty_question():
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_ingest_requires_input():
    assert client.post("/ingest").status_code == 400


def test_ingest_rejects_bad_extension():
    r = client.post("/ingest", files={"file": ("x.pdf", b"data", "application/pdf")})
    assert r.status_code == 415


def test_feedback_validation():
    r = client.post("/feedback", json={"question": "q", "answer": "a", "rating": "meh"})
    assert r.status_code == 422


def test_feedback_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(main.config, "FEEDBACK_FILE", tmp_path / "fb.jsonl")
    r = client.post("/feedback", json={"question": "q", "answer": "a", "rating": "up", "comment": "good"})
    assert r.status_code == 201
    assert (tmp_path / "fb.jsonl").exists()


def test_query_with_mocked_graph(monkeypatch):
    class FakeGraph:
        def invoke(self, _):
            return {"generation": "Use {id} in the path [1].", "documents": [],
                    "query_type": "how-to", "rewritten_query": "path parameter",
                    "retries": 1, "grounded": True}
    monkeypatch.setattr(main, "graph", FakeGraph())
    monkeypatch.setattr(main, "condense_question", lambda q, h: q)
    r = client.post("/query", json={"question": "How do path params work?"})
    assert r.status_code == 200
    assert r.json()["retries"] == 1


def test_documents_endpoint(monkeypatch):
    class Store:
        def get(self, **kwargs):
            return {"metadatas": [{"source": "doc.md"}]}
    monkeypatch.setattr(main, "get_vectorstore", lambda: Store())
    r = client.get("/documents")
    assert r.status_code == 200
    assert "documents" in r.json()


def test_empty_upload_rejected():
    r = client.post("/ingest", files={"file": ("doc.md", b"  ", "text/plain")})
    assert r.status_code == 400


def test_grounding_routes_to_fallback_after_limit():
    from app.graph.workflow import route_after_grounding
    from app import config
    assert route_after_grounding({"grounded": False, "gen_attempts": 1}) == "generate"
    assert route_after_grounding({"grounded": False, "gen_attempts": config.MAX_GEN_ATTEMPTS}) == "fallback"
    assert route_after_grounding({"grounded": True, "gen_attempts": 1}) == "end"


def test_grading_retry_is_bounded(monkeypatch):
    from app.graph.workflow import route_after_grading
    from app import config
    monkeypatch.setattr(config, "TAVILY_API_KEY", None)
    assert route_after_grading({"documents": [object()], "retries": 0}) == "generate"
    assert route_after_grading({"documents": [], "retries": 0}) == "rewrite_query"
    assert route_after_grading({"documents": [], "retries": config.MAX_RETRIES}) == "fallback"
