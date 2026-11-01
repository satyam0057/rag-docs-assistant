"""Minimal Gradio UI. Start the API first, then run:  python ui.py"""
import uuid
import requests
import gradio as gr

API = "http://127.0.0.1:8000"


def ask(question, session, last):
    session = session or str(uuid.uuid4())
    r = requests.post(f"{API}/query", json={"question": question, "session_id": session}, timeout=120)
    if r.status_code != 200:
        return f"Error {r.status_code}: {r.text}", "", session, last
    d = r.json()
    last = {"question": question, "answer": d["answer"]}
    srcs = "\n".join(f"- **{s['source']}** ({s['section'] or 'n/a'})" for s in d["sources"]) or "_none_"
    meta = (f"\n\n---\n*type: {d['query_type']} | retries: {d['retries']} | "
            f"grounded: {d['grounded']} | web search: {d['used_web_search']}*")
    return d["answer"] + meta, srcs, session, last


def rate(rating, last):
    if not last:
        return "Ask a question first."
    r = requests.post(f"{API}/feedback", json={**last, "rating": rating}, timeout=30)
    return "Thanks for the feedback!" if r.status_code == 201 else f"Error: {r.text}"


with gr.Blocks(title="RAG Docs Assistant") as demo:
    gr.Markdown("# RAG Technical Documentation Assistant")
    session = gr.State("")
    last = gr.State({})
    q = gr.Textbox(label="Your question", placeholder="How do I declare a path parameter?")
    btn = gr.Button("Ask", variant="primary")
    ans = gr.Markdown()
    gr.Markdown("### Sources")
    src = gr.Markdown()
    with gr.Row():
        up, down = gr.Button("👍"), gr.Button("👎")
    status = gr.Markdown()
    btn.click(ask, [q, session, last], [ans, src, session, last])
    q.submit(ask, [q, session, last], [ans, src, session, last])
    up.click(lambda value: rate("up", value), last, status)
    down.click(lambda value: rate("down", value), last, status)

if __name__ == "__main__":
    demo.launch()
