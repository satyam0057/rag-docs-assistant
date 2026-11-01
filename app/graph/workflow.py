from langgraph.graph import StateGraph, START, END
from app import config
from app.graph.state import GraphState
from app.graph import nodes


def route_after_grading(state) -> str:
    if state["documents"]:
        return "generate"
    if state["retries"] < config.MAX_RETRIES:
        return "rewrite_query"
    return "web_search" if config.TAVILY_API_KEY else "fallback"


def route_after_web_search(state) -> str:
    return "generate" if state["documents"] else "fallback"


def route_after_grounding(state) -> str:
    if state["grounded"]:
        return "end"
    if state["gen_attempts"] >= config.MAX_GEN_ATTEMPTS:
        return "fallback"
    return "generate"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("analyze_query", nodes.analyze_query)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("web_search", nodes.web_search)
    g.add_node("generate", nodes.generate)
    g.add_node("check_grounding", nodes.check_grounding)
    g.add_node("fallback", nodes.fallback)

    g.add_edge(START, "analyze_query")
    g.add_edge("analyze_query", "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grading, {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
        "web_search": "web_search",
        "fallback": "fallback",
    })
    g.add_edge("rewrite_query", "retrieve")  # the self-correction loop
    g.add_conditional_edges("web_search", route_after_web_search,
                            {"generate": "generate", "fallback": "fallback"})
    g.add_edge("generate", "check_grounding")
    g.add_conditional_edges("check_grounding", route_after_grounding,
                            {"end": END, "generate": "generate", "fallback": "fallback"})
    g.add_edge("fallback", END)
    return g.compile()


graph = build_graph()

if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
