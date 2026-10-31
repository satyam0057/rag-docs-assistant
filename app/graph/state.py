from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str              # standalone user question, never overwritten
    rewritten_query: str       # what we actually search with (changes on retries)
    query_type: str            # conceptual | how-to | troubleshooting | api-reference
    documents: List[Document]  # retrieved, then overwritten with graded-relevant docs
    generation: str            # final answer
    retries: int               # number of query rewrites so far (drives routing)
    gen_attempts: int          # number of generation attempts (drives grounding loop)
    grounded: bool             # hallucination check result
    used_web_search: bool      # True if answer came from the web fallback
