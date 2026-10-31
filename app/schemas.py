from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    session_id: Optional[str] = Field(None, description="Send the same id to enable follow-ups")


class Source(BaseModel):
    source: str
    section: Optional[str] = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    query_type: str
    rewritten_query: str
    retries: int
    grounded: bool
    used_web_search: bool = False


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None
