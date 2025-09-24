from typing import List, Optional
from pydantic import BaseModel, Field


class Context(BaseModel):
    text: str = Field(..., description="Source text")
    source: str = Field(..., description="File name")


class RAGAnswer(BaseModel):
    question: str = Field(..., description="User question")
    answer: str = Field(..., description="Answer to the question")
    contexts: List[Context] = Field(default_factory=list)
