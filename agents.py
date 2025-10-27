from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rich.pretty import pprint as rprint
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from typing_extensions import TypedDict

# record agent actions

# Pydantic model for a single step in the reasoning plan
class Step(BaseModel):
    sub_question: str = Field(description="A specific, answerable question for this step.")
    justification: str = Field(description="A brief explanation of why this step is necessary to answer the main query.")
    tool: Literal["search_10k", "search_web"] = Field(description="The tool to use for this step.")
    keywords: List[str] = Field(description="A list of critical keywords for searching relevant document sections.")
    document_section: Optional[str] = Field(description="A likely document section title (e.g., 'Item 1A. Risk Factors') to search within. Only for 'search_10k' tool.")

# Pydantic model for the overall plan
class Plan(BaseModel):
    steps: List[Step] = Field(description="A detailed, multi-step plan to answer the user's query.")

# TypedDict for storing the results of a completed step
class PastStep(TypedDict):
    step_index: int
    sub_question: str
    retrieved_docs: List[Document]
    summary: str

# The main state dictionary that will flow through the graph
class RAGState(TypedDict):
    original_question: str
    plan: Plan
    past_steps: List[PastStep]
    current_step_index: int
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    synthesized_context: str
    final_answer: str