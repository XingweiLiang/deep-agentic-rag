from typing import Dict, List
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.pretty import pprint as rprint
from typing_extensions import TypedDict
from agents import RAGState, Plan, Step, PastStep
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

console = Console()


# Import retrieval functions from the separate module
from retrieval_strategies import vector_search_only, bm25_search_only, hybrid_search

# Other functions that still need to be initialized from main module
web_search_function = None
rerank_documents_function = None
policy_agent = None

def initialize_graphs_node(web_search, rerank_func, policy_ag):
    """Initialize the remaining global functions and agents needed by graph nodes"""
    global web_search_function, rerank_documents_function, policy_agent
    web_search_function = web_search
    rerank_documents_function = rerank_func
    policy_agent = policy_ag

# Import the agents and config from the main module
# You can either import them directly from deep-agentic-rag or create them here
# For now, we'll create them here to make this module self-contained

# Central Configuration Dictionary
from config import config

# Utility function
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Create the LLM instance
reasoning_llm = ChatOpenAI(model=config["reasoning_llm"], temperature=0)

# Create planner_agent
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research planner. Your task is to create a clear, multi-step plan to answer a complex user query by retrieving information from multiple sources.
You have two tools available:
1. `search_10k`: Use this to search for information within NVIDIA's 2023 10-K financial filing. This is best for historical facts, financial data, and stated company policies or risks from that specific time period.
2. `search_web`: Use this to search the public internet for recent news, competitor information, or any topic that is not specific to NVIDIA's 2023 10-K.

Decompose the user's query into a series of simple, sequential sub-questions. For each step, decide which tool is more appropriate.
For `search_10k` steps, also identify the most likely section of the 10-K (e.g., 'Item 1A. Risk Factors', 'Item 7. Management's Discussion and Analysis...').
It is critical to use the exact section titles found in a 10-K filing where possible."""),
    ("human", "User Query: {question}")
])
planner_agent = planner_prompt | reasoning_llm.with_structured_output(Plan)


# Create query_rewriter_agent
query_rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at transforming vague research questions into precise, effective search queries.
Transform the given sub-question into a targeted search query by:
1. Incorporating relevant keywords from the provided list
2. Using past context to avoid redundancy
3. Making the query specific and searchable

Return only the rewritten search query, nothing else."""),
    ("human", "Sub-question: {sub_question}\nKeywords: {keywords}\nPast context: {past_context}")
])
query_rewriter_agent = query_rewriter_prompt | reasoning_llm | StrOutputParser()

# Create RetrievalDecision class
from pydantic import BaseModel, Field
from typing import Literal

class RetrievalDecision(BaseModel):
    strategy: Literal["vector_search", "keyword_search", "hybrid_search"] = Field(description="The chosen retrieval strategy")
    justification: str = Field(description="Brief explanation of why this strategy is most appropriate for the query")

# Create retrieval_supervisor_agent
retrieval_supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at selecting the optimal retrieval strategy for different types of queries.

Choose the best strategy:
- **vector_search**: Best for conceptual, thematic, or semantic queries that require understanding context and meaning
- **keyword_search**: Best for specific terms, exact phrases, technical jargon, or when looking for precise factual information
- **hybrid_search**: Best for complex queries that benefit from both semantic understanding and exact term matching

Consider the nature of the query and what type of information is most likely to help answer it."""),
    ("human", "Query: {sub_question}")
])
retrieval_supervisor_agent = retrieval_supervisor_prompt | reasoning_llm.with_structured_output(RetrievalDecision)

# Create distiller_agent
distiller_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting and synthesizing relevant information from multiple documents.
Extract only the information that directly addresses the given question. Be concise but comprehensive.
Maintain important details, figures, and context while removing irrelevant content."""),
    ("human", "Question: {question}\n\nDocuments:\n{context}")
])
distiller_agent = distiller_prompt | reasoning_llm | StrOutputParser()


# Create reflection_agent
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research analyst. Your task is to create a concise summary of the key findings that directly answer the given sub-question.
Focus on the most important information and insights. Be specific and factual."""),
    ("human", "Sub-question: {sub_question}\n\nContext: {context}")
])
reflection_agent = reflection_prompt | reasoning_llm | StrOutputParser()


class Decision(BaseModel):
    next_action: Literal["CONTINUE_PLAN", "FINISH"]
    justification: str

policy_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a master strategist. Your role is to analyze the research progress and decide the next action.
You have the original question, the initial plan, and a log of completed steps with their summaries.
- If the collected information in the Research History is sufficient to comprehensively answer the Original Question, decide to FINISH.
- Otherwise, if the plan is not yet complete, decide to CONTINUE_PLAN."""),
    ("human", "Original Question: {question}\n\nInitial Plan:\n{plan}\n\nResearch History (Completed Steps):\n{history}")
])
policy_agent = policy_prompt | reasoning_llm.with_structured_output(Decision)



# define graph node, node functions, and edges below
def get_past_context_str(past_steps: List[PastStep]) -> str:
    return "\n\n".join([f"Step {s['step_index']}: {s['sub_question']}\nSummary: {s['summary']}" for s in past_steps])

def plan_node(state: RAGState) -> Dict:
    console.print("--- 🧠: Generating Plan ---")
    plan = planner_agent.invoke({"question": state["original_question"]})
    rprint(plan)
    return {"plan": plan, "current_step_index": 0, "past_steps": []}

def retrieval_node(state: RAGState) -> Dict:
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    console.print(f"--- 🔍: Retrieving from 10-K (Step {current_step_index + 1}: {current_step.sub_question}) ---")
    past_context = get_past_context_str(state['past_steps'])
    rewritten_query = query_rewriter_agent.invoke({
        "sub_question": current_step.sub_question,
        "keywords": current_step.keywords,
        "past_context": past_context
    })
    console.print(f"  Rewritten Query: {rewritten_query}")
    
    # NEW: Adaptive Retrieval Strategy
    retrieval_decision = retrieval_supervisor_agent.invoke({"sub_question": rewritten_query})
    console.print(f"  Supervisor Decision: Use `{retrieval_decision.strategy}`. Justification: {retrieval_decision.justification}")

    if retrieval_decision.strategy == 'vector_search':
        retrieved_docs = vector_search_only(rewritten_query, section_filter=current_step.document_section, k=config['top_k_retrieval'])
    elif retrieval_decision.strategy == 'keyword_search':
        retrieved_docs = bm25_search_only(rewritten_query, k=config['top_k_retrieval'])
    else: # hybrid_search
        retrieved_docs = hybrid_search(rewritten_query, section_filter=current_step.document_section, k=config['top_k_retrieval'])
    
    return {"retrieved_docs": retrieved_docs}

def web_search_node(state: RAGState) -> Dict:
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    console.print(f"--- 🌐: Searching Web (Step {current_step_index + 1}: {current_step.sub_question}) ---")
    past_context = get_past_context_str(state['past_steps'])
    rewritten_query = query_rewriter_agent.invoke({
        "sub_question": current_step.sub_question,
        "keywords": current_step.keywords,
        "past_context": past_context
    })
    console.print(f"  Rewritten Query: {rewritten_query}")
    retrieved_docs = web_search_function(rewritten_query)
    return {"retrieved_docs": retrieved_docs}

def rerank_node(state: RAGState) -> Dict:
    console.print("--- 🎯: Reranking Documents ---")
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    reranked_docs = rerank_documents_function(current_step.sub_question, state["retrieved_docs"])
    console.print(f"  Reranked to top {len(reranked_docs)} documents.")
    return {"reranked_docs": reranked_docs}

def compression_node(state: RAGState) -> Dict:
    console.print("--- ✂️: Distilling Context ---")
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    context = format_docs(state["reranked_docs"])
    synthesized_context = distiller_agent.invoke({"question": current_step.sub_question, "context": context})
    console.print(f"  Distilled Context Snippet: {synthesized_context[:200]}...")
    return {"synthesized_context": synthesized_context}

def reflection_node(state: RAGState) -> Dict:
    console.print("--- 🤔: Reflecting on Findings ---")
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    summary = reflection_agent.invoke({"sub_question": current_step.sub_question, "context": state['synthesized_context']})
    console.print(f"  Summary: {summary}")
    new_past_step = {
        "step_index": current_step_index + 1,
        "sub_question": current_step.sub_question,
        "retrieved_docs": state['reranked_docs'],
        "summary": summary
    }
    return {"past_steps": state["past_steps"] + [new_past_step], "current_step_index": current_step_index + 1}

def final_answer_node(state: RAGState) -> Dict:
    console.print("--- ✅: Generating Final Answer with Citations ---")
    # Create a consolidated context using summaries instead of full documents
    final_context = ""
    for i, step in enumerate(state['past_steps']):
        final_context += f"\n--- Step {i+1}: {step['sub_question']} ---\n"
        # Limit summary length to prevent token overflow
        summary = step['summary']
        if len(summary) > 500:
            summary = summary[:500] + "..."
        final_context += f"{summary}\n"
        
        # Add just the first few sources for citations
        sources = []
        for doc in step['retrieved_docs'][:2]:  # Limit to first 2 docs
            source = doc.metadata.get('section') or doc.metadata.get('source', 'Unknown')
            if source not in sources:
                sources.append(source)
        final_context += f"Sources: {', '.join(sources)}\n\n"
    
    final_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial analyst. Synthesize the research findings from internal documents and web searches into a comprehensive, multi-paragraph answer for the user's original question.
Your answer must be grounded in the provided context. At the end of any sentence that relies on specific information, you MUST add a citation. For 10-K documents, use [Source: <section title>]. For web results, use [Source: <URL>]."""),
        ("human", "Original Question: {question}\n\nResearch History and Context:\n{context}")
    ])
    
    reasoning_llm = ChatOpenAI(model=config["reasoning_llm"], temperature=0)
    final_answer_agent = final_answer_prompt | reasoning_llm | StrOutputParser()
    final_answer = final_answer_agent.invoke({"question": state['original_question'], "context": final_context})
    return {"final_answer": final_answer}

def route_next_step_node(state: RAGState) -> Dict:
    """Route to the next step in the plan without regenerating the plan."""
    console.print("--- 🔄: Routing to Next Step ---")
    current_step_index = state["current_step_index"]
    
    # Check if we've completed all steps
    if current_step_index >= len(state["plan"].steps):
        console.print("  -> All steps completed. This should not happen here.")
        return state  # Return unchanged state
    
    current_step = state["plan"].steps[current_step_index]
    console.print(f"  -> Next step: {current_step.sub_question} using {current_step.tool}")
    return state  # State is unchanged, just routing

def route_by_tool(state: RAGState) -> str:
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    return current_step.tool

def should_continue_node(state: RAGState) -> str:
    console.print("--- 🚦: Evaluating Policy ---")
    current_step_index = state["current_step_index"]
    
    if current_step_index >= len(state["plan"].steps):
        console.print("  -> Plan complete. Finishing.")
        return "finish"
    
    if current_step_index >= config["max_reasoning_iterations"]:
        console.print("  -> Max iterations reached. Finishing.")
        return "finish"

    # Check if the last retrieval step failed to find documents
    if not state["reranked_docs"]:
        console.print("  -> Retrieval failed for the last step. Continuing with next step in plan.")
        return "continue"

    history = get_past_context_str(state['past_steps'])
    plan_str = json.dumps([s.dict() for s in state['plan'].steps])
    decision = policy_agent.invoke({"question": state["original_question"], "plan": plan_str, "history": history})
    console.print(f"  -> Decision: {decision.next_action} | Justification: {decision.justification}")
    
    if decision.next_action == "FINISH":
        return "finish"
    else: # CONTINUE_PLAN
        return "continue"

