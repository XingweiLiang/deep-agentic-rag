#
# 1. Introduction: The Limits of Shallow RAG
#
import os
import re
import json
from getpass import getpass
from pprint import pprint

from typing import List, Dict, TypedDict, Literal, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_openai import OpenAIEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rich.pretty import pprint as rprint
from pydantic import BaseModel, Field

import numpy as np

from langchain_community.vectorstores import Chroma


load_dotenv()
openai_api_key = os.getenv("OPEN_AI_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Configure LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Deep-Thinking-RAG"

from config import config

os.makedirs(config["data_dir"], exist_ok=True)
os.makedirs(config["vector_store_dir"], exist_ok=True)

 
# Preparing Our Knowledge Base from Documents

# URL for NVIDIA's 2023 10-K filing (filed Feb 2023 for fiscal year ending Jan 2023)
url_10k = "https://www.sec.gov/Archives/edgar/data/1045810/000104581023000017/nvda-20230129.htm"
doc_path_raw = os.path.join(config["data_dir"], "nvda_10k_2023_raw.html")
doc_path_clean = os.path.join(config["data_dir"], "nvda_10k_2023_clean.txt")

print("Downloading and parsing NVIDIA's 2023 10-K filing...")
from tools import download_and_parse_10k
download_and_parse_10k(url_10k, doc_path_raw, doc_path_clean)

# To expose the limits of a standard RAG pipeline, we introduce a deliberately difficult query that demands reasoning across 
# multiple sources and timeframes. The task requires both factual extraction and analytical synthesis—combining static 
# corporate disclosures with dynamic, real-world developments.

# 2: The Baseline - Building and Breaking a "Vanilla" RAG Pipeline

# Document Loading and Naive Chunking Strategy
print("Loading and chunking the document...")
loader = TextLoader(doc_path_clean, encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
doc_chunks = text_splitter.split_documents(documents)

print(f"Document loaded and split into {len(doc_chunks)} chunks.")


# basic rag chain

complex_query_adv = "Based on NVIDIA's 2023 10-K filing, identify their key risks related to competition. Then, find recent news (post-filing, from 2024) about AMD's AI chip strategy and explain how this new strategy directly addresses or exacerbates one of NVIDIA's stated risks."
embedding_function = OpenAIEmbeddings(model=config['embedding_model'])

from basic_rag import basic_rag_chain
baseline_retriever, baseline_result = basic_rag_chain(complex_query_adv, doc_chunks, embedding_function, config) 


# 3: The "Deep Thinking" Upgrade: Engineering an Autonomous Reasoning Engine 
# RAGState is the agent’s central nervous system: a single, well-defined record that carries the user query, 
# the live plan of subgoals (“Plans”), a complete audit trail of what has been done (“PastSteps”), all gathered evidence 
# from static stores and the web, the fused context actually fed to the model, provisional hypotheses, and self-critiques. 
# Each node in the reasoning graph reads this state and returns an updated one—planning expands or revises goals, 
# retrieval populates evidence, fusion deduplicates and reranks, reasoning drafts answers, critique identifies gaps or staleness,
# and decision either loops back for more evidence or finalizes the response. By centralizing plans, steps, and artifacts in 
# one immutable structure, the system gains determinism (every transition is explicit), observability (end-to-end provenance 
# for debugging and trust), and true agentic control (the state itself declares the next action), enabling cyclical, 
# multi-hop thinking rather than a brittle linear pipeline.

from agents import Plan, Step, RAGState, PastStep



# Dynamic Planning and Query Formulation

# Decomposing the user query and selecting the right tool for each step.

from graphs_node import planner_agent
print("Tool-Aware Planner Agent created successfully.")


# Using an LLM to transform naive sub-questions into high-quality search queries.

from graphs_node import query_rewriter_agent
print("Query Rewriter Agent created successfully.")


# Identifying metadata filters to enable filtered vector search
from tools import process_document_with_metadata
doc_chunks_with_metadata = process_document_with_metadata(documents, doc_path_clean, text_splitter)
print(f"Created {len(doc_chunks_with_metadata)} chunks with section metadata.")


# The Multi-Stage, Adaptive Retrieval Funnel

# The Retrieval Supervisor Agent
# The supervisor agent analyzes each incoming query and automatically selects the optimal retrieval strategy from 
# three available options: vector_search, keyword_search, and hybrid_search.

from graphs_node import retrieval_supervisor_agent
print("Retrieval Supervisor Agent created.")
# test_retrieval_supervisor_agent.py


# Implementing the Retrieval Strategies
# test_retrieval_strategies.py
# The Retrieval Strategies are the core search engines that power your RAG system's ability to find relevant information 
# from documents. They provide three distinct approaches to document retrieval, each optimized for different types of queries.

print("Creating advanced vector store with metadata...")
advanced_vector_store = Chroma.from_documents(
    documents=doc_chunks_with_metadata,
    embedding=embedding_function
)
print(f"Advanced vector store created with {advanced_vector_store._collection.count()} embeddings.")

# # Initialize retrieval strategies
from retrieval_strategies import initialize_retrieval_engine, vector_search_only, bm25_search_only, hybrid_search
initialize_retrieval_engine(advanced_vector_store, doc_chunks_with_metadata)


# High Precision: Cross-Encoder Reranker.
print("Initializing CrossEncoder reranker...")
from tools import rerank_documents_function
print("Cross-Encoder ready.")


# Contextual Distillation: Implementing logic to synthesize a concise context.
from graphs_node import distiller_agent
print("Contextual Distiller Agent created.")


# Tool Augmentation with Web Search
from tools import web_search_function
print("Web search tool (Tavily) initialized.")


# Self-Critique and Control Flow Policy
# The "Update and Reflect" Step: An agent that synthesizes new findings into the RAGState's reasoning history.
from graphs_node import reflection_agent
print("Reflection Agent created.")


# the Policy Agent

# The Policy Agent - The Strategic Decision Maker
# The Policy Agent is the "brain" of the autonomous RAG system - it acts as an intelligent controller that decides when 
# the system has gathered enough information to answer the user's question.

# The policy agent serves as an LLM-as-a-Judge that analyzes the research progress and makes critical decisions about 
# what to do next: 1) CONTINUE_PLAN → Keep researching, more information needed 
# 2) FINISH → Sufficient information gathered, generate final answer

# The policy agent receives three key inputs:
# Original Question - The user's initial query
# Initial Plan - The step-by-step research plan created by the planner agent
# Research History - Summaries of all completed research steps

from graphs_node import policy_agent
print("Policy Agent created.")


# Defining Robust Stopping Criteria
# Our system needs clear and robust conditions to stop the reasoning loop. We have three such criteria:

# Policy Decision: The primary stopping condition is when the policy_agent confidently decides to FINISH.
# Plan Completion: If the agent has executed every step in its plan, it will naturally conclude its work.
# Max Iterations: As a safeguard against infinite loops or runaway processes, we enforce a hard limit (max_reasoning_iterations from our config) on the number of research cycles.


# 4. Assembly with LangGraph - Orchestrating the Reasoning Loop


# Defining the Conditional Edges - Implementing the Self-Critique Policy Logic
from graphs_structure import build_graph
graph = build_graph()

# Compiling and Visualizing the Iterative Workflow

deep_thinking_rag_graph = graph.compile()
print("Graph compiled successfully.")

# draw graph
try:
    # Save the graph visualization as a PNG file instead of displaying
    png_image = deep_thinking_rag_graph.get_graph().draw_png()
    graph_image_path = os.path.join(config["data_dir"], "rag_graph_visualization.png")
    with open(graph_image_path, "wb") as f:
        f.write(png_image)
    print(f"Graph visualization saved to: {graph_image_path}")
except Exception as e:
    print(f"Graph visualization failed: {e}. Please ensure pygraphviz is installed.")


# 5. Running the Deep Thinking Pipeline on the Challenge Query

final_state = None
graph_input = {"original_question": complex_query_adv}

print("--- Invoking Deep Thinking RAG Graph ---")
for chunk in deep_thinking_rag_graph.stream(graph_input, stream_mode="values"):
    final_state = chunk

print("\n--- Graph Stream Finished ---")

# Display the final answer
if final_state and "final_answer" in final_state:
    print("\n" + "="*80)
    print("DEEP THINKING RAG - FINAL ANSWER")
    print("="*80)
    print(final_state["final_answer"])
    print("="*80)
else:
    print("No final answer generated")



# 6. Evaluating Performance with RAGAs
# To assess the reasoning and retrieval quality of our system, we use the RAGAs evaluation framework. It provides 
# a structured, LLM-assisted method to quantify each component of the RAG pipeline. Context Precision and 
# Context Recall evaluate the relevance and completeness of retrieved information—Precision reflects 
# the signal-to-noise ratio, while Recall measures coverage of all relevant facts. Answer Faithfulness 
# tests whether the generated response is grounded in the retrieved evidence, minimizing hallucinations, 
# and Answer Correctness measures alignment with an ideal, human-crafted answer.

# For evaluation, we build a dedicated Dataset object that contains our multi-source query, the retrieved contexts 
# and generated answers from both the baseline and advanced pipelines, and a manually written ground-truth answer. 
# RAGAs then leverages LLM-based scoring to compute all key metrics, providing a transparent, quantitative comparison 
# that demonstrates the advanced agent’s superior retrieval accuracy, reasoning integrity, and answer reliability.
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
)

metrics = [
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
]


from eval_dataset_build import build_eval_dataset
eval_dataset = build_eval_dataset(baseline_retriever, baseline_result, final_state, complex_query_adv)


print("Running RAGAs evaluation...")
result = evaluate(eval_dataset, metrics=metrics)
print("Evaluation complete.")

results_df = result.to_pandas()
results_df.index = ['baseline_rag', 'deep_thinking_rag']
print("\n--- RAGAs Evaluation Results ---")
print(results_df[['context_precision', 'context_recall', 'faithfulness', 'answer_correctness']].T)