
# Central Configuration Dictionary
config = {
    "data_dir": "./data",
    "vector_store_dir": "./vector_store",
    "llm_provider": "openai",
    "reasoning_llm": "gpt-4o",
    "fast_llm": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "max_reasoning_iterations": 7, # Maximum loops for the reasoning agent
    "top_k_retrieval": 10,       # Number of documents for initial broad recall
    "top_n_rerank": 3,           # Number of documents to keep after precision reranking
}