"""
Retrieval strategies module for the deep agentic RAG system.

This module contains different search functions that can be used by the agent:
- Vector-only search using semantic embeddings
- BM25-only search using keyword matching  
- Hybrid search combining both approaches with reciprocal rank fusion
"""

import numpy as np
from typing import List, Optional
from langchain_core.documents import Document


class RetrievalEngine:
    """
    Encapsulates all retrieval strategies and their dependencies.
    """
    
    def __init__(self, vector_store, bm25_index, doc_map, doc_ids):
        """
        Initialize the retrieval engine with necessary components.
        
        Args:
            vector_store: ChromaDB vector store for semantic search
            bm25_index: BM25Okapi index for keyword search
            doc_map: Dictionary mapping document IDs to documents
            doc_ids: List of document IDs in the same order as BM25 corpus
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.doc_map = doc_map
        self.doc_ids = doc_ids
    
    def vector_search_only(self, query: str, section_filter: str = None, k: int = 10) -> List[Document]:
        """
        Perform semantic search using vector embeddings only.
        
        Args:
            query: Search query string
            section_filter: Optional section to filter by (e.g., "Risk Factors")
            k: Number of documents to return
            
        Returns:
            List of retrieved documents
        """
        filter_dict = {"section": section_filter} if section_filter and "Unknown" not in section_filter else None
        return self.vector_store.similarity_search(query, k=k, filter=filter_dict)

    def bm25_search_only(self, query: str, k: int = 10) -> List[Document]:
        """
        Perform keyword search using BM25 algorithm only.
        
        Args:
            query: Search query string
            k: Number of documents to return
            
        Returns:
            List of retrieved documents
        """
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_k_indices = np.argsort(bm25_scores)[::-1][:k]
        return [self.doc_map[self.doc_ids[i]] for i in top_k_indices]

    def hybrid_search(self, query: str, section_filter: str = None, k: int = 10) -> List[Document]:
        """
        Perform hybrid search combining BM25 and vector search with reciprocal rank fusion.
        
        Args:
            query: Search query string
            section_filter: Optional section to filter by
            k: Number of documents to return
            
        Returns:
            List of retrieved documents ranked by RRF score
        """
        # 1. Keyword Search (BM25)
        bm25_docs = self.bm25_search_only(query, k=k)

        # 2. Semantic Search (with metadata filtering)
        semantic_docs = self.vector_search_only(query, section_filter=section_filter, k=k)

        # 3. Reciprocal Rank Fusion (RRF)
        all_docs = {doc.metadata["id"]: doc for doc in bm25_docs + semantic_docs}.values()
        ranked_lists = [
            [doc.metadata["id"] for doc in bm25_docs], 
            [doc.metadata["id"] for doc in semantic_docs]
        ]
        
        rrf_scores = {}
        for doc_list in ranked_lists:
            for i, doc_id in enumerate(doc_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1 / (i + 61)  # RRF rank constant k = 60

        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        final_docs = [self.doc_map[doc_id] for doc_id in sorted_doc_ids[:k]]
        return final_docs


# Global retrieval engine instance (will be initialized from main module)
retrieval_engine = None


def vector_search_only(query: str, section_filter: str = None, k: int = 10) -> List[Document]:
    """
    Global function wrapper for vector search.
    """
    if retrieval_engine is None:
        raise RuntimeError("Retrieval engine not initialized. Call initialize_retrieval_engine() first.")
    return retrieval_engine.vector_search_only(query, section_filter, k)


def bm25_search_only(query: str, k: int = 10) -> List[Document]:
    """
    Global function wrapper for BM25 search.
    """
    if retrieval_engine is None:
        raise RuntimeError("Retrieval engine not initialized. Call initialize_retrieval_engine() first.")
    return retrieval_engine.bm25_search_only(query, k)


def hybrid_search(query: str, section_filter: str = None, k: int = 10) -> List[Document]:
    """
    Global function wrapper for hybrid search.
    """
    if retrieval_engine is None:
        raise RuntimeError("Retrieval engine not initialized. Call initialize_retrieval_engine() first.")
    return retrieval_engine.hybrid_search(query, section_filter, k)


def initialize_retrieval_engine(vector_store, doc_chunks_with_metadata):
    """
    Initialize the global retrieval engine instance.
    
    Args:
        vector_store: ChromaDB vector store for semantic search
        bm25_index: BM25Okapi index for keyword search
        doc_map: Dictionary mapping document IDs to documents
        doc_ids: List of document IDs in the same order as BM25 corpus
    """
    from rank_bm25 import BM25Okapi

    print("Building BM25 index for keyword search...")
    tokenized_corpus = [doc.page_content.split(" ") for doc in doc_chunks_with_metadata]
    doc_ids = [doc.metadata["id"] for doc in doc_chunks_with_metadata]
    doc_map = {doc.metadata["id"]: doc for doc in doc_chunks_with_metadata}
    bm25_index = BM25Okapi(tokenized_corpus)

    global retrieval_engine
    retrieval_engine = RetrievalEngine(vector_store, bm25_index, doc_map, doc_ids)
    print("Retrieval engine initialized successfully.")