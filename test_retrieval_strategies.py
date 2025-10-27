import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.pretty import pprint as rprint
from dotenv import load_dotenv
load_dotenv()
import os
from config import config
from langchain_community.document_loaders import TextLoader


# Create directories if they don't exist
os.makedirs(config["data_dir"], exist_ok=True)
os.makedirs(config["vector_store_dir"], exist_ok=True)


url_10k = "https://www.sec.gov/Archives/edgar/data/1045810/000104581023000017/nvda-20230129.htm"
doc_path_raw = os.path.join(config["data_dir"], "nvda_10k_2023_raw.html")
doc_path_clean = os.path.join(config["data_dir"], "nvda_10k_2023_clean.txt")

print("Downloading and parsing NVIDIA's 2023 10-K filing...")
from tools import download_and_parse_10k
download_and_parse_10k(url_10k, doc_path_raw, doc_path_clean)

print("Loading and chunking the document...")
loader = TextLoader(doc_path_clean, encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
doc_chunks = text_splitter.split_documents(documents)

print(f"Document loaded and split into {len(doc_chunks)} chunks.")

from tools import process_document_with_metadata
doc_chunks_with_metadata = process_document_with_metadata(documents, doc_path_clean, text_splitter)

embedding_function = OpenAIEmbeddings(model=config['embedding_model'])

# retreieval strategies test setup
print("Creating advanced vector store with metadata...")
advanced_vector_store = Chroma.from_documents(
    documents=doc_chunks_with_metadata,
    embedding=embedding_function
)
print(f"Advanced vector store created with {advanced_vector_store._collection.count()} embeddings.")

print("Building BM25 index for keyword search...")
tokenized_corpus = [doc.page_content.split(" ") for doc in doc_chunks_with_metadata]
doc_ids = [doc.metadata["id"] for doc in doc_chunks_with_metadata]
doc_map = {doc.metadata["id"]: doc for doc in doc_chunks_with_metadata}
bm25 = BM25Okapi(tokenized_corpus)

# Initialize retrieval strategies
from retrieval_strategies import initialize_retrieval_engine, vector_search_only, bm25_search_only, hybrid_search
initialize_retrieval_engine(advanced_vector_store, bm25, doc_map, doc_ids)

print("All retrieval strategy functions ready.")

# Test Keyword Search
print("--- Testing Keyword Search ---")
test_query = "Item 1A. Risk Factors"
test_results = bm25_search_only(test_query)
print(f"Query: {test_query}")
print(f"Found {len(test_results)} documents. Top result section: {test_results[0].metadata['section'][:100]}")