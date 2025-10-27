from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.pretty import pprint as rprint
from dotenv import load_dotenv
import os
from config import config
from langchain_community.document_loaders import TextLoader

load_dotenv()

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

print(f"Created {len(doc_chunks_with_metadata)} chunks with section metadata.")
print("--- Sample Chunk with Metadata ---")
sample_chunk = next(c for c in doc_chunks_with_metadata if "Risk Factors" in c.metadata.get("section", ""))

# Show just the structure and preview
print("Metadata:")
# Truncate long metadata values for better readability
truncated_metadata = {}
for key, value in sample_chunk.metadata.items():
    if isinstance(value, str) and len(value) > 100:
        truncated_metadata[key] = value[:100] + "..."
    else:
        truncated_metadata[key] = value
rprint(truncated_metadata)
print(f"\nContent preview (first 100 chars):")
print(f"'{sample_chunk.page_content[:100]}...'")
print(f"\nTotal content length: {len(sample_chunk.page_content)} characters")
