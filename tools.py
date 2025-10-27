import os
import re
import uuid
from typing import List
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from sentence_transformers import CrossEncoder
from config import config


def download_and_parse_10k(url, doc_path_raw, doc_path_clean):
    if os.path.exists(doc_path_clean):
        print(f"Cleaned 10-K file already exists at: {doc_path_clean}")
        return

    # SEC requires User-Agent header
    headers = {
        'User-Agent': 'Your Company Name your.email@company.com',  # Replace with your info
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    print(f"Downloading 10-K filing from {url}...")
    # headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.raise_for_status() # Ensure we got a valid response
    
    with open(doc_path_raw, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"Raw document saved to {doc_path_raw}")
    
    # Use BeautifulSoup to parse and clean the HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove tables, which are often noisy for text-based RAG
    for table in soup.find_all('table'):
        table.decompose()

    # Get clean text, attempting to preserve paragraph breaks
    text = ''
    for p in soup.find_all(['p', 'div', 'span']):
        # Simple heuristic to add newlines between blocks
        text += p.get_text(strip=True) + '\n\n'
    
    # A more robust regex to clean up excessive newlines and whitespace
    clean_text = re.sub(r'\n{3,}', '\n\n', text).strip()
    clean_text = re.sub(r'\s{2,}', ' ', clean_text).strip()
    
    with open(doc_path_clean, 'w', encoding='utf-8') as f:
        f.write(clean_text)
    print(f"Cleaned text content extracted and saved to {doc_path_clean}")

def process_document_with_metadata(documents: Document, doc_path_clean: str, text_splitter) -> Document:
    """
    Process a Document to ensure it has 'section' metadata.
    If missing, assign 'Unknown' as the section.
    
    Args:
        doc: Input Document object
    """
    print("Processing document and adding metadata...")
    # Regex to match the 'Item X' and 'Item X.Y' patterns for section titles
    section_pattern = r"(ITEM\s+\d[A-Z]?\.\s*.*?)(?=\nITEM\s+\d[A-Z]?\.|$)"
    raw_text = documents[0].page_content

    # Find all matches for section titles
    section_titles = re.findall(section_pattern, raw_text, re.IGNORECASE | re.DOTALL)
    section_titles = [title.strip().replace('\n', ' ') for title in section_titles]

    # Split the document content by these titles
    sections_content = re.split(section_pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
    sections_content = [content.strip() for content in sections_content if content.strip() and not content.strip().lower().startswith('item ')]

    print(f"Identified {len(section_titles)} document sections.")
    assert len(section_titles) == len(sections_content), "Mismatch between titles and content sections"

    doc_chunks_with_metadata = []
    for i, content in enumerate(sections_content):
        section_title = section_titles[i]
        # Chunk the content of this specific section
        section_chunks = text_splitter.split_text(content)
        for chunk in section_chunks:
            chunk_id = str(uuid.uuid4())
            doc_chunks_with_metadata.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "section": section_title,
                        "source_doc": doc_path_clean,
                        "id": chunk_id
                    }
                )
            )
    return doc_chunks_with_metadata

def rerank_documents_function(query: str, documents: List[Document]) -> List[Document]:
    if not documents: return []

    reranker = CrossEncoder(config["reranker_model"])

    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)
    
    # Combine documents with their scores and sort
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N documents
    reranked_docs = [doc for doc, score in doc_scores[:config["top_n_rerank"]]]
    return reranked_docs

def web_search_function(query: str) -> List[Document]:
    
    web_search_tool = TavilySearchResults(k=3)

    results = web_search_tool.invoke({"query": query})
    return [Document(page_content=res["content"], metadata={"source": res["url"]}) for res in results]
