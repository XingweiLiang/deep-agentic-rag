from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.markdown import Markdown
 

def basic_rag_chain(complex_query_adv, doc_chunks, embedding_function, config):

    print("Creating baseline vector store...")
  
    # Creating the Vector Store with Dense Embeddings
    baseline_vector_store = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embedding_function
    )
    baseline_retriever = baseline_vector_store.as_retriever(search_kwargs={"k": 3})

    print(f"Vector store created with {baseline_vector_store._collection.count()} embeddings.")

    # Assembling the Simple RAG Chain
    template = """You are an AI financial analyst. Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model=config["fast_llm"], temperature=0)

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    baseline_rag_chain = (
        {"context": baseline_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Baseline RAG chain assembled successfully.")

    # Demonstrating the Need for Advanced Techniques
    console = Console()
    print("Executing complex query on the baseline RAG chain...")
    baseline_result = baseline_rag_chain.invoke(complex_query_adv)

    console.print("--- BASELINE RAG FAILED OUTPUT ---")
    console.print(Markdown(baseline_result))

    return baseline_retriever, baseline_result

# This case exemplifies the core limitations of a static RAG system.

# Irrelevant Context: The retriever attempted to cover every aspect of the query simultaneously, retrieving only generic 
# mentions of “competition” and “AMD” from the 10-K—insufficient for the question’s specific focus.

# Outdated Knowledge: The 2023 filing contains no reference to 2024 developments. Without external retrieval, 
# the system cannot access or integrate recent information.

# Lack of Synthesis and Self-Awareness: The model recognizes missing data but has no mechanism to identify knowledge 
# gaps or trigger alternative tools (e.g., web search) to complete the reasoning chain.

# Together, these flaws illustrate why traditional, single-source RAG architectures fail on complex, multi-hop analytical queries.
