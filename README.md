# Agentic RAG: Advanced Retrieval-Augmented Generation System

## Overview

This project demonstrates the evolution from basic RAG (Retrieval-Augmented Generation) to sophisticated agentic reasoning systems. It explores the limitations of traditional similarity-based retrieval and introduces autonomous agents capable of multi-step reasoning, dynamic planning, and cross-source synthesis.

## Main.py Program Structure - First Section Description

### 1. Introduction: The Limits of Shallow RAG

This section demonstrates the foundational components and limitations of a basic RAG (Retrieval-Augmented Generation) system. The program explores how traditional RAG pipelines struggle with complex, multi-step reasoning tasks that require synthesizing information from multiple sources and time periods.

#### Import Dependencies

The program begins by importing essential libraries organized into logical groups:

**Core Python Libraries:**
- `os`, `re`, `json` - File operations, pattern matching, and data handling
- `getpass`, `pprint` - Secure input handling and formatted output
- `typing` - Type annotations for better code clarity and IDE support

**Environment and Configuration:**
- `dotenv` - Environment variable management for secure API key handling
- Custom `config` module - Application configuration settings

**LangChain Framework Components:**
- `TextLoader` - Document loading from text files
- `RecursiveCharacterTextSplitter` - Intelligent text chunking that respects semantic boundaries
- `Document` - Core document abstraction for text processing
- `ChatPromptTemplate`, `StrOutputParser` - Prompt management and response parsing
- `OpenAIEmbeddings`, `ChatOpenAI` - OpenAI integration for embeddings and chat models
- `Chroma` - Vector database for storing and retrieving document embeddings

**Additional Tools:**
- `rich` - Enhanced console output formatting
- `pydantic` - Data validation and structured output parsing
- `numpy` - Numerical operations and array manipulations

#### Environment Setup and API Configuration

The program establishes the runtime environment by:

1. **Loading Environment Variables**: Securely retrieving API keys from a `.env` file:
   - `OPEN_AI_KEY` - OpenAI API for LLM and embeddings
   - `LANGSMITH_API_KEY` - LangSmith for tracing and debugging
   - `TAVILY_API_KEY` - Tavily for web search capabilities

2. **Configuring LangSmith Tracing**: Enables detailed logging of all LLM calls, retrieval operations, and agent decisions for monitoring and debugging the RAG pipeline

3. **Directory Management**: Creates necessary directories for data storage and vector embeddings, ensuring the file structure exists before attempting to save documents or vector stores

#### Document Acquisition and Preprocessing Pipeline

This section demonstrates programmatic document preparation for RAG processing:

**Knowledge Base Selection**: Uses NVIDIA's 2023 10-K filing as the primary knowledge source - a complex financial document that requires sophisticated reasoning to extract meaningful insights across multiple sections.

**File Path Configuration**: 
- Downloads the raw HTML filing from the SEC EDGAR database
- Processes it into clean text format for optimal RAG performance
- Stores both versions locally for comparison and debugging

**Document Processing**: Utilizes a custom `download_and_parse_10k` function that handles:
- HTML parsing and content extraction
- Table processing and text normalization
- Clean text output suitable for chunking and embedding

This foundation sets up the core infrastructure needed for both baseline and advanced RAG implementations, demonstrating the progression from simple similarity-based retrieval to sophisticated agentic reasoning systems.

### 2. The Baseline - Building and Breaking a "Vanilla" RAG Pipeline

This section implements a traditional RAG system to establish performance benchmarks and expose the limitations of simple similarity-based retrieval when faced with complex reasoning tasks.

#### Defining the Challenge: A Multi-Source, Temporal Reasoning Task

The program introduces a deliberately difficult query that tests the system's ability to:

1. **Extract specific risk factors** from NVIDIA's 10-K filing
2. **Search for recent external information** (2024 news about AMD)
3. **Synthesize connections** between historical risks and current market developments
4. **Provide analytical insights** that go beyond simple fact retrieval

**Challenge Query**: *"Based on NVIDIA's 2023 10-K filing, identify their key risks related to competition. Then, find recent news (post-filing, from 2024) about AMD's AI chip strategy and explain how this new strategy directly addresses or exacerbates one of NVIDIA's stated risks."*

#### Document Loading and Naive Chunking Strategy

Traditional RAG systems use simple text splitting without considering document structure:

- **TextLoader**: Loads the cleaned NVIDIA 10-K document
- **RecursiveCharacterTextSplitter**: Attempts to maintain semantic coherence
  - `chunk_size=1000`: Maximum characters per chunk (balances context window vs. precision)
  - `chunk_overlap=150`: Overlap between chunks to preserve context across boundaries
- **Limitation**: Often breaks semantic boundaries and loses important contextual relationships

#### Baseline RAG Implementation

- **OpenAI Embeddings**: Converts text into high-dimensional vectors for similarity-based retrieval
- **Basic RAG Chain**: Demonstrates simple vector similarity search limitations
- **Performance Benchmark**: Establishes baseline metrics for comparison with advanced system

### 3. The "Deep Thinking" Upgrade: Engineering an Autonomous Reasoning Engine

This section introduces the core innovation - a sophisticated agentic system that can perform multi-step reasoning, dynamic planning, and adaptive retrieval.

#### RAGState: The Central Nervous System

**RAGState** serves as the agent's central nervous system - a single, well-defined record containing:

- **User Query**: The original question requiring complex reasoning
- **Live Plan**: Current subgoals and action steps ("Plans")
- **Audit Trail**: Complete history of actions taken ("PastSteps")
- **Evidence**: All gathered information from static stores and web search
- **Fused Context**: Processed information fed to the model
- **Provisional Hypotheses**: Intermediate reasoning results
- **Self-Critiques**: Quality assessments and gap identification

**Key Benefits**:
- **Determinism**: Every transition is explicit and traceable
- **Observability**: End-to-end provenance for debugging and trust
- **Agentic Control**: The state itself declares the next action
- **Cyclical Reasoning**: Enables multi-hop thinking vs. linear pipelines

#### Dynamic Planning and Query Formulation

**Planner Agent**: 
- Decomposes complex user queries into manageable sub-tasks
- Selects appropriate tools for each step
- Creates adaptive execution plans

**Query Rewriter Agent**:
- Transforms naive sub-questions into high-quality search queries
- Optimizes queries for different retrieval strategies
- Enhances search precision through intelligent reformulation

**Metadata Enhancement**:
- Processes documents with section-based metadata
- Enables filtered vector search by document sections
- Improves retrieval precision through structural awareness

#### The Multi-Stage, Adaptive Retrieval Funnel

**Retrieval Supervisor Agent**:
- Analyzes incoming queries to select optimal retrieval strategy
- Chooses from three available options:
  - **Vector Search**: Semantic similarity-based retrieval
  - **Keyword Search**: Traditional BM25-based search
  - **Hybrid Search**: Combines semantic and keyword approaches

**Advanced Vector Store**:
- Enhanced Chroma vector database with metadata support
- Improved embedding storage and retrieval capabilities
- Section-aware document organization

**Cross-Encoder Reranker**:
- High-precision document relevance scoring
- Secondary ranking to improve retrieval quality
- Reduces noise in retrieved context

**Contextual Distiller Agent**:
- Synthesizes retrieved information into concise context
- Removes redundancy and focuses on relevant information
- Optimizes context for downstream reasoning

#### Tool Augmentation with Web Search

**Web Search Integration**:
- Tavily API integration for real-time information retrieval
- Bridges the gap between static documents and current events
- Enables temporal reasoning across different time periods

#### Self-Critique and Control Flow Policy

**Reflection Agent**:
- Synthesizes new findings into the reasoning history
- Updates RAGState with research progress
- Maintains coherent knowledge accumulation

**Policy Agent - The Strategic Decision Maker**:
- Acts as the "brain" of the autonomous RAG system
- Functions as an LLM-as-a-Judge analyzing research progress
- Makes critical decisions about next actions:
  - **CONTINUE_PLAN**: Keep researching, more information needed
  - **FINISH**: Sufficient information gathered, generate final answer

**Input Analysis**:
- Original Question: The user's initial query
- Initial Plan: Step-by-step research plan
- Research History: Summaries of completed research steps

**Robust Stopping Criteria**:
1. **Policy Decision**: Primary condition when policy agent decides to FINISH
2. **Plan Completion**: Natural conclusion when all steps are executed
3. **Max Iterations**: Safeguard against infinite loops (configurable limit)

### 4. Assembly with LangGraph - Orchestrating the Reasoning Loop

This section demonstrates how to orchestrate the autonomous reasoning system using LangGraph's workflow management capabilities.

#### Graph Construction and Compilation

**LangGraph Integration**:
- **Graph Building**: Constructs the reasoning workflow from individual agent nodes
- **Conditional Edges**: Implements self-critique policy logic for dynamic flow control
- **State Management**: Handles RAGState transitions between agents

**Workflow Compilation**:
- Compiles the graph into an executable reasoning engine
- Validates agent connections and state transitions
- Creates optimized execution pipeline

#### Visualization and Debugging

**Graph Visualization**:
- Generates PNG visualization of the reasoning workflow
- Shows agent connections and decision points
- Assists in debugging and system understanding
- Saves visualization to `data/rag_graph_visualization.png`

**Error Handling**:
- Graceful fallback when visualization dependencies are missing
- Informative error messages for troubleshooting

### 5. Running the Deep Thinking Pipeline on the Challenge Query

This section demonstrates the execution of the complete agentic RAG system on the complex multi-source query.

#### Pipeline Execution

**Graph Invocation**:
- Initializes the reasoning loop with the challenge query
- Streams execution progress through multiple reasoning cycles
- Captures intermediate states and final results

**State Streaming**:
- Real-time monitoring of agent decisions and actions
- Progressive state updates through the reasoning process
- Transparent execution tracking for observability

**Result Processing**:
- Extracts final answer from the completed reasoning state
- Formats and displays the comprehensive response
- Handles cases where no final answer is generated

#### Output Management

**Structured Results**:
- Clear separation between process execution and final results
- Formatted output for easy readability
- Error handling for incomplete executions

### 6. Evaluating Performance with RAGAs

This section implements comprehensive evaluation using the RAGAs framework to quantitatively assess both baseline and advanced RAG systems.

#### RAGAs Evaluation Framework

**Evaluation Metrics**:
- **Context Precision**: Measures the signal-to-noise ratio in retrieved information
- **Context Recall**: Evaluates coverage of all relevant facts
- **Answer Faithfulness**: Tests whether responses are grounded in retrieved evidence
- **Answer Correctness**: Measures alignment with ideal, human-crafted answers

**Structured Assessment**:
- LLM-assisted method to quantify each RAG pipeline component
- Transparent, quantitative comparison methodology
- Demonstrates advanced agent's superior performance

#### Dataset Construction

**Evaluation Dataset**:
- Contains multi-source query and retrieved contexts
- Includes generated answers from both baseline and advanced pipelines
- Features manually written ground-truth answers for comparison
- Structured format compatible with RAGAs evaluation

#### Performance Comparison

**Quantitative Results**:
- Side-by-side comparison of baseline vs. deep thinking RAG
- Detailed metrics breakdown across all evaluation dimensions
- Clear demonstration of advanced system's superior:
  - Retrieval accuracy
  - Reasoning integrity  
  - Answer reliability

**Results Presentation**:
- Pandas DataFrame format for easy analysis
- Transposed metric view for clear comparison
- Comprehensive performance assessment

## Project Structure

```
agentic-rag/
├── main.py                          # Main program demonstrating RAG evolution
├── config.py                        # Configuration settings
├── agents.py                        # Agent definitions and state management
├── basic_rag.py                     # Baseline RAG implementation
├── graphs_node.py                   # Individual agent nodes
├── graphs_structure.py              # Graph workflow definition
├── retrieval_strategies.py          # Advanced retrieval methods
├── tools.py                         # Utility functions and tools
├── eval_dataset_build.py            # Evaluation dataset construction
├── requirement.txt                  # Python dependencies
├── data/                            # Document storage
└── vector_store/                    # Vector database storage
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

2. **Environment Setup**:
   Create a `.env` file with your API keys:
   ```
   OPEN_AI_KEY=your_openai_api_key
   LANGSMITH_API_KEY=your_langsmith_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

3. **Run the Program**:
   ```bash
   python main.py
   ```

## Key Features

- **Baseline RAG**: Traditional similarity-based retrieval system
- **Agentic RAG**: Advanced reasoning with autonomous agents
- **Multi-source Integration**: Combines static documents with real-time web search
- **Dynamic Planning**: Adaptive query decomposition and execution
- **Cross-encoder Reranking**: High-precision document relevance scoring
- **Evaluation Framework**: RAGAs-based performance assessment

## License

This project is for educational purposes and demonstrates advanced RAG techniques using real-world financial documents.