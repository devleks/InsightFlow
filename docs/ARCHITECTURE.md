# InsightFlow - Architecture Overview

This document provides a high-level overview of the InsightFlow application's architecture.

## 1. System Diagram

```mermaid
graph TD
    User[End User] -- Interacts via Browser --> FE[Streamlit Frontend]

    subgraph ApplicationServer[Python Application Server]
        FE -- User Input/Queries --> BE_App[app.py - Backend Logic]
        BE_App -- Loads Documents --> DS[Data Sources: Local Directory]
        DS -- Files (PDF, TXT, DOCX) --> DL[Document Loaders: DirectoryLoader, UnstructuredFileLoader]
        DL -- Raw Document Content & Metadata --> TS[Text Splitter: RecursiveCharacterTextSplitter]
        TS -- Document Chunks --> EMB[Embedding Model: OpenAI, SentenceTransformers, etc.]
        EMB -- Embeddings & Metadata --> VS[Vector Store: ChromaDB]
        BE_App -- Query + Context --> LLM[LLM Service: OpenAI, Groq, Ollama]
        VS -- Retrieves Relevant Chunks --> BE_App
        LLM -- Answer --> BE_App
        BE_App -- Displays Response & Sources --> FE
    end

    subgraph Configuration
        ConfigYAML[config.yaml] -- App Settings --> BE_App
        EnvFile[.env] -- API Keys, Secrets --> BE_App
    end

    subgraph ExternalServices
        LLM_API[LLM Provider API]
        Embedding_API[Embedding Provider API (if applicable)]
    end

    BE_App --> LLM_API
    BE_App --> Embedding_API
```

*   **Frontend (Streamlit)**: Handles user interaction, displays chat interface, and renders results.
*   **Backend Logic (`app.py`)**: Orchestrates the application flow, manages session state, processes user queries, interacts with Langchain components, and serves data to the frontend.
*   **Data Sources**: Local file system directories containing user-uploaded documents.
*   **Document Processing Pipeline (Langchain)**:
    *   `DirectoryLoader` & `UnstructuredFileLoader`: Load and parse various document formats.
    *   `RecursiveCharacterTextSplitter`: Breaks documents into manageable chunks.
    *   `Embedding Model`: Converts text chunks into vector embeddings.
    *   `ChromaDB (Vector Store)`: Stores and indexes document embeddings for efficient similarity search.
*   **LLM Service**: A Large Language Model (e.g., OpenAI GPT series) used for question answering based on retrieved context.
*   **Configuration**: `config.yaml` for application-level settings and `.env` for secrets like API keys.

## 2. Key Technologies and Their Roles

*   **Python**: Core programming language for the backend.
*   **Streamlit**: Web application framework for building the interactive user interface.
*   **Langchain**: Framework for developing applications powered by LLMs. It provides tools for:
    *   Document loading and transformation.
    *   Text embedding and vector storage.
    *   Interacting with LLMs (chains, agents).
    *   Managing chat history and memory.
*   **Unstructured**: Library used by Langchain for parsing various file types (PDFs, DOCX, TXT) and extracting text content and metadata (including page numbers when `mode="elements"` is used).
*   **ChromaDB**: Open-source vector database used to store document embeddings and perform similarity searches.
*   **OpenAI API / Other LLM APIs (Groq, Ollama)**: Provide access to powerful language models for text generation and understanding.
*   **Embedding Models (e.g., OpenAI `text-embedding-ada-002`, SentenceTransformers)**: Generate vector representations of text.
*   **PyYAML & python-dotenv**: For managing application configuration and environment variables.

## 3. Flow of Data

1.  **Initialization**:
    *   The application starts, loading configurations from `config.yaml` and `.env`.
    *   The vector store (ChromaDB) is initialized. If it's new or needs re-population:
        *   Documents are loaded from the specified `data_source.path`.
        *   Files are parsed (PDF, TXT, DOCX) by `UnstructuredFileLoader`.
        *   Text is split into chunks by `RecursiveCharacterTextSplitter`.
        *   Chunks are converted to embeddings by the chosen embedding model.
        *   Embeddings and associated metadata (including source and page number) are stored in ChromaDB after complex metadata types are filtered out by `filter_complex_metadata`.

2.  **User Interaction (Chat)**:
    *   User types a query in the Streamlit chat interface.
    *   The query is sent to the backend (`app.py`).
    *   The backend orchestrates a RAG (Retrieval Augmented Generation) process:
        *   Relevant document chunks are retrieved from ChromaDB based on similarity to the user's query.
        *   The user's query, chat history, and the retrieved chunks (context) are passed to the LLM.
        *   The LLM generates an answer based on the provided information.
    *   The backend processes the LLM's response.
    *   Source documents and page numbers for the retrieved chunks are identified and formatted.
    *   The answer and its sources are displayed to the user in the Streamlit UI.
    *   Chat history is updated.

## 4. Modularity

*   **Configuration Files**: Decouple settings from code.
*   **Langchain Components**: Utilize pre-built and customizable components for various LLM-related tasks.
*   **Streamlit Callbacks & Session State**: Manage UI state and interactions.
*   **Helper Functions**: Encapsulate specific logic (e.g., document loading, vector store initialization, response generation).
