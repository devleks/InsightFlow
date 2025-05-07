# InsightFlow - API Documentation

InsightFlow primarily interacts with external APIs for its core functionalities, rather than exposing its own public API for third-party consumption. This document outlines the key external APIs used.

## 1. LLM Provider APIs

InsightFlow is designed to be flexible with LLM providers. The specific API used depends on the configuration in `config.yaml`.

### a. OpenAI API

*   **Purpose**: Used for generating answers to user queries based on provided context, and potentially for generating embeddings.
*   **Key Endpoints (Conceptual, via Langchain client)**:
    *   Chat Completions (e.g., `gpt-3.5-turbo`, `gpt-4`)
    *   Embeddings (e.g., `text-embedding-ada-002`, `text-embedding-3-small`)
*   **Authentication**: API Key (specified in `.env` as `OPENAI_API_KEY`).
*   **Request/Response Format**: JSON, managed by the Langchain OpenAI client library.
*   **Documentation**: [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

### b. Groq API

*   **Purpose**: Used for fast LLM inference for generating answers.
*   **Key Endpoints (Conceptual, via Langchain client)**:
    *   Chat Completions (e.g., `llama3-8b-8192`, `mixtral-8x7b-32768`)
*   **Authentication**: API Key (specified in `.env` as `GROQ_API_KEY`).
*   **Request/Response Format**: JSON, managed by the Langchain Groq client library.
*   **Documentation**: [Groq API Documentation](https://console.groq.com/docs/)

### c. Ollama (Local LLMs)

*   **Purpose**: Allows running open-source LLMs locally.
*   **Interaction**: Langchain's Ollama integration communicates with a locally running Ollama server.
*   **Key Endpoints**: The Langchain client interacts with the Ollama server's API (typically `http://localhost:11434` by default).
*   **Authentication**: Generally none required for local interaction unless Ollama server is configured otherwise.
*   **Request/Response Format**: JSON, managed by the Langchain Ollama client library.
*   **Documentation**: [Ollama GitHub](https://github.com/ollama/ollama), [Langchain Ollama Integration](https://python.langchain.com/docs/integrations/llms/ollama)

## 2. Embedding Provider APIs

If not using OpenAI for embeddings, other services might be used (e.g., Hugging Face for SentenceTransformers, Cohere, etc.).

### a. Hugging Face (via SentenceTransformers)

*   **Purpose**: Used for generating document and query embeddings locally using models from Hugging Face.
*   **Interaction**: The `SentenceTransformer` library downloads and runs models locally. Internet access may be required for initial model download.
*   **Authentication**: Typically not required for using pre-trained public models.
*   **Documentation**: [SentenceTransformers Library](https://www.sbert.net/)

## 3. Tavily Search API (Optional - if integrated for web search)

*   **Purpose**: If integrated, used to provide web search capabilities to the RAG system for more up-to-date or broader context.
*   **Authentication**: API Key (specified in `.env` as `TAVILY_API_KEY`).
*   **Documentation**: [Tavily API Documentation](https://tavily.com/)

## Internal Application Structure (Not a Public API)

While InsightFlow itself does not expose a REST API for external use, its internal structure is modular:

*   `app.py`: Contains the main application logic and Streamlit UI definitions.
*   Functions within `app.py` handle:
    *   Configuration loading.
    *   Document loading, splitting, and embedding.
    *   Vector store initialization and querying.
    *   Interaction with the RAG chain.
    *   Chat history management.
    *   UI rendering.

If the project were to evolve to include a separate backend service, API documentation tools like Swagger/OpenAPI, Postman, or Redoc would be used to document its endpoints.
