# InsightFlow - General Purpose RAG Application

InsightFlow is a configurable Streamlit application that allows you to chat with your documents using Retrieval-Augmented Generation (RAG).

## Features

*   **Configurable:** Easily adapt to different document sources (PDFs, directories), vector stores (Chroma), and LLMs (OpenAI, Groq, etc.) by modifying `config.yaml`.
*   **Document Ingestion:** Loads and processes documents, splits them into chunks, and stores them in a vector database.
*   **RAG Pipeline:** Retrieves relevant document chunks based on your query and uses an LLM to generate answers grounded in the provided context.
*   **Streamlit UI:** Provides a simple chat interface for interacting with your documents.
*   **(Optional) Input Filtering:** Uses models like Llama Guard to check user input for safety.

## Setup

1.  **Clone the repository (or create the files):**
    ```bash
    # If cloning from Git
    # git clone <repository-url>
    # cd InsightFlow
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   Rename `.env.example` to `.env`.
    *   Open `.env` and add your API keys for the services you intend to use (as configured in `config.yaml`).

5.  **Configure the Application:**
    *   Edit `config.yaml` to specify:
        *   The path to your document(s) (`data_source.path`).
        *   Your desired vector store settings.
        *   The embedding and LLM models you want to use.
        *   Any UI customizations.
    *   **Important:** Create the document directory specified in `data_source.path` (e.g., `./documents/`) and place your source file(s) there.

## Running the Application

```bash
streamlit run app.py
```

The application will launch in your web browser. On the first run (or if the vector store doesn't exist), it will ingest the documents specified in the configuration.

## Customization

Modify `config.yaml` to change:
*   Document sources
*   Vector store types and locations
*   Embedding models
*   LLMs
*   Prompts
*   UI text

