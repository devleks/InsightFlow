# InsightFlow - Chat with Your Documents

InsightFlow is an intelligent application built with Python and Streamlit that allows users to upload various document types (PDF, TXT, DOCX), ask questions, and receive insightful answers based on the content of those documents. It leverages Large Language Models (LLMs) via Langchain for question-answering and uses a vector store for efficient document retrieval.

This project was collaboratively developed with Cascade, a powerful agentic AI coding assistant from Windsurf.

## 🚀 Features

*   **Multi-Document Support**: Upload and process PDF, TXT, and DOCX files.
*   **Conversational Q&A**: Engage in a chat-like interface to query your documents.
*   **Source Attribution**: Responses include references to source documents and page numbers for verification.
*   **Configurable**: Settings for LLMs, embedding models, chunking strategy, and data sources are managed via a `config.yaml` file.
*   **Persistent Vector Store**: Uses ChromaDB to store document embeddings, allowing for quick re-initialization and persistence of processed data.
*   **User-Friendly Interface**: Built with Streamlit for an interactive web application experience.
*   **Session Management**: Chat history is maintained per user session.
*   **Flexible Data Source Configuration**: Supports loading documents from local directories.
*   **Dynamic Sidebar**: Displays available source files and their status.

## 🛠️ Tech Stack

*   **Backend**: Python
*   **Frontend**: Streamlit
*   **Core Logic**: Langchain, OpenAI (or other LLM providers like Groq, Ollama as configured)
*   **Document Loading**: `langchain_community.document_loaders` (specifically `DirectoryLoader` with `UnstructuredFileLoader`)
*   **Text Splitting**: `RecursiveCharacterTextSplitter`
*   **Vector Store**: ChromaDB
*   **Embedding Models**: Configurable (e.g., OpenAI embeddings, SentenceTransformers)
*   **Configuration**: PyYAML, python-dotenv
*   **Development Assistant**: Cascade (Agentic AI Coding Assistant by Windsurf)

## ⚙️ Installation

1.  **Clone the repository (if applicable) or ensure you have the project files.**

2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Make sure you have `pip` installed. From the project root directory:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the project root directory by copying from `.env.example` (if provided) or creating a new one. Add your API keys and other necessary configurations:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    # Add other keys like GROQ_API_KEY, TAVILY_API_KEY if used
    ```

5.  **Configure `config.yaml`:**
    Review and update `config.yaml` in the project root to set your desired LLM, embedding model, vector store path, data source path, and other application settings.

## 🚀 Usage

1.  Ensure your virtual environment is activated and dependencies are installed.
2.  Place the documents you want to query into the directory specified in `config.yaml` under `data_source.path`.
3.  Run the Streamlit application from the project root directory:
    ```bash
    streamlit run app.py
    ```
4.  The application will open in your web browser.
5.  Use the chat interface to ask questions about your uploaded documents.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details (if a specific LICENSE file is added, otherwise assume MIT or specify).

## 📞 Contact & Support

For support, questions, or contributions, please refer to the project's issue tracker or contact the development team.
*   Primary Developer: [Your Name/Organization Name]
*   AI Pairing Partner: Cascade (Windsurf)
