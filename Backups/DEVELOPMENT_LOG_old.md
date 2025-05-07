# InsightFlow Development Log

This document summarizes the key steps, decisions, and outcomes during the development of the InsightFlow application.

| Step/Phase                 | User Request / Goal                                       | Cascade Actions & Recommendations                                                                                                                               | Outcome/Resolution                                                                                                                               |
| :------------------------- | :-------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project Setup**          | Create a general-purpose RAG app named InsightFlow.     | - Created project directory.
- Created initial files: `app.py`, `config.yaml`, `requirements.txt`, `.env.example`, `README.md`.
- Defined configurable structure in `config.yaml`. | Basic project structure established. Configuration-driven design adopted.                                                                          |
| **Environment Setup**      | Use existing API keys from another project (`nutricare`). | - Viewed `nutricare/.env` file content.
- Created `InsightFlow/.env` file with the copied content.                                                              | API keys successfully transferred and available to InsightFlow via `.env` file.                                                                  |
| **Document Ingestion**   | Implement document loading and vector store population.   | - Added functions (`load_config`, `get_api_key`, `check_api_key`).
- Implemented `get_embedding_function` (OpenAI default).
- Implemented `load_and_split_documents` (PDF default).
- Implemented `initialize_vector_store` (Chroma default).
- Implemented `ensure_vector_store_populated` to check and ingest docs. | `app.py` now loads config, initializes embeddings, loads/splits PDF, initializes Chroma DB, and ingests documents if the DB is empty.              |
| **RAG Chain Setup**        | Implement the core RAG logic.                             | - Added imports (ChatOpenAI, PromptTemplate, Runnables, etc.).
- Implemented `get_llm` function (OpenAI default).
- Implemented `create_rag_chain` using retriever, prompt from config, and LLM. | RAG chain is created using components defined in `config.yaml` and the populated vector store.                                                  |
| **Chat Interface**         | Build the user interaction interface.                     | - Added Streamlit session state for message history.
- Implemented chat message display loop.
- Added `st.chat_input`.
- Implemented RAG chain invocation on user input.
- Added optional Llama Guard input filtering (`filter_input_with_guard`). | Functional chat interface allowing users to query the document via the RAG chain, with optional safety filtering.                                |
| **Initial Run & Debug**  | Run the application and test functionality.               | - Ran `streamlit run app.py`.
- Identified `StreamlitSetPageConfigMustBeFirstCommandError`.                                                                                  | Application ran but failed immediately due to incorrect `st.set_page_config` placement.                                                          |
| **Error Resolution**     | Fix the `StreamlitSetPageConfigMustBeFirstCommandError`.    | - Explained the cause (other `st.` commands before `set_page_config`).
- Attempted automated edits (failed due to tool errors).
- Guided USER through manual code modification to move `st.set_page_config` to the top of the `if config:` block. | User successfully moved `st.set_page_config`. Application runs correctly and displays the chat interface. Initial query successful.             |
| **UI Refinement**        | Improve UI layout based on user feedback.               | - Move status notifications to sidebar.
- Display filename persistently in sidebar.
- Ensure app title/header remain at the top of the main area.         | *Pending implementation.*                                                                                                                         |
