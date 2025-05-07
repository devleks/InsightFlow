# InsightFlow - General Purpose RAG Application

import streamlit as st
import os
import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.prompts import PromptTemplate # Keep for potential basic use
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Added
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings # Added
# from sentence_transformers import SentenceTransformer # No longer directly needed?
from groq import Groq # For Llama Guard
from langchain_community.chat_message_histories import ChatMessageHistory # Added
from langchain_core.runnables.history import RunnableWithMessageHistory # Added

# --- Setup & Configuration Loading ---
load_dotenv()

# --- Constants ---
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx']

CONFIG_PATH = 'config.yaml'

# --- Functions ---
def load_config(config_path=CONFIG_PATH):
    """Loads configuration from the YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                st.error(f"Configuration file '{config_path}' is empty.")
                return None
            # Basic validation (can be expanded)
            if 'data_source' not in config or 'vector_store' not in config or 'embedding_model' not in config or 'llm' not in config:
                 st.warning(f"Configuration file '{config_path}' seems incomplete. Ensure data_source, vector_store, embedding_model, and llm sections are present.")
            return config
    except FileNotFoundError:
        st.error(f"Configuration file not found at '{config_path}'. Please create it.")
        return None
    except yaml.YAMLError as e:
        st.error(f"Error parsing configuration file '{config_path}': {e}")
        return None
    except Exception as e:
         st.error(f"An unexpected error occurred loading config: {e}")
         return None

# --- Helper Functions ---
def get_api_key(config, config_section, default_env_var):
    """Gets the API key env variable name from config or uses default."""
    # Check if config is None before accessing it
    if config is None:
        return default_env_var
    return config.get(config_section, {}).get('api_key_env', default_env_var)

def check_api_key(env_var_name):
    """Checks if the required API key is present in environment variables."""
    api_key = os.getenv(env_var_name)
    # Display errors in the sidebar during initialization
    if not api_key:
        st.sidebar.error(f"Env variable '{env_var_name}' not set.")
        return False
    return True

# --- Provider Initialization Helpers ---

def _initialize_openai_embedding(model_name, api_key_env):
    """Initializes OpenAI Embeddings."""
    if not check_api_key(api_key_env):
        st.sidebar.error(f"Env variable '{api_key_env}' for OpenAI Embeddings not set.")
        return None
    try:
        return OpenAIEmbeddings(model=model_name)
    except Exception as e:
        st.sidebar.error(f"Failed to init OpenAI embeddings ({model_name}): {e}")
        return None

def _initialize_huggingface_embedding(model_name, api_key_env=None):
    """Initializes HuggingFace Embeddings."""
    # api_key_env is typically not required for public sentence-transformers
    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        st.sidebar.error(f"Failed to init HuggingFace embeddings ({model_name}): {e}")
        return None

def _initialize_openai_llm(model_name, temperature, api_key_env):
    """Initializes ChatOpenAI LLM."""
    if not check_api_key(api_key_env):
        st.sidebar.error(f"Env variable '{api_key_env}' for ChatOpenAI not set.")
        return None
    try:
        return ChatOpenAI(model_name=model_name, temperature=temperature)
    except Exception as e:
        st.sidebar.error(f"Failed to init OpenAI LLM ({model_name}): {e}")
        return None

def _initialize_groq_llm(model_name, temperature, api_key_env):
    """Initializes ChatGroq LLM."""
    if not check_api_key(api_key_env):
        st.sidebar.error(f"Env variable '{api_key_env}' for ChatGroq not set.")
        return None
    try:
        return ChatGroq(model_name=model_name, temperature=temperature)
    except Exception as e:
        st.sidebar.error(f"Failed to init Groq LLM ({model_name}): {e}")
        return None

# --- Provider Maps ---

EMBEDDING_PROVIDERS = {
    "openai": _initialize_openai_embedding,
    "huggingface": _initialize_huggingface_embedding,
    # Add new embedding providers here
    # "azure": _initialize_azure_embedding,
}

LLM_PROVIDERS = {
    "openai": _initialize_openai_llm,
    "groq": _initialize_groq_llm,
    # Add new LLM providers here
    # "anthropic": _initialize_anthropic_llm,
}

# --- Core Logic Functions (Refactored) ---

def get_embedding_function(config):
    """Gets the embedding function based on config, with defaults."""
    embedding_config = config.get('embedding_model', {})
    provider = embedding_config.get('provider', 'openai').lower() # Default to openai

    # Determine model name based on provider and potential defaults
    if provider == 'openai':
        default_model = 'text-embedding-3-small'
        default_api_key_env = 'OPENAI_API_KEY'
    elif provider == 'huggingface':
        default_model = 'sentence-transformers/all-MiniLM-L6-v2' # A common default
        default_api_key_env = None # Usually not needed
    else:
        # Attempt to load without defaults if provider is unknown but configured
        default_model = None
        default_api_key_env = None

    model_name = embedding_config.get('model_name', default_model)
    api_key_env = embedding_config.get('api_key_env', default_api_key_env)

    if not model_name:
        st.sidebar.error(f"Embedding model_name missing for provider '{provider}' and no default set.")
        return None, provider, None

    initializer = EMBEDDING_PROVIDERS.get(provider)
    if initializer:
        embedding_func = initializer(model_name=model_name, api_key_env=api_key_env)
        if embedding_func:
            return embedding_func, provider, model_name
        else:
            # Initialization failed, error shown by helper
            return None, provider, model_name
    else:
        st.sidebar.error(f"Unsupported embedding provider in config: {provider}")
        return None, provider, model_name

def get_llm(config):
    """Gets the LLM based on config, with defaults. Returns (LLM object, provider, model_name) or (None, provider, model_name)."""
    llm_config = config.get('llm', {})
    provider = llm_config.get('provider', 'groq').lower() # Default to groq
    temperature = llm_config.get('temperature', 0.7)

    # Determine model name and API key env based on provider and potential defaults
    if provider == 'openai':
        default_model = 'gpt-4o-mini'
        default_api_key_env = 'OPENAI_API_KEY'
    elif provider == 'groq':
        default_model = 'llama3-8b-8192'
        default_api_key_env = 'GROQ_API_KEY'
    else:
        # Attempt to load without defaults if provider is unknown but configured
        default_model = None
        default_api_key_env = None

    model_name = llm_config.get('model_name', default_model)
    api_key_env = llm_config.get('api_key_env', default_api_key_env)

    if not model_name:
        st.sidebar.error(f"LLM model_name missing for provider '{provider}' and no default set.")
        return None, provider, None

    initializer = LLM_PROVIDERS.get(provider)
    if initializer:
        llm = initializer(model_name=model_name, temperature=temperature, api_key_env=api_key_env)
        if llm:
            return llm, provider, model_name
        else:
            # Initialization failed, error shown by helper
            return None, provider, model_name
    else:
        st.sidebar.error(f"Unsupported LLM provider in config: {provider}")
        return None, provider, model_name

def load_and_split_documents(config):
    """Loads documents based on config and splits them."""
    st.sidebar.error("--- ENTERING load_and_split_documents ---") # Prominent entry log

    source_type = config.get('type', 'pdf').lower() # Corrected: directly get 'type'
    source_path = config.get('path', None)         # Corrected: directly get 'path'
    st.sidebar.info(f"[L&S_DEBUG] Initial source_type: {source_type}, source_path: {source_path}")

    # Define supported extensions for directory loading
    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]

    if not source_path:
        st.sidebar.error("[L&S_DEBUG] Exiting: source_path is None or empty.")
        return None
    
    st.sidebar.info(f"[L&S_DEBUG] Checking existence of source_path: {source_path}")
    if not os.path.exists(source_path):
        if source_path != 'Not configured': # Avoid error if path is just placeholder
             st.sidebar.error(f"[L&S_DEBUG] Exiting: Document path '{source_path}' not found.")
        return None
    st.sidebar.info(f"[L&S_DEBUG] source_path '{source_path}' exists.")

    documents = []
    try:
        if source_type == 'pdf':
            st.sidebar.info(f"[L&S_DEBUG] Mode: PDF. Path: {source_path}")
            if not source_path.lower().endswith(".pdf"):
                st.sidebar.error(f"[L&S_DEBUG] Exiting: Specified path '{source_path}' is not a PDF file for PDF loading.")
                return None
            st.sidebar.info(f"Loading PDF: {source_path}")
            loader = PyPDFLoader(source_path)
            documents = loader.load()
        elif source_type == 'directory':
            st.sidebar.info(f"[L&S_DEBUG] Mode: Directory. Path: {source_path}")
            st.sidebar.info(f"[L&S_DEBUG] Checking if '{source_path}' is a directory.")
            if not os.path.isdir(source_path):
                st.sidebar.error(f"[L&S_DEBUG] Exiting: Specified path '{source_path}' is not a directory.")
                return None
            st.sidebar.info(f"[L&S_DEBUG] '{source_path}' IS a directory. Proceeding with scan.")
            
            st.sidebar.info(f"Scanning directory: {source_path}")
            all_files_in_dir = []
            for root, _, files in os.walk(source_path):
                for file in files:
                    all_files_in_dir.append(os.path.join(root, file))
            
            supported_files_to_load = []
            if not all_files_in_dir:
                st.sidebar.warning(f"No files found in directory: {source_path}")

            for file_path in all_files_in_dir:
                _, ext = os.path.splitext(file_path)
                if ext.lower() in SUPPORTED_EXTENSIONS:
                    supported_files_to_load.append(file_path)
                else:
                    st.sidebar.warning(f"Skipping unsupported file: {os.path.basename(file_path)} (type: {ext})")

            if not supported_files_to_load:
                st.sidebar.warning(f"No supported files ({', '.join(SUPPORTED_EXTENSIONS)}) found in directory: {source_path}")
                return None

            # Using DirectoryLoader with a generic glob for supported types.
            # Specific loaders might be needed if DirectoryLoader's auto-detection fails.
            st.sidebar.info(f"Loading {len(supported_files_to_load)} supported file(s) from directory: {source_path}")
            
            # Create a glob pattern that matches any of the supported extensions.
            # Example: "**/*.[pP][dD][fF]" or "**/*.[tT][xX][tT]"
            # For DirectoryLoader, it's often easier to load one type or provide a list of files.
            # Here, we'll iterate through patterns for clarity and specific handling if needed.
            
            temp_documents = []
            # Initialize DirectoryLoader once without a specific glob initially
            # It will use its internal logic to find files if no glob is set, or use the glob if set.
            loader = DirectoryLoader(
                source_path,
                # glob will be set in the loop
                recursive=True,
                show_progress=True,
                use_multithreading=True,
                silent_errors=True # Important to skip files that error out during loading
            )

            extension_patterns = {
                ".pdf": "**/*.[pP][dD][fF]",
                ".txt": "**/*.[tT][xX][tT]",
                ".docx": "**/*.[dD][oO][cC][xX]",
            }

            loader.silent_errors = True  # Set for all types in the loop
            for ext_type in SUPPORTED_EXTENSIONS:
                pattern = extension_patterns[ext_type]
                loader.glob = pattern # Set the glob for the current file type
                st.sidebar.info(f"Attempting to load files matching: {pattern}")
                try:
                    loaded_docs_for_type = loader.load()
                    if loaded_docs_for_type:
                        st.sidebar.info(f"  -> Loaded {len(loaded_docs_for_type)} '{ext_type}' document(s).")
                        temp_documents.extend(loaded_docs_for_type)
                    else:
                        st.sidebar.info(f"  -> No '{ext_type}' documents found or loaded with pattern {pattern}.")
                except Exception as e:
                    st.sidebar.warning(f"Could not load '{ext_type}' files. Error: {e}. Ensure parsers are installed.")
            
            documents = temp_documents

            if not documents:
                 st.sidebar.warning(f"No documents successfully loaded from directory: {source_path}")
                 return None
            st.sidebar.info(f"Successfully loaded a total of {len(documents)} documents from directory.")

        else:
            st.sidebar.error(f"Unsupported data source type: {source_type}")
            return None

        if not documents:
             st.sidebar.warning(f"No documents were loaded from {source_path}.")
             return None

        # Save original loaded documents to session state for sidebar display (optional)
        # This helps in debugging what was initially loaded before splitting.
        st.session_state["loaded_documents_metadata"] = [{'source': doc.metadata.get('source', 'Unknown'), 'page_count': doc.metadata.get('page_count', 'N/A') if hasattr(doc, 'metadata') else 'N/A'} for doc in documents]

        # Split documents
        text_splitter_config = config.get('text_splitter', {})
        chunk_size = text_splitter_config.get('chunk_size', 1000)
        chunk_overlap = text_splitter_config.get('chunk_overlap', 200)

        st.sidebar.info(f"Splitting {len(documents)} documents (Chunk Size: {chunk_size}, Overlap: {chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            st.sidebar.error("Failed to split documents or no content to split.")
            return None
            
        st.sidebar.info(f"Created {len(splits)} document splits.")
        return splits # Return the splits for vector store creation

    except Exception as e:
        st.sidebar.error(f"Error during document loading/splitting: {e}")
        import traceback
        st.sidebar.error(traceback.format_exc()) # Detailed traceback in sidebar
        return None

def get_source_files_for_display(data_source_config):
    """Scans the source directory for supported files to display in the UI."""
    files_info = []
    source_type = data_source_config.get('type', 'directory').lower()
    source_path = data_source_config.get('path', None)

    if source_type == "directory":
        if not source_path or not os.path.exists(source_path) or not os.path.isdir(source_path):
            # Using st.sidebar.warning here might be an issue if called before sidebar is fully ready
            # Consider logging or returning a specific status if this function is called very early
            # For now, let's assume it's called at a point where st.sidebar is safe.
            st.sidebar.warning(f"Source path '{source_path}' is invalid or not a directory (for display).")
            return [] # Return empty list on invalid path

        try:
            for item in os.listdir(source_path):
                item_path = os.path.join(source_path, item)
                if os.path.isfile(item_path):
                    _, ext = os.path.splitext(item)
                    if ext.lower() in SUPPORTED_EXTENSIONS:
                        files_info.append({
                            "name": item,
                            "type": ext.lower()
                        })
        except Exception as e:
            st.sidebar.error(f"Error listing directory '{source_path}' for display: {e}")
            return [] # Return empty on error
    else:
        st.sidebar.warning(f"Source type '{source_type}' not supported for displaying files list.")
        return []
    
    return files_info

# --- Vector Store Initialization ---
def initialize_vector_store(config, embedding_function):
    """Initializes the vector store based on config."""
    # config here is the full application config
    vector_store_config = config.get('vector_store', {})
    data_source_config = config.get('data_source', {})

    store_type = vector_store_config.get('type', 'chroma').lower()
    persist_dir = vector_store_config.get('persist_directory', './vector_db')
    collection_name = vector_store_config.get('collection_name', 'insightflow_collection')
    force_recreate = False # Or get from config if you add this option

    st.sidebar.info(f"DEBUG: Starting vector store initialization")
    st.sidebar.info(f"DEBUG: Persist directory: {persist_dir}")
    st.sidebar.info(f"DEBUG: Collection name: {collection_name}")
    st.sidebar.info(f"DEBUG: Embedding function client: {embedding_function.client}")

    if store_type == 'chroma':
        if persist_dir and os.path.exists(persist_dir) and not force_recreate:
            st.sidebar.info(f"DEBUG: Checking if persist_dir exists and not force_recreate: {os.path.exists(persist_dir)} / {not force_recreate}")
            try:
                st.sidebar.info(f"Attempting to load existing Chroma DB from: {persist_dir} with collection: {collection_name}")
                vector_store = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embedding_function,
                    collection_name=collection_name
                )
                st.sidebar.success(f"Successfully loaded vector store from {persist_dir}")
                return vector_store
            except Exception as e:
                st.sidebar.warning(f"Failed to load existing vector store (will attempt to create new): {e}")
                # Fall through to create a new one if loading fails

        st.sidebar.info("DEBUG: Creating new vector store...")
        st.sidebar.info(f"DEBUG: data_source_config: {data_source_config}")
        
        # This is where documents are loaded and split using the data_source_config
        documents = load_and_split_documents(data_source_config) 
        st.sidebar.info(f"DEBUG: load_and_split_documents returned type {type(documents)} with length {len(documents) if documents is not None else 'None'}")

        if documents is None or len(documents) == 0:
            # Log the error specifically if no documents were loaded
            st.sidebar.error("DEBUG: No documents loaded for vector store creation!") 
            raise RuntimeError("DEBUG: No documents loaded for vector store creation!")
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=persist_dir,
            collection_name=collection_name
        )
        st.sidebar.success("New vector store created and documents ingested.")
        return vector_store
    else:
        st.sidebar.error(f"Unsupported vector store type: {store_type}")
        return None

def ensure_vector_store_populated(vector_store, config):
    """Checks if the vector store needs documents and ingests them if necessary."""
    # config here is the full application config
    vector_store_config = config.get('vector_store', {})
    data_source_config = config.get('data_source', {})

    st.sidebar.info("Ensuring vector store is populated...")
    populated = False
    needs_population = False
    try:
        count = vector_store._collection.count()
        st.sidebar.info(f"Current vector store count: {count}")
        if count == 0:
            needs_population = True
            st.sidebar.warning("Vector store is empty. Attempting to populate.")
        else:
            populated = True # Already populated
    except Exception as e:
        st.sidebar.error(f"Error checking vector store count (assuming it needs population): {e}")
        needs_population = True # If we can't check, assume it needs to be populated

    if needs_population:
        with st.spinner("Loading and splitting documents for population..."):
            st.sidebar.info(f"Loading docs from {data_source_config.get('path', 'N/A')} for population...")
            doc_splits = load_and_split_documents(data_source_config)

        if doc_splits:
            st.sidebar.info(f"Found {len(doc_splits)} document splits to add.")
            with st.spinner(f"Ingesting {len(doc_splits)} document chunks into vector store..."):
                st.sidebar.info(f"Ingesting {len(doc_splits)} chunks...")
                vector_store.add_documents(doc_splits)
                vector_store.persist() # Persist after adding new documents
                st.sidebar.success(f"{len(doc_splits)} chunks ingested and persisted.")
                populated = True
        else:
            st.sidebar.warning("No document splits found to populate the vector store.")
            populated = False # Still not populated
    else:
        st.sidebar.info("Vector store already contains documents or population check failed safely.")

    return populated, vector_store # Return original or potentially modified store

def filter_input_with_guard(user_input, config):
    """Filters user input using a guard model specified in config (e.g., Llama Guard)."""
    filter_config = config.get('input_filter', {})
    if not filter_config.get('enabled', False):
        return user_input # Filtering disabled

    provider = filter_config.get('provider', 'groq').lower()
    model_name = filter_config.get('model_name', 'llama-guard-3-8b')
    env_var = get_api_key(config, 'input_filter', 'GROQ_API_KEY') # Default to Groq key

    if provider == 'groq':
        if not check_api_key(env_var):
            # Keep runtime errors in main area
            st.error("Cannot filter input: Groq API key not found for filter model.")
            return None # Indicate failure
        try:
            client = Groq(api_key=os.getenv(env_var))
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": user_input}],
                model=model_name,
                temperature=0.0, max_tokens=10, stream=False # Short response needed (safe/unsafe)
            )
            response = chat_completion.choices[0].message.content.strip().lower()
            # print(f"[Guard Filter] Raw Response: {response}") # Optional debug
            if 'unsafe' in response: # Llama Guard typically responds 'safe' or 'unsafe...'
                # Keep runtime warnings in main area
                st.warning("Input flagged as potentially unsafe. Please rephrase.")
                return None # Indicate unsafe
            return user_input # Return original input if safe
        except Exception as e:
            # Keep runtime errors in main area
            st.error(f"Error during input filtering with {provider}: {e}")
            return None # Indicate failure
    else:
        # Keep runtime errors in main area
        st.error(f"Unsupported input filter provider: {provider}")
        return None # Indicate failure

def create_rag_chain(retriever, llm):
    """Creates the core RAG chain that returns answer and source documents."""
    template = [
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\nQuestion: {question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(template)

    def format_docs(docs):
        """Helper function to format retrieved documents for the prompt."""
        return "\n---\n".join([d.page_content for d in docs])

    # Chain segment to retrieve documents (both formatted and original)
    retrieval_chain = (
        # Extract 'question' from the input dictionary passed by RunnableWithMessageHistory
        (lambda input_dict: input_dict["question"])
        | retriever
    )

    # Parallel chain to prepare context string and pass original docs
    context_and_docs = RunnableParallel(
        {
            "context": retrieval_chain | format_docs,
            "source_documents": retrieval_chain # Pass the original Document objects
        }
    )

    # Core RAG chain logic
    rag_chain = (
        # Assign context and source_documents to the input dict (which has question and chat_history)
        RunnablePassthrough.assign(
            retrieved=context_and_docs # Adds 'context' and 'source_documents' keys
        )
        # Define the prompt processing part
        | RunnablePassthrough.assign(
            answer = (
                # Select necessary keys for the prompt
                (lambda x: {"context": x["retrieved"]["context"], "question": x["question"], "chat_history": x["chat_history"]})
                | prompt
                | llm
                | StrOutputParser()
            )
        )
        # Select the final output dictionary keys
        | (lambda x: {"answer": x["answer"], "source_documents": x["retrieved"]["source_documents"]})
    )
    return rag_chain

# --- Streamlit Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "embedding_info" not in st.session_state:
    st.session_state.embedding_info = None
if "llm_info" not in st.session_state:
    st.session_state.llm_info = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "ingestion_checked" not in st.session_state:
    st.session_state.ingestion_checked = False
if "llm" not in st.session_state:
     st.session_state.llm = None
if "rag_chain" not in st.session_state:
     st.session_state.rag_chain = None
if "chat_history" not in st.session_state: # Added for conversation memory
    st.session_state.chat_history = ChatMessageHistory()
if "displayable_source_files" not in st.session_state:
    st.session_state.displayable_source_files = []

# --- Main Application Logic ---
config = load_config() # Load config once

if config:
    data_source_cfg = config.get('data_source', {})
    # Populate displayable files early, after config is loaded
    if not st.session_state.displayable_source_files: # Avoid re-populating if already done, e.g. by a button later
        st.session_state.displayable_source_files = get_source_files_for_display(data_source_cfg)

    # --- UI Configuration (MUST be the first Streamlit command) ---
    st.set_page_config(
        page_title=config.get('ui', {}).get('page_title', 'InsightFlow'),
        page_icon=config.get('ui', {}).get('page_icon', ':page_with_curl:') # Use page_icon from config or default
    )

    # Display Filename in Sidebar
    doc_path_display = config.get('data_source', {}).get('path', 'Not configured')
    if doc_path_display != 'Not configured':
        try:
            doc_filename = os.path.basename(doc_path_display)
            st.sidebar.caption(f"Current Document: **{doc_filename}**")
        except Exception:
            st.sidebar.caption(f"Current Document Path: {doc_path_display}") # Fallback
    else:
        st.sidebar.warning("Document source path not configured.")

    # Display Title and Header in the main area
    st.title(config.get('ui', {}).get('title', 'DocuChat InsightFlow'))
    st.header(config.get('ui', {}).get('header', 'Chat with your documents to gain instant insights'))

    # --- Initialize Components (showing status in sidebar) ---
    # Use session state to store components once initialized
    if st.session_state.embedding_info is None:
        with st.spinner("Initializing embedding model..."):
            embedding_func, emb_provider, emb_model = get_embedding_function(config)
            if embedding_func:
                st.session_state.embedding_info = {
                    "function": embedding_func,
                    "provider": emb_provider,
                    "model": emb_model
                }
                st.sidebar.success("Embedding model initialized.")
            else:
                st.sidebar.error("Embedding model init failed.")
                st.stop()

    # Display Embedding Info
    if st.session_state.embedding_info:
        st.sidebar.caption(f"Embedding: {st.session_state.embedding_info['provider']} / {st.session_state.embedding_info['model']}")
    else:
        st.sidebar.caption("Embedding: Not Initialized")

    # Initialize LLM
    if st.session_state.llm_info is None:
        with st.spinner("Initializing language model..."):
            llm, llm_provider, llm_model = get_llm(config)
            if llm:
                st.session_state.llm_info = {
                    "llm": llm,
                    "provider": llm_provider,
                    "model": llm_model
                }
                st.success("Language model initialized.")
            else:
                st.error("Language model init failed.")
                st.stop()

    # Display LLM Info
    if st.session_state.llm_info:
        st.sidebar.caption(f"LLM: {st.session_state.llm_info['provider']} / {st.session_state.llm_info['model']}")
    else:
        st.sidebar.caption("LLM: Not Initialized")

    # Initialize Vector Store (only if embedding is ready)
    if st.session_state.embedding_info and st.session_state.vector_store is None:
        with st.spinner("Initializing vector store..."):
            # Pass the full config to initialize_vector_store
            vector_store = initialize_vector_store(config, st.session_state.embedding_info['function'])
            if vector_store:
                st.session_state.vector_store = vector_store
                st.sidebar.success("Vector store initialized.")
            else:
                st.sidebar.error("Failed to initialize vector store.")

    # Ensure Vector Store Populated (only if store is ready)
    if st.session_state.vector_store and not st.session_state.ingestion_checked:
        # Pass the full config to ensure_vector_store_populated
        populated, updated_store = ensure_vector_store_populated(st.session_state.vector_store, config)
        if populated:
            st.session_state.vector_store = updated_store # Update store in state if recreated
            st.session_state.ingestion_checked = True
        else:
            st.sidebar.error("Failed to populate vector store.")
            st.stop()

    # Initialize Retriever and RAG Chain (only if all components ready)
    if st.session_state.vector_store and st.session_state.llm_info and st.session_state.rag_chain is None:
        with st.spinner("Initializing RAG chain..."):
            retriever = st.session_state.vector_store.as_retriever()
            rag_chain_core = create_rag_chain(retriever, st.session_state.llm_info['llm'])

            # Define function to get history from session state
            def get_session_history(session_id: str) -> ChatMessageHistory:
                return st.session_state.chat_history

            # Wrap the core chain with history management
            st.session_state.rag_chain = RunnableWithMessageHistory(
                rag_chain_core,
                get_session_history,
                input_messages_key="question",  # The key for the input question
                history_messages_key="chat_history", # Key for MessagesPlaceholder
                output_messages_key="answer" # Key for the final answer output (optional)
            )
            st.sidebar.success("RAG chain ready.")

    # --- Sidebar: Debug Info for Loaded Files ---
    st.sidebar.markdown('---')
    st.sidebar.subheader("Available Source Files")
    
    display_files = st.session_state.get("displayable_source_files", [])
    
    if display_files:
        st.sidebar.markdown(f"**Detected Files:** {len(display_files)}")
        # Sort by name for consistent display
        for file_info in sorted(display_files, key=lambda x: x['name']):
            st.sidebar.markdown(f"- `{file_info['name']}`") # Displaying only name for brevity
    else:
        st.sidebar.info("No supported source files found in the configured directory.")

    # --- Chat Interface ---
    st.header("Chat Window")

    # Container for the chat history with a border
    chat_container = st.container()
    with chat_container:
        st.markdown("""
        <div style="
            border: 1px solid #4a4a4a; /* Light gray border */
            border-radius: 5px;      /* Rounded corners */
            padding: 10px;           /* Padding inside the box */
            margin-bottom: 20px;   /* Space below the box */
            height: 500px;           /* Fixed height for the chat area */
            overflow-y: auto;        /* Enable vertical scrolling */
        ">
        """, unsafe_allow_html=True)

        # Display chat messages from history inside the styled div
        for message in st.session_state.messages:
            avatar_icon = "👤" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar_icon):
                st.markdown(message["content"])
                # Show sources directly below the assistant response
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    st.markdown("**Sources:**")
                    for i, source_meta in enumerate(message["sources"]):
                        source_name = os.path.basename(source_meta.get('source', 'Unknown'))
                        page_num = source_meta.get('page', 'N/A')
                        st.markdown(f"- `{source_name}` (Page: {page_num})")

        # Close the styled div
        st.markdown("</div>", unsafe_allow_html=True)

    # Accept user input (remains outside the container for bottom anchoring)
    if prompt := st.chat_input("Ask a question about the documents:"):
        # Check if RAG chain is ready
        if st.session_state.rag_chain:
            # Always display the user's raw input first
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            # Always add the user's raw input to the display history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Optional: Filter input
            safe_prompt = filter_input_with_guard(prompt, config)

            if safe_prompt:
                # Proceed with RAG chain invocation if safe
                # Display assistant response
                with st.chat_message("assistant", avatar="🤖"): 
                    message_placeholder = st.empty() # Use placeholder for streaming/spinner
                    try:
                        with st.spinner("Thinking...",show_time=True):
                            # Use the history-aware chain
                            response = st.session_state.rag_chain.invoke(
                                {"question": safe_prompt}, # Use the safe prompt here
                                config={"configurable": {"session_id": "streamlit_session"}}
                            )
                            # Response is now a dictionary
                            assist_msg = response.get("answer", "Sorry, I couldn't formulate an answer.")
                            source_docs = response.get("source_documents", [])

                            # Extract relevant metadata, ensuring hashable types
                            source_metadata_list = []
                            if source_docs:
                                for doc in source_docs:
                                    if hasattr(doc, 'metadata'):
                                        meta = doc.metadata
                                        # Ensure only hashable types are stored
                                        clean_meta = {
                                            'source': str(meta.get('source', 'Unknown')), # Ensure string
                                            'page': meta.get('page') # Keep as is (likely int/None/str)
                                            # Add other needed metadata keys here, ensuring they are converted if necessary
                                        }
                                        source_metadata_list.append(clean_meta)

                            # Add assistant response to Streamlit chat history
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": assist_msg,
                                    "sources": source_metadata_list # Store the cleaned list
                                }
                            )
                            # Display final answer in the placeholder
                            message_placeholder.markdown(assist_msg)

                    except Exception as e:
                        st.error(f"Error during RAG chain execution: {e}")
                        assist_msg = f"Sorry, an error occurred: {e}"
                        # Add error message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": assist_msg, "sources": []})
                        # Display error message in placeholder
                        message_placeholder.markdown(assist_msg)

            else:
                # If input was flagged or filtering failed
                assist_msg = "Your query was flagged as potentially harmful or could not be processed. Please try rephrasing."
                st.session_state.messages.append({"role": "assistant", "content": assist_msg, "sources": []}) # Add assistant warning
                with st.chat_message("assistant", avatar="🤖"): # Display assistant warning with avatar
                    st.markdown(assist_msg)
        else:
             st.warning("RAG chain not ready. Please wait for initialization.")

    # --- Previous structure - commented out/removed ---
    # # React to user input only if RAG chain is ready
    # if 'rag_chain' in st.session_state and st.session_state.rag_chain:
#{{ ... }}
