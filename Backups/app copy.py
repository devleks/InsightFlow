# ... (rest of the code remains the same)

     page_icon=config.get('ui', {}).get('page_icon', '📄')
     )
     # Display Filename in Sidebar if config loaded
     doc_path_display = config.get('data_source', {}).get('path', 'Not configured')
     if doc_path_display != 'Not configured':
         # Attempt to get just the filename
         try:
             doc_filename = os.path.basename(doc_path_display)
             st.sidebar.caption(f"Current Document: **{doc_filename}**")
         except Exception:
             st.sidebar.caption(f"Current Document Path: {doc_path_display}") # Fallback to full path
     else:
         st.sidebar.warning("Document source path not configured in config.yaml.")

     # Display Title and Header early in the main area
     st.title(config.get('ui', {}).get('title', 'InsightFlow - Chat with Your Documents'))
     st.header(config.get('ui', {}).get('header', 'Ask questions about your uploaded data'))

     # --- Initialize Components ---
     embedding_function = get_embedding_function(config)

# ... (rest of the code remains the same)

 def check_api_key(env_var_name):
     """Checks if the required API key is present in environment variables."""
     api_key = os.getenv(env_var_name)
     # Display errors in the sidebar during initialization
     if not api_key:
         st.sidebar.error(f"Env variable '{env_var_name}' not set.")
         return False
     return True

# ... (rest of the code remains the same)

             # Note: OpenAIEmbeddings uses OPENAI_API_KEY env var by default if not passed explicitly
             return OpenAIEmbeddings(model=model_name)
         except Exception as e:
             st.sidebar.error(f"Failed to init OpenAI embeddings: {e}")
             return None
     # --- Add other providers like HuggingFace, Azure OpenAI here ---
     # elif provider == 'huggingface': # Example placeholder
# ... (rest of the code remains the same)

             #    return HuggingFaceEmbeddings(model_name=model_name)
     else:
         st.sidebar.error(f"Unsupported embedding provider: {provider}")
         return None

# ... (rest of the code remains the same)

 def load_and_split_documents(config):
# ... (rest of the code remains the same)

     source_path = config.get('data_source', {}).get('path', None)

     if not source_path or not os.path.exists(source_path):
         # Already handled by sidebar caption logic, but double-check
         if doc_path_display == 'Not configured': # Avoid duplicate message if path was missing
             st.sidebar.error(f"Doc path '{source_path}' not found.")
         return None

     documents = []
# ... (rest of the code remains the same)

             #     pass
         else:
             st.sidebar.error(f"Unsupported data source type: {source_type}")
             return None

         if not documents:
             st.sidebar.warning(f"No documents loaded from {source_path}.")
             return None

         # Split documents
# ... (rest of the code remains the same)

         # TODO: Make chunk size/overlap configurable?
         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
         splits = text_splitter.split_documents(documents)
         st.sidebar.success(f"Split {len(documents)} doc(s) into {len(splits)} chunks.")
         return splits

     except Exception as e:
         st.sidebar.error(f"Error loading/splitting docs: {e}")
         return None

# ... (rest of the code remains the same)

 def initialize_vector_store(config, embedding_function):
# ... (rest of the code remains the same)

                 embedding_function=embedding_function
             )
             return vector_store
         except Exception as e:
             st.sidebar.error(f"Failed to init Chroma DB at {persist_dir}: {e}")
             return None
     # --- Add other vector store types here ---
     else:
         st.sidebar.error(f"Unsupported vector store type: {store_type}")
         return None

# ... (rest of the code remains the same)

 def ensure_vector_store_populated(vector_store, config):
# ... (rest of the code remains the same)

             count = vector_store._collection.count()
             if count == 0:
                 st.sidebar.info(f"DB collection '{vector_store._collection.name}' empty. Ingesting...")
                 needs_population = True
             else:
                 st.sidebar.success(f"DB collection '{vector_store._collection.name}' has {count} docs.")
         except Exception as e: # Handle case where collection might not exist yet or other errors
             st.sidebar.warning(f"Could not get DB count, assuming ingestion needed (Error: {e})")
             needs_population = True

         if needs_population:
             st.sidebar.info(f"Loading docs from {config.get('data_source', {}).get('path', 'N/A')}...")
             doc_splits = load_and_split_documents(config)
             if doc_splits:
                 st.sidebar.info(f"Ingesting {len(doc_splits)} chunks...")
                 # Use from_documents to populate
                 # Re-initialize with from_documents if it's the first time
                 persist_dir = config.get('vector_store', {}).get('persist_directory', './vector_db')
# ... (rest of the code remains the same)

                     collection_name=collection_name
                 )
                 # vector_store.persist() # Chroma persists automatically with persist_directory
                 st.sidebar.success("Ingestion complete!")
             else:
                 st.sidebar.error("Doc loading/splitting failed. Cannot populate DB.")
                 return False # Indicate failure
         return True # Indicate success or already populated

     except Exception as e:
         st.sidebar.error(f"Error during DB population check/ingestion: {e}")
         return False

# ... (rest of the code remains the same)

 def filter_input_with_guard(user_input, config):
# ... (rest of the code remains the same)

     if provider == 'groq':
         if not check_api_key(env_var):
             # Keep runtime errors in main area
             st.error("Cannot filter input: Groq API key not found for filter model.")
             return None # Indicate failure
         try:
             client = Groq(api_key=os.getenv(env_var))
# ... (rest of the code remains the same)

             if 'unsafe' in response: # Llama Guard typically responds 'safe' or 'unsafe
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

# ... (rest of the code remains the same)

 def get_llm(config):
# ... (rest of the code remains the same)

             return ChatOpenAI(model_name=model_name, temperature=temperature)
         except Exception as e:
             st.sidebar.error(f"Failed to init OpenAI LLM: {e}")
             return None
     # --- Add other providers like Groq, Anthropic here ---
     # elif provider == 'groq':
# ... (rest of the code remains the same)

         #    pass
     else:
         st.sidebar.error(f"Unsupported LLM provider: {provider}")
         return None

# ... (rest of the code remains the same)

 def create_rag_chain(retriever, llm, config):
# ... (rest of the code remains the same)

         page_icon=config.get('ui', {}).get('page_icon', '📄')
     )
     # Display Title and Header early in the main area
     st.title(config.get('ui', {}).get('title', 'InsightFlow - Chat with Your Documents'))
     st.header(config.get('ui', {}).get('header', 'Ask questions about your uploaded data'))

     # Display Filename in Sidebar if config loaded
     doc_path_display = config.get('data_source', {}).get('path', 'Not configured')
     if doc_path_display != 'Not configured':
# ... (rest of the code remains the same)

             st.session_state.rag_chain = create_rag_chain(retriever, st.session_state.llm, config)

         # Check if RAG chain is ready before proceeding with chat
         rag_ready = 'rag_chain' in st.session_state and st.session_state.rag_chain
         if 'rag_chain' in st.session_state and st.session_state.rag_chain:
             st.sidebar.success("RAG chain ready!")

             # --- Chat Interface ---
             # Display existing messages
# ... (rest of the code remains the same)

                     st.rerun()

         else:
             # RAG chain failed to initialize, status already shown in sidebar
             # RAG chain failed to initialize, but page config/title/header are set
             pass # Error messages were displayed in the sidebar during init
     else:
         # Vector store failed, status already shown in sidebar
         # Vector store failed, but page config/title/header are set
         pass # Error messages were displayed in the sidebar during init

-    # Example: Displaying a config value
-    doc_path = config.get('data_source', {}).get('path', 'Not configured')
-    st.sidebar.write(f"Looking for documents in: {doc_path}")
-
 else:
     # This error should occur outside the main 'if config:' block if YAML fails
     st.error("Application cannot start without a valid configuration file.")