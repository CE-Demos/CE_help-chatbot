import os
import streamlit as st
import google.generativeai as genai

# from docx import Document as DocxDocument # Alias to avoid conflict
# import faiss
# from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# import vertexai

# Store your API key securely - consider environment variables or secrets management

# If you have gcloud CLI authenticated, the library might pick up credentials automatically
try:
    genai.configure()
    print("Gemini API configured using default Google Cloud credentials.")
except Exception as e:
    print(f"Error configuring Gemini API with default credentials: {e}")
    print("Falling back to API key from environment variable or direct input.")
    # You can still provide the API key as a fallback if needed
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input("Enter your Gemini API key:", type="password")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        st.error("Gemini API key is required. Please enter it in the sidebar or set the GEMINI_API_KEY environment variable.")
        st.stop()


# Select the Gemini model you want to use
model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')        

st.title("Gemini Chatbot")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Display chat messages from history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask me anything:"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from Gemini
    # Generate response from Gemini
    try:
        response = model.generate_content(prompt)

        # Check for errors using response.prompt_feedback
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            st.error(f"The response was blocked due to: {response.prompt_feedback.block_reason}")
            if response.prompt_feedback.safety_ratings:
                for rating in response.prompt_feedback.safety_ratings:
                    st.warning(f"- {rating.category}: {rating.probability}")
            st.stop()
        elif response.text:
            st.session_state["messages"].append({"role": "assistant", "content": response.text})
            with st.chat_message("assistant"):
                st.markdown(response.text)
        else:
            st.warning("No response received from the model.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# # --- Configuration ---
# PROJECT_ID = "ce-ai-chatbot"  # <--- CHANGE THIS
# LOCATION = "us-central1"        # <--- e.g., "us-central1"
# GEMINI_MODEL_NAME = "gemini-1.5-flash-001" # Or other suitable Gemini model
# EMBEDDING_MODEL_NAME = "text-embedding-004" # Or other suitable embedding model

# # --- Initialization ---
# try:
#     vertexai.init(project=PROJECT_ID, location=LOCATION)
# except Exception as e:
#     st.error(f"Vertex AI initialization failed. Check PROJECT_ID and LOCATION. Error: {e}")
#     st.stop() # Stop execution if initialization fails

# # --- Helper Functions ---

# @st.cache_resource # Cache the resourceheavy embedding model
# def get_embedding_model():
#     """Initializes and returns the Vertex AI embedding model."""
#     try:
#         return VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME)
#     except Exception as e:
#         st.error(f"Failed to initialize embedding model: {e}")
#         st.stop()

# @st.cache_resource # Cache the LLM model instance
# def get_llm_model():
#     """Initializes and returns the Vertex AI LLM (Gemini)."""
#     try:
#         # Adjust parameters as needed (temperature, max_output_tokens, etc.)
#         return VertexAI(model_name=GEMINI_MODEL_NAME, temperature=0.2, max_output_tokens=1024)
#     except Exception as e:
#         st.error(f"Failed to initialize LLM model: {e}")
#         st.stop()

# def load_word_document(uploaded_file):
#     """Loads text content from an uploaded Word document."""
#     try:
#         # Read the uploaded file's bytes directly
#         doc = DocxDocument(uploaded_file)
#         full_text = []
#         for para in doc.paragraphs:
#             full_text.append(para.text)
#         return '\n'.join(full_text)
#     except Exception as e:
#         st.error(f"Error reading Word document: {e}")
#         return None

# def create_vector_store(text_content, embeddings_model):
#     """Splits text, creates embeddings, and builds a FAISS vector store."""
#     if not text_content:
#         return None
#     try:
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, # Adjust chunk size as needed
#             chunk_overlap=100 # Adjust overlap as needed
#         )
#         texts = text_splitter.split_text(text_content)

#         if not texts:
#             st.warning("Document processed, but no text chunks were generated. Is the document empty?")
#             return None

#         st.info(f"Splitting document into {len(texts)} chunks...")

#         # Create embeddings and FAISS store
#         # Note: FAISS.from_texts handles embedding creation internally
#         vector_store = FAISS.from_texts(texts, embeddings_model)
#         st.success(f"Vector store created successfully with {len(texts)} chunks!")
#         return vector_store
#     except Exception as e:
#         st.error(f"Error creating vector store: {e}")
#         return None

# def create_rag_chain(vector_store, llm_model):
#     """Creates the RetrievalQA chain."""
#     if vector_store is None:
#         return None
#     try:
#         retriever = vector_store.as_retriever(
#             search_type="similarity", # Or "mmr"
#             search_kwargs={'k': 5} # Retrieve top 5 relevant chunks
#         )

#         # --- Crucial RAG Prompt ---
#         prompt_template = """SYSTEM: You are a helpful assistant. Answer the following question based ONLY on the context provided below. If the context doesn't contain the answer, state that you cannot answer based on the provided document. Do not use any prior knowledge.

#         CONTEXT:
#         {context}

#         QUESTION:
#         {question}

#         ANSWER:"""
#         rag_prompt = PromptTemplate(
#             template=prompt_template, input_variables=["context", "question"]
#         )

#         # Create the chain
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm_model,
#             chain_type="stuff", # Options: stuff, map_reduce, refine, map_rerank
#             retriever=retriever,
#             return_source_documents=True, # Optional: to see which chunks were used
#             chain_type_kwargs={"prompt": rag_prompt}
#         )
#         return qa_chain
#     except Exception as e:
#         st.error(f"Error creating RAG chain: {e}")
#         return None

# # --- Streamlit UI ---
# st.set_page_config(layout="wide")
# st.title("📄 Chat with your Word Document using Vertex AI Gemini")
# st.write(f"Powered by: Vertex AI Gemini ({GEMINI_MODEL_NAME}) & Embeddings ({EMBEDDING_MODEL_NAME})")

# # File Uploader
# uploaded_file = st.file_uploader("Upload your .docx file", type="docx")

# # Initialize session state variables
# if 'vector_store' not in st.session_state:
#     st.session_state.vector_store = None
# if 'rag_chain' not in st.session_state:
#     st.session_state.rag_chain = None
# if 'processed_file_name' not in st.session_state:
#     st.session_state.processed_file_name = None

# # Process Document if uploaded and not already processed
# if uploaded_file is not None:
#     # Check if this is a new file or the same one re-uploaded
#     if st.session_state.processed_file_name != uploaded_file.name:
#         st.info(f"Processing document: {uploaded_file.name}")
#         with st.spinner("Loading and processing document... This may take a minute."):
#             document_text = load_word_document(uploaded_file)
#             if document_text:
#                 embeddings = get_embedding_model()
#                 st.session_state.vector_store = create_vector_store(document_text, embeddings)
#                 if st.session_state.vector_store:
#                     llm = get_llm_model()
#                     st.session_state.rag_chain = create_rag_chain(st.session_state.vector_store, llm)
#                     st.session_state.processed_file_name = uploaded_file.name # Mark as processed
#                 else:
#                     st.warning("Could not create vector store from the document.")
#                     # Reset state if processing failed
#                     st.session_state.vector_store = None
#                     st.session_state.rag_chain = None
#                     st.session_state.processed_file_name = None
#             else:
#                 # Reset state if loading failed
#                 st.session_state.vector_store = None
#                 st.session_state.rag_chain = None
#                 st.session_state.processed_file_name = None
#     else:
#         st.info(f"Document '{uploaded_file.name}' already processed.")

# # Chat Interface (only if document is processed successfully)
# if st.session_state.rag_chain is not None:
#     st.markdown("---")
#     st.header("Ask questions about the document:")
#     user_question = st.text_input("Your Question:")

#     if user_question:
#         with st.spinner("Thinking..."):
#             try:
#                 response = st.session_state.rag_chain({"query": user_question})
#                 st.subheader("Answer:")
#                 st.write(response["result"])

#                 # Optional: Display source documents for transparency
#                 with st.expander("Show relevant document chunks used"):
#                     for i, doc in enumerate(response["source_documents"]):
#                          # Limit page content display length
#                         display_content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
#                         st.write(f"**Chunk {i+1}:**")
#                         st.caption(display_content) # Use caption for smaller text
#             except Exception as e:
#                 st.error(f"An error occurred while fetching the answer: {e}")
# else:
#     st.info("Please upload a Word document to begin.")