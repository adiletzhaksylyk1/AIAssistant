import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
import requests
from pymongo import MongoClient
from constitution_scraper import scrape_constitution
import time

# Constants
CONSTITUTION_URL = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
TEMP_DIR = "temp_docs"
MONGODB_URI = "mongodb+srv://adiletzhaksylyk2:7psi6nPCwrRFEYe8@cluster0.fdndizc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "constitution_db"
COLLECTION_NAME = "document_embeddings"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create temporary directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Page setup
st.set_page_config(page_title="Kazakhstan Constitution AI Assistant", layout="wide")
st.title("Kazakhstan Constitution AI Assistant")


# Function to check if Ollama is running and available
def check_ollama_status(model_name="llama2"):
    try:
        ollama = Ollama(base_url="http://localhost:11434", model=model_name)
        _ = ollama.invoke("test")
        return True
    except Exception as e:
        st.error(f"Error connecting to Ollama (model={model_name}): {e}")
        return False


# Function to check if MongoDB is running and available
def check_mongodb_status():
    try:
        client = MongoClient(MONGODB_URI)
        # Check connection
        client.admin.command('ping')
        return True
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return False

# Function to initialize the knowledge base
def initialize_knowledge_base(document_paths, force_reload=False):
    # Check if services are available
    if not check_ollama_status() or not check_mongodb_status():
        st.error("Cannot initialize knowledge base: Ollama or MongoDB is not available")
        return False

    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # If force reload, clear the collection
    if force_reload:
        collection.drop()
        collection = db[COLLECTION_NAME]

    # Process documents
    documents = []
    for path in document_paths:
        # Determine loader based on file extension
        if path.endswith('.txt'):
            loader = TextLoader(path)
        elif path.endswith('.pdf'):
            loader = PyPDFLoader(path)
        elif path.endswith('.docx'):
            loader = Docx2txtLoader(path)
        else:
            continue  # Skip unsupported files

        documents.extend(loader.load())

    # If no documents, exit
    if not documents:
        return False

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_chunks = text_splitter.split_documents(documents)

    # Create embeddings using Ollama
    embeddings = OllamaEmbeddings(model=model_option)

    # Check if documents already in DB
    if collection.count_documents({}) > 0 and not force_reload:
        # Use existing vector store
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index"
        )
    else:
        # Create new vector store
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=doc_chunks,
            embedding=embeddings,
            collection=collection,
            index_name="vector_index"
        )

    # Set up conversation chain
    setup_conversation_chain(vector_store)
    return True


# Sidebar for configuration and settings
with st.sidebar:
    st.header("Configuration")

    # MongoDB connection string
    mongodb_uri = st.text_input("MongoDB URI", value=MONGODB_URI)

    # Check if services are running
    st.subheader("Service Status")

    if check_ollama_status():
        st.success("✅ Ollama is running")
    else:
        st.error("❌ Ollama is not running. Please start Ollama service.")

    if check_mongodb_status():
        st.success("✅ MongoDB is running")
    else:
        st.error("❌ MongoDB is not running. Please start MongoDB service.")

    # Model selection
    model_option = st.selectbox(
        "Select Ollama Model",
        ["llama2", "mistral", "gemma", "phi3"]
    )

    # Temperature setting
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    # Adding the constitution data
    if st.button("Load Constitution Data"):
        with st.spinner("Fetching and processing Constitution data..."):
            # Use our constitution scraper to get the data
            success = scrape_constitution()
            if success:
                constitution_path = os.path.join(TEMP_DIR, "constitution.txt")
                # Process the constitution file
                try:
                    initialize_knowledge_base([constitution_path], force_reload=True)
                    st.success("Constitution data loaded and processed!")
                except Exception as e:
                    st.error(f"Error processing constitution data: {e}")
            else:
                st.error("Failed to fetch constitution data.")



# Function to set up conversation chain
def setup_conversation_chain(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = Ollama(model=model_option, temperature=temperature)

    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=True
    )


# File uploader
uploaded_files = st.file_uploader("Upload additional documents",
                                  accept_multiple_files=True,
                                  type=['txt', 'pdf', 'docx'])

if uploaded_files:
    # Save uploaded files to temp directory
    temp_paths = []
    for file in uploaded_files:
        temp_file_path = os.path.join(TEMP_DIR, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())
        temp_paths.append(temp_file_path)

    # Process these documents
    if st.button("Process Uploaded Documents"):
        with st.spinner("Processing documents..."):
            success = initialize_knowledge_base(temp_paths)
            if success:
                st.success(f"Processed {len(temp_paths)} documents!")
            else:
                st.error("Failed to process documents. Check file formats and services.")

# Try to initialize knowledge base from existing MongoDB collection
if check_ollama_status() and check_mongodb_status() and not st.session_state.conversation:
    try:
        client = MongoClient(mongodb_uri)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        if collection.count_documents({}) > 0:
            embeddings = OllamaEmbeddings(model=model_option)
            vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=embeddings,
                index_name="vector_index"
            )
            setup_conversation_chain(vector_store)
            st.success("Loaded existing knowledge base from MongoDB!")
    except Exception as e:
        st.error(f"Error loading existing knowledge base: {e}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask about the Constitution of Kazakhstan"):
    if not check_ollama_status():
        st.error("Ollama service is not available. Please start Ollama.")
    elif not check_mongodb_status():
        st.error("MongoDB service is not available. Please start MongoDB.")
    elif not st.session_state.conversation:
        st.error("Please initialize the knowledge base first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                response = st.session_state.conversation.invoke({"question": prompt})
                ai_response = response.get("answer", "I couldn't find information about that in the Constitution.")

                # Display AI response
                message_placeholder.write(ai_response)

                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

                # Update conversation history
                st.session_state.chat_history.append((prompt, ai_response))

            except Exception as e:
                message_placeholder.write(f"Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

# Add information about the application
with st.expander("About the Kazakhstan Constitution AI Assistant"):
    st.write("""
    This application allows you to ask questions about the Constitution of the Republic of Kazakhstan.

    Features:
    - Chat with an AI (powered by Ollama) about the Constitution
    - Upload additional documents for context
    - Store conversation history
    - MongoDB vector database for efficient retrieval

    The AI uses MongoDB to store and retrieve relevant information from the Constitution and any uploaded documents.
    """)