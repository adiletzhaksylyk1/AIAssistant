# Kazakhstan Constitution AI Assistant

This is an MVP (Minimum Viable Product) of an AI Assistant that can answer questions related to the Constitution of the Republic of Kazakhstan.

## Features

- Interactive chat interface using Streamlit
- Integration with Ollama for open-source LLM inference
- Vector storage using MongoDB for efficient retrieval
- Document upload functionality (multiple files, various formats)
- Automatic context-based retrieval and answering

## Setup Instructions

### Prerequisites

- Python 3.8+
- MongoDB
- Ollama

### Installation

#### Method 1: Docker Compose (Recommended)

1. Clone this repository
2. Run the application using Docker Compose:

```bash
docker-compose up -d
```

This will start:
- The Streamlit application
- MongoDB for vector storage
- Ollama service with pre-downloaded models

3. Access the application at http://localhost:8501

#### Method 2: Manual Setup

1. Clone this repository
2. Install MongoDB and start the service
3. Install Ollama and start the service
4. Install the required packages:

```bash
pip install -r requirements.txt
```

5. Run the application:

```bash
streamlit run app.py
```

### Using the Application

1. Make sure MongoDB and Ollama services are running (check the sidebar status indicators)
2. Click "Load Constitution Data" to fetch and process the Constitution
3. Start asking questions about the Constitution
4. Upload additional documents if needed for more context

## Technical Details

This application uses:

- **Streamlit**: For the web interface
- **LangChain**: For the document processing and retrieval pipeline
- **MongoDB**: As the vector database for storing embeddings
- **Ollama**: For local language model inference

## Document Processing

The application can process:
- PDF files
- Text files
- DOCX (Word) files

Each document is split into chunks, embedded, and stored in MongoDB for efficient retrieval.

## Supported Ollama Models

By default, the application supports:
- llama2
- mistral
- gemma
- phi3

You can add more models by pulling them through Ollama.

## File Structure

- `app.py`: Main Streamlit application
- `constitution_scraper.py`: Utility to fetch and process the Constitution text
- `requirements.txt`: Required Python packages
- `temp_docs/`: Temporary storage for uploaded documents
- `docker-compose.yml`: Docker Compose configuration

## Notes

- For a production deployment, consider setting up authentication for MongoDB
- Add user authentication for multi-user support
- Implement rate limiting and more extensive error handling


![image](https://github.com/user-attachments/assets/055f51a2-d305-4f06-a993-8b153e0604c8)
![image](https://github.com/user-attachments/assets/0d0b32db-525b-4b5f-84de-610c1eb7919c)

