# Pydantic Document Retrieval System

A RAG (Retrieval-Augmented Generation) application for searching and querying Pydantic documentation using vector embeddings and LLM.

## Features

- **Search Documentation**: Query the Pydantic documentation using natural language. The system retrieves relevant text and code chunks from a vector database and generates answers using an LLM.
- **Process URL**: Scrape content from any Pydantic documentation URL and ask questions about it.

## Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- API keys for:
  - Pinecone (vector database)
  - MongoDB (code chunks storage)
  - Groq (LLM)
  - Firecrawl (web scraping)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YogeshChebrolu/RAG_ON_CODE.git
cd RAG_ON_CODE
```

### 2. Create Virtual Environment

```bash
uv venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
uv sync
```

### 5. Configure Environment Variables

Create a `.env` file in the project root with your API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key
MONGO_CLENT_URI=your_mongodb_connection_string
GROQ_API_KEY=your_groq_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

## Running the Application

Start the Streamlit application:

```bash
uv run streamlit run app2.py
```

The application will open in your browser at `http://localhost:8501`.

## Testing the Application

### Tab 1: Search Documentation

1. Navigate to the "Search Documentation" tab
2. Enter a question about Pydantic (e.g., "What is pydantic ai?")
3. The system will:
   - Retrieve relevant text and code chunks from the vector database
   - Display the retrieved context in an expandable section
   - Generate and display an answer using the LLM

### Tab 2: Process URL

1. Navigate to the "Process URL" tab
2. Enter a Pydantic documentation URL (e.g., `https://docs.pydantic.dev/latest/concepts/models/`)
3. The system will scrape the content and display it
4. Enter a question about the documentation
5. The system will generate an answer based on the scraped content

## Project Structure

```
RAG_ON_CODE/
├── app2.py           # Main Streamlit application
├── pyproject.toml    # Project dependencies
├── .env              # Environment variables (not tracked)
├── .python-version   # Python version specification
└── README.md         # This file
```

## Dependencies

- `streamlit` - Web application framework
- `langchain-huggingface` - HuggingFace embeddings integration
- `langchain-pinecone` - Pinecone vector store integration
- `langchain-groq` - Groq LLM integration
- `pymongo` - MongoDB client
- `firecrawl` - Web scraping
- `python-dotenv` - Environment variable management
- `sentence-transformers` - Embedding models
