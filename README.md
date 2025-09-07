# Agentic Assistant

An advanced agentic assistant chatbot built with LangChain and LangGraph that handles complex, multi-step user queries through a multi-agent architecture. Features persistent memory, document processing with RAG, image analysis, and a modern React web interface.

## Key Features

- **Multi-Agent Architecture**: Specialized agents for web search, document analysis, image processing, and more
- **Web Search & Scraping**: Finds and summarizes information from the internet
- **Document Processing**: Upload and analyze PDFs, DOCX, TXT, CSV, XLSX, and code files with RAG
- **Image Analysis**: Analyze images with multimodal capabilities
- **Long-Term Memory**: Stores and retrieves user preferences across sessions
- **Modern React Interface**: Clean, responsive web UI with real-time chat
- **Multiple LLM Providers**: Supports Google Qwen/Gemini and Ollama (local models)

## Architecture

The assistant uses a supervisor-agent architecture built with LangGraph:

**Supervisor Agent** (Routes requests to appropriate specialized agents)
- Web Search Agent (Searches the web using Tavily)
- Web Scraping Agent (Extracts content from URLs)
- Image Analysis Agent (Analyzes images with multimodal capabilities)
- RAG/Document Agent (Answers questions about uploaded documents)
- Memory Agent (Manages long-term memory)
- Final Response Agent (Generates the final user-facing response)

## Prerequisites

- Python 3.12
- Node.js 14+ (for React frontend)
- Google API key for Qwen/Gemini (optional)
- Tavily API key for web search
- Ollama with embedding models (for LangMem)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/biuinvincible/agentic-chatbot.git
   cd agentic-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\\Scripts\\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Pull required Ollama models:
   ```bash
   ollama pull bge-m3:567m
   ```

5. Install frontend dependencies:
   ```bash
   cd frontend_react
   npm install
   cd ..
   ```

6. Set up environment variables:
   Create a `.env` file in the project root:
   ```env
   TAVILY_API_KEY=your_tavily_api_key
   GOOGLE_API_KEY=your_google_api_key  # Optional
   ```

## Usage

### Web Interface

1. Start the backend API server:
   ```bash
   python backend_api.py
   ```
   The backend runs on http://localhost:8000

2. In a new terminal, start the React frontend:
   ```bash
   cd frontend_react
   npm run dev
   ```
   The frontend is accessible at http://localhost:3000

3. Open your browser to http://localhost:3000

### API Server

Run just the backend API:
```bash
python backend_api.py
```

Key API endpoints:
- `POST /session` - Create a new session
- `POST /upload-document` - Upload and process a document
- `POST /chat` - Send a chat message and receive a response

## Agent System

The assistant uses specialized agents for different tasks:

- **Web Search Agent**: Searches the web using Tavily
- **Web Scraping Agent**: Extracts content from URLs
- **Image Analysis Agent**: Analyzes images (requires Qwen/Gemini)
- **RAG/Document Agent**: Answers questions about uploaded documents
- **Memory Agent**: Manages long-term memory
- **Supervisor Agent**: Routes requests to appropriate agents
- **Final Response Agent**: Generates the final user-facing response

## Document Processing

Supported formats: PDF, DOCX, TXT, CSV, XLSX, and code files.

Workflow:
1. Upload documents through the web interface
2. System automatically processes and creates embeddings
3. Ask detailed questions about document content
4. RAG agent retrieves relevant chunks and generates responses

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required for web search
TAVILY_API_KEY=your_tavily_api_key

# Required for Qwen/Gemini provider (optional)
GOOGLE_API_KEY=your_google_api_key

# Optional for Ollama provider
OLLAMA_BASE_URL=http://localhost:11434
```

### LLM Providers

1. **Google Qwen/Gemini** (default):
   - Set `GOOGLE_API_KEY`
   - Uses `gemini-2.5-flash` model

2. **Ollama** (local):
   - Install and start [Ollama](https://ollama.com/)
   - Pull required models:
     ```bash
     ollama pull granite3.3:latest
     ollama pull gemma3:1b
     ollama pull bge-m3:567m
     ```

## Troubleshooting

Common issues:
- **Port already in use**: Kill the process or change the port
- **Network Error**: Ensure backend API runs on port 8000
- **Missing API keys**: Verify `.env` file contains required keys
- **Document processing failures**: Check Ollama and required models

Enable verbose logging:
```bash
export DEBUG=1
python backend_api.py
```