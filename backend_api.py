#!/usr/bin/env python3
"""
Main backend API for the Agentic Assistant with enhanced deep search capabilities.
This is the single, consolidated API file that contains all the latest features.
"""

import asyncio
import aiosqlite
import uuid
import os
import json
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# WebSocket support
from fastapi import WebSocket
import websockets

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.errors import GraphInterrupt
# Use the supervisor with deep search for enhanced capabilities
from agents.supervisor import create_agent_graph
from agents.file_processing.workflow import DocumentProcessor
from agents.memory_agent.manager import get_memory_manager, MemoryManager
from utils.logging_config import get_logger, setup_logging

# Global variables
llm = None
graph = None
memory = None
langmem_store = None
memory_manager = None

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Initialize logger
logger = get_logger(__name__)

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: str
    document_info: List[Dict[str, Any]] = []
    user_forced_agent: Optional[str] = None  # For user control feature

class ChatResponse(BaseModel):
    response: str
    session_id: str
    document_info: List[Dict[str, Any]]
    messages: List[ChatMessage]

class DocumentInfo(BaseModel):
    doc_id: str
    file_name: str
    chunk_count: int
    processing_complete: bool

class DocumentProcessResponse(BaseModel):
    status: str
    document_info: Optional[Dict[str, Any]]
    session_id: str
    message: str

class AgentListResponse(BaseModel):
    agents: List[str]

class ProgressUpdate(BaseModel):
    task_id: str
    status: str
    message: str
    percent: Optional[int] = None

# Conversation History Models
class ConversationBase(BaseModel):
    title: str

class ConversationCreate(BaseModel):
    title: str
    id: Optional[str] = None

class Conversation(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class ConversationMessage(BaseModel):
    role: str
    content: str
    created_at: str

class ConversationDetail(Conversation):
    messages: List[ConversationMessage]

def initialize_llm(provider: str, model_name: str = None):
    """Initializes the LLM based on the provider."""
    if provider == "ollama":
        model = model_name if model_name else "granite3.3:latest"
        return ChatOllama(model=model)
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'ollama' or 'gemini'.")

async def process_memory_operations(result: Dict[str, Any], session_id: str):
    """Process memory operations from agent responses."""
    # With our new direct integration, this function is no longer needed
    # Memory operations are now handled directly by agents using LangMem tools
    pass

async def init_conversation_db():
    """Initialize the conversation history database tables"""
    async with aiosqlite.connect("checkpoints.db") as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        """)
        
        await conn.commit()
        print("[Backend] Conversation history database initialized")

async def cleanup_unused_sessions():
    """Clean up sessions that were created but never used (no messages)"""
    try:
        async with aiosqlite.connect("checkpoints.db") as conn:
            # Find conversations with no messages
            cursor = await conn.execute("""
                SELECT c.id FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE m.conversation_id IS NULL
                AND c.created_at < datetime('now', '-1 hour')
            """)
            unused_convs = await cursor.fetchall()
            
            # Delete unused conversations
            for conv in unused_convs:
                conv_id = conv[0]
                await conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
                print(f"[Backend] Cleaned up unused conversation: {conv_id}")
            
            await conn.commit()
    except Exception as e:
        print(f"[Backend] Error cleaning up unused sessions: {e}")

def cleanup_old_files(max_age_hours: int = 24):
    """Clean up old files in the uploads directory"""
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        return
        
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getctime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    print(f"Error cleaning up file {file_path}: {e}")

@asynccontextmanager
async def lifespan_context(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize LLM and graph on startup, cleanup on shutdown"""
    global llm, graph, memory, langmem_store, memory_manager
    
    # Set up logging
    setup_logging()
    logger.info("Starting Agentic Assistant API")
    
    # Initialize conversation database
    await init_conversation_db()
    
    # Initialize LLM (you can make this configurable via environment variables)
    provider = os.getenv("LLM_PROVIDER", "gemini")
    
    try:
        if provider == "ollama":
            main_llm = initialize_llm(provider, "granite3.3:latest")
            # Use Google Qwen/Gemini for final response agent for better quality
            from langchain_google_genai import ChatGoogleGenerativeAI
            final_response_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        else:
            main_llm = initialize_llm(provider)
            final_response_llm = None
            
        conn = await aiosqlite.connect("checkpoints.db")
        memory = AsyncSqliteSaver(conn=conn)
        
        # Initialize LangMem store
        langmem_store = InMemoryStore(
            index={
                "dims": 512,  # bge-m3:567m embedding dimension
                "embed": "ollama:bge-m3:567m",  # Ollama embedding model
            }
        )
        
        # Initialize memory manager
        memory_manager = get_memory_manager(langmem_store)
        
        # Create the agent graph with user control
        graph = create_agent_graph(main_llm, final_response_llm).compile(
            checkpointer=memory,
            store=langmem_store
        )
        
        print(f"LLM initialized with provider: {provider}")
        print("LangMem store and memory manager initialized")
        
        # Clean up old files and unused sessions on startup
        cleanup_old_files()
        await cleanup_unused_sessions()
    except Exception as e:
        print(f"[Backend] Error initializing LLM or LangMem: {e}")
    
    # Start background task for periodic cleanup
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Cancel cleanup task on shutdown
    cleanup_task.cancel()
    
    # Cleanup
    if memory and hasattr(memory, 'conn'):
        await memory.conn.close()

async def periodic_cleanup():
    """Periodically clean up unused sessions"""
    while True:
        try:
            await cleanup_unused_sessions()
            await asyncio.sleep(3600)  # Run every hour
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Backend] Error in periodic cleanup: {e}")
            await asyncio.sleep(3600)  # Continue running even if there's an error

app = FastAPI(title="Agentic Assistant API", version="1.0.0", lifespan=lifespan_context)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React and Vue dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Agentic Assistant API", "version": "1.0.0"}

@app.get("/agents", response_model=AgentListResponse)
async def get_available_agents():
    """Get list of available agents for user control"""
    agents = [
        "Auto",  # Let supervisor decide
        "web_search_agent",
        "web_scraping_agent", 
        "image_analysis_agent",
        "rag_agent",
        "deep_research_agent",
        "deep_search_agent",
        "deep_researcher_agent",  # NEW: Added deep researcher agent
        "memory_agent",
        "final_response_agent"
    ]
    return AgentListResponse(agents=agents)

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    active_connections[session_id] = websocket
    try:
        while True:
            # Keep the connection alive
            data = await websocket.receive_text()
            # Echo back for now - can be extended for client commands
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        if session_id in active_connections:
            del active_connections[session_id]
        print(f"WebSocket disconnected for session {session_id}")

async def send_progress_update(session_id: str, update: ProgressUpdate):
    """Send progress update to WebSocket client if connected"""
    if session_id in active_connections:
        try:
            websocket = active_connections[session_id]
            await websocket.send_text(json.dumps(update.dict()))
        except Exception as e:
            print(f"Error sending progress update: {e}")

async def stream_response(config, input_dict, session_id: str):
    """Stream response tokens from the agent graph"""
    global graph
    
    # Send typing indicator
    await send_progress_update(session_id, ProgressUpdate(
        task_id="chat_response",
        status="typing",
        message="Assistant is typing...",
        percent=None
    ))
    
    try:
        # Use the graph to stream responses
        async for chunk in graph.astream(input_dict, config):
            if "messages" in chunk and chunk["messages"]:
                # Get the latest message
                latest_message = chunk["messages"][-1]
                if hasattr(latest_message, 'content') and latest_message.content:
                    # Send each token as it's generated
                    await send_progress_update(session_id, ProgressUpdate(
                        task_id="chat_response",
                        status="streaming",
                        message=latest_message.content,
                        percent=None
                    ))
                    yield latest_message.content
                    
        # Send completion indicator
        await send_progress_update(session_id, ProgressUpdate(
            task_id="chat_response",
            status="completed",
            message="Response completed",
            percent=100
        ))
    except Exception as e:
        await send_progress_update(session_id, ProgressUpdate(
            task_id="chat_response",
            status="error",
            message=f"Error: {str(e)}",
            percent=None
        ))
        raise e

@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    """Process a chat message and stream the response"""
    global graph
    
    if not graph:
        raise HTTPException(status_code=500, detail="LLM not initialized")
    
    try:
        # Create input message
        input_message = HumanMessage(content=request.message)
        
        # Run the agent graph
        input_dict = {
            "messages": [input_message],
            "session_id": request.session_id,
            "document_info": request.document_info
        }
        
        # Add user_forced_agent if provided
        if request.user_forced_agent and request.user_forced_agent != "Auto":
            input_dict["user_forced_agent"] = request.user_forced_agent
            
        config = {"configurable": {"thread_id": request.session_id}}
        
        # Return streaming response
        return StreamingResponse(stream_response(config, input_dict, request.session_id), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Process a chat message and return the response"""
    global graph
    
    if not graph:
        raise HTTPException(status_code=500, detail="LLM not initialized")
    
    try:
        logger.info(f"=== BACKEND API: Received chat request ===")
        logger.info(f"Message: {request.message}")
        logger.info(f"Session ID: {request.session_id}")
        logger.info(f"Document count: {len(request.document_info)}")
        logger.info(f"User forced agent: {request.user_forced_agent}")
        
        # Create input message
        input_message = HumanMessage(content=request.message)
        
        # Use the regular graph
        target_graph = graph
            
        # Run the agent graph
        input_dict = {
            "messages": [input_message],
            "session_id": request.session_id,
            "document_info": request.document_info,
            "next": "supervisor",  # Start with supervisor
            "user_forced_agent": request.user_forced_agent if request.user_forced_agent and request.user_forced_agent != "Auto" else None
        }
            
        config = {"configurable": {"thread_id": request.session_id}}
        # Pass the config through to agents
        input_dict["config"] = config
        logger.info(f"Invoking agent graph with session: {request.session_id}")
        
        result = await target_graph.ainvoke(input_dict, config)
        # Log important parts of result instead of entire result object
        messages_count = len(result.get("messages", [])) if isinstance(result, dict) else "Unknown"
        logger.info(f"Agent graph completed - Messages in result: {messages_count}")
        
        # Get the final response
        final_message = result["messages"][-1]
        response_content = final_message.content
        # print(f"Final response content: {response_content}")
        
        # Format messages for response
        messages = [
            ChatMessage(role="user", content=request.message),
            ChatMessage(role="assistant", content=response_content)
        ]
        
        response = ChatResponse(
            response=response_content,
            session_id=request.session_id,
            document_info=request.document_info,
            messages=messages
        )
        # print(f"Returning response: {response}")
        
        # Save messages to conversation history
        try:
            async with aiosqlite.connect("checkpoints.db") as conn:
                # Check if conversation exists, create if not
                cursor = await conn.execute(
                    "SELECT id FROM conversations WHERE id = ?",
                    (request.session_id,)
                )
                conv_row = await cursor.fetchone()
                
                if not conv_row:
                    # Create conversation only when first message is sent
                    await conn.execute(
                        "INSERT INTO conversations (id, title) VALUES (?, ?)",
                        (request.session_id, f"Conversation {request.session_id[:8]}")
                    )
                
                # Save user message
                await conn.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    (request.session_id, "user", request.message)
                )
                
                # Save assistant response
                await conn.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    (request.session_id, "assistant", response_content)
                )
                
                # Update conversation title if it's still the default and this is the first message
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (request.session_id,)
                )
                count_row = await cursor.fetchone()
                message_count = count_row[0] if count_row else 0
                
                if message_count == 2:  # First user message and first assistant response
                    # Generate a title based on the first user message
                    title = request.message[:50]
                    if len(request.message) > 50:
                        title += "..."
                    await conn.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (title, request.session_id)
                    )
                
                # Update conversation timestamp
                await conn.execute(
                    "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (request.session_id,)
                )
                
                await conn.commit()
        except Exception as e:
            print(f"Error saving conversation history: {e}")
            # Don't fail the request if history saving fails
        
        return response
        
    except GraphInterrupt as e:
        # Handle interrupt for clarification
        print(f"[Backend] Caught GraphInterrupt: {e}")
        interrupts = getattr(e, 'interrupts', [])
        print(f"[Backend] Interrupts: {interrupts}")
        if interrupts:
            # Extract the interrupt value
            interrupt_value = interrupts[0].value if hasattr(interrupts[0], 'value') else {}
            print(f"[Backend] Interrupt value: {interrupt_value}")
            if isinstance(interrupt_value, dict) and interrupt_value.get("type") == "clarification_request":
                clarification_question = interrupt_value.get("question", "Please provide clarification:")
                print(f"[Backend] Returning clarification question: {clarification_question}")
                
                # Return the clarification question to the user
                return ChatResponse(
                    response=clarification_question,
                    session_id=request.session_id,
                    document_info=request.document_info,
                    messages=[
                        ChatMessage(role="user", content=request.message),
                        ChatMessage(role="assistant", content=clarification_question)
                    ]
                )
        # Re-raise if not a clarification interrupt
        print(f"[Backend] Re-raising interrupt")
        raise e
        
    except Exception as e:
        print(f"BACKEND API ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/chat/resume", response_model=ChatResponse)
async def resume_chat(request: ChatRequest):
    """Resume an interrupted workflow with user input."""
    global graph
    
    if not graph:
        raise HTTPException(status_code=500, detail="LLM not initialized")
    
    try:
        # Get the user's response from the message
        user_response = request.message
        
        # Resume the interrupted workflow with user input
        config = {"configurable": {"thread_id": request.session_id}}
        
        # Use Command to resume with the user's response
        from langgraph.types import Command
        resume_command = Command(resume=user_response)
        
        result = await graph.ainvoke(resume_command, config)
        
        # Process result the same way as the regular chat endpoint
        final_message = result["messages"][-1]
        response_content = final_message.content
        
        # Format messages for response
        messages = [
            ChatMessage(role="user", content=user_response),
            ChatMessage(role="assistant", content=response_content)
        ]
        
        response = ChatResponse(
            response=response_content,
            session_id=request.session_id,
            document_info=request.document_info,
            messages=messages
        )
        
        # Save to conversation history
        try:
            async with aiosqlite.connect("checkpoints.db") as conn:
                # Save user message
                await conn.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    (request.session_id, "user", user_response)
                )
                
                # Save assistant response
                await conn.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    (request.session_id, "assistant", response_content)
                )
                
                # Update conversation timestamp
                await conn.execute(
                    "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (request.session_id,)
                )
                
                await conn.commit()
        except Exception as e:
            print(f"Error saving conversation history: {e}")
            # Don't fail the request if history saving fails
        
        return response
        
    except Exception as e:
        print(f"BACKEND API ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/upload-document", response_model=DocumentProcessResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = None
):
    """Upload and process a document"""
    if not session_id:
        # Don't create a new session automatically
        # Return an error indicating that a session is required
        raise HTTPException(status_code=400, detail="Session ID is required for document upload")
    
    print(f"[Backend] Uploading document: {file.filename}")
    print(f"[Backend] Session ID: {session_id}")
    
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        # Save file to permanent location for images, temporary for other files
        file_path = f"{uploads_dir}/{file.filename}"
        
        # Check if it's an image file
        _, ext = os.path.splitext(file.filename)
        ext = ext.lower()
        is_image = ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
        
        # For images, save to permanent location
        if is_image:
            file_path = f"{uploads_dir}/{file.filename}"
            print(f"[Backend] Saving image file to permanent location: {file_path}")
        else:
            # For non-images, use temporary location
            file_path = f"temp_{file.filename}"
            print(f"[Backend] Saving non-image file to temporary location: {file_path}")
            
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        print(f"[Backend] Saved file to: {file_path}")
        
        # Process document with progress updates
        async def progress_callback(message, percent=None):
            # Send progress update via WebSocket if connected
            await send_progress_update(session_id, ProgressUpdate(
                task_id="document_processing",
                status="processing",
                message=message,
                percent=percent
            ))
            print(f"[Backend] Processing {file.filename}: {message}")
        
        processor = DocumentProcessor(session_id, progress_callback)
        result = processor.upload_and_process(file_path)
        print(f"[Backend] Document processing result: Success")
        
        # Clean up temp file only for non-image files
        if not is_image:
            os.remove(file_path)
            print(f"[Backend] Cleaned up temp file: {file_path}")
        else:
            print(f"[Backend] Keeping image file for later analysis: {file_path}")
        
        # Send completion update
        await send_progress_update(session_id, ProgressUpdate(
            task_id="document_processing",
            status="completed",
            message="Document processing completed",
            percent=100
        ))
        
        if result["status"] == "success":
            print(f"[Backend] Document processed successfully")
            return DocumentProcessResponse(
                status="success",
                document_info=result["document_info"],
                session_id=session_id,
                message=f"Document '{file.filename}' processed successfully"
            )
        else:
            print(f"Document processing failed: {result['message']}")
            # Clean up file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up file after processing failure: {file_path}")
            return DocumentProcessResponse(
                status="error",
                document_info=None,
                session_id=session_id,
                message=result["message"]
            )
            
    except Exception as e:
        # Send error update
        await send_progress_update(session_id, ProgressUpdate(
            task_id="document_processing",
            status="error",
            message=f"Error processing document: {str(e)}",
            percent=None
        ))
        print(f"[Backend] Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# Conversation History API Endpoints
@app.post("/conversations", response_model=Conversation)
async def create_conversation(conversation: ConversationCreate):
    """Create a new conversation"""
    conversation_id = conversation.id if conversation.id else str(uuid.uuid4())
    
    # Generate a default title if none provided
    title = conversation.title if conversation.title else f"Conversation {conversation_id[:8]}"
    
    async with aiosqlite.connect("checkpoints.db") as conn:
        await conn.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conversation_id, title)
        )
        await conn.commit()
    
    # Return with placeholder timestamps (in a real implementation, you'd get actual timestamps)
    return Conversation(
        id=conversation_id,
        title=title,
        created_at="2023-01-01T00:00:00Z",
        updated_at="2023-01-01T00:00:00Z"
    )

@app.get("/conversations", response_model=List[Conversation])
async def list_conversations():
    """List all conversations"""
    async with aiosqlite.connect("checkpoints.db") as conn:
        cursor = await conn.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
        rows = await cursor.fetchall()
        
        conversations = []
        for row in rows:
            conversations.append(Conversation(
                id=row[0],
                title=row[1],
                created_at=row[2] if row[2] else "2023-01-01T00:00:00Z",
                updated_at=row[3] if row[3] else "2023-01-01T00:00:00Z"
            ))
            
        return conversations

@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with its messages"""
    async with aiosqlite.connect("checkpoints.db") as conn:
        # Get conversation info
        cursor = await conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,)
        )
        conv_row = await cursor.fetchone()
        
        if not conv_row:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages
        cursor = await conn.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,)
        )
        message_rows = await cursor.fetchall()
        
        messages = []
        for row in message_rows:
            messages.append(ConversationMessage(
                role=row[0],
                content=row[1],
                created_at=row[2] if row[2] else "2023-01-01T00:00:00Z"
            ))
            
        return ConversationDetail(
            id=conv_row[0],
            title=conv_row[1],
            created_at=conv_row[2] if conv_row[2] else "2023-01-01T00:00:00Z",
            updated_at=conv_row[3] if conv_row[3] else "2023-01-01T00:00:00Z",
            messages=messages
        )

@app.put("/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation(conversation_id: str, conversation: ConversationCreate):
    """Update a conversation (e.g., title)"""
    async with aiosqlite.connect("checkpoints.db") as conn:
        # Check if conversation exists
        cursor = await conn.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        row = await cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        await conn.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation.title, conversation_id)
        )
        await conn.commit()
        
        # Return updated conversation
        cursor = await conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,)
        )
        row = await cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        return Conversation(
            id=row[0],
            title=row[1],
            created_at=row[2] if row[2] else "2023-01-01T00:00:00Z",
            updated_at=row[3] if row[3] else "2023-01-01T00:00:00Z"
        )

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its messages"""
    async with aiosqlite.connect("checkpoints.db") as conn:
        # Check if conversation exists
        cursor = await conn.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        row = await cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Delete messages first (foreign key constraint)
        await conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        # Delete conversation
        await conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        await conn.commit()
        
    return {"message": "Conversation deleted successfully"}

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    # In a real implementation, you would retrieve session data from the database
    # This is a placeholder that returns basic session info
    return {
        "session_id": session_id,
        "created_at": "2023-01-01T00:00:00Z",  # Placeholder
        "document_count": 0  # Placeholder
    }

@app.post("/session")
async def create_session():
    """Create a new session (without creating a conversation record yet)"""
    session_id = str(uuid.uuid4())
    
    # Only create session ID, don't create conversation record yet
    # Conversation will be created when user sends first message
    
    return {"session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)