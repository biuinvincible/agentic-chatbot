"""
RAG Agent for the Agentic Chatbot.
"""
import asyncio
from typing import List, Dict, Optional
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.file_processing.core import retrieve_relevant_chunks, get_vector_store_by_session
from agents.rag_agent.query_expansion import expand_query_for_rag
from utils.logging_config import get_logger
# Import unified state
from agents.state import UnifiedState
import traceback
import sys

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama not available, document reference detection will use main LLM")

# Initialize logger
logger = get_logger(__name__)

# Define the RAG prompt for document analysis
rag_prompt = PromptTemplate.from_template(
    "You are the 'RAG Agent', specialized in analyzing documents using Retrieval Augmented Generation.\n"
    "Your core capabilities are:\n"
    "1. Retrieving relevant information from documents using vector search.\n"
    "2. Synthesizing retrieved information to provide comprehensive document analysis.\n"
    "3. Answering both general and specific questions about document content.\n"
    "4. Summarizing key points from documents of any size.\n\n"
    "Your limitations and protocols:\n"
    "5. Base your answers strictly on the provided document chunks.\n"
    "6. If the relevant information is not in the provided chunks, clearly state that.\n"
    "7. Focus on providing accurate and detailed information from the document content.\n"
    "8. When summarizing, provide a comprehensive overview of the document's main points.\n"
    "9. You have access to a large context window, so you can analyze multiple chunks together for better understanding.\n\n"
    "Supervisor's Task Guidance: {task_guidance}\n\n"
    "Your goal is to provide thorough analysis and answers based on the document content, following the supervisor's specific guidance above.\n\n"
    "User's question: {question}\n\n"
    "Relevant document chunks from '{document_title}':\n{context}\n\n"
    "Based on the document chunks provided, answer the user's question following the supervisor's guidance. "
    "At the end of your response, you MUST cite the sources you used from the provided chunks in a 'Sources:' section, like this:\n"
    "Sources:\n- [Source: document_name.pdf]\n"
    "Take your time to analyze all the provided chunks and synthesize a coherent response. "
    "Always be thorough and accurate in your response."
)

# Define the document reference detection prompt
document_reference_prompt = PromptTemplate.from_template(
    "You are a precise document reference detection system. Identify which document the user is referring to.\n\n"
    "Available documents:\n"
    "{document_list}\n\n"
    "Recent conversation context:\n"
    "{conversation_context}\n\n"
    "User query: \"{question}\"\n\n"
    "Instructions:\n"
    "1. If the user explicitly mentions a document by name, return that exact name\n"
    "2. If the user refers implicitly (e.g., \"the paper\", \"that document\"), determine the most likely one\n"
    "3. If the query relates to the most recent document discussion, return that document\n"
    "4. If no document is referenced, respond with \"None\"\n"
    "5. Respond ONLY with the exact document name or \"None\" - no other text\n\n"
    "Document name or \"None\":"
)

def initialize_detection_llm():
    """Initialize a lightweight LLM specifically for document reference detection"""
    if OLLAMA_AVAILABLE:
        try:
            # Use Google Qwen/Gemini for efficient document reference detection
            from langchain_google_genai import ChatGoogleGenerativeAI
            detection_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.0,  # Deterministic for consistent results
            )
            return detection_llm
        except Exception as e:
            print(f"Failed to initialize gemma3:1b for detection: {e}")
            return None
    else:
        return None

def detect_document_reference(question: str, document_info: List[Dict], detection_llm, conversation_history: List = None) -> Optional[str]:
    """
    Use a lightweight LLM to detect which document the user is referring to.
    
    Args:
        question: The user's current query
        document_info: List of available documents
        detection_llm: Lightweight LLM for detection
        conversation_history: Previous conversation messages for context
        
    Returns:
        Document name if identified, None otherwise
    """
    if not detection_llm or not document_info:
        return None
    
    # Format document list
    doc_names = [doc.get("file_name") for doc in document_info if isinstance(doc, dict) and "file_name" in doc]
    
    if not doc_names:
        return None
    
    # Format document list for the prompt
    document_list_str = "\n".join([f"- {name}" for name in doc_names])
    
    # Extract recent conversation context (last 3 messages)
    conversation_context = ""
    if conversation_history:
        recent_messages = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        for msg in recent_messages:
            if hasattr(msg, 'content'):
                role = "User" if not hasattr(msg, 'name') else (msg.name or "Assistant")
                conversation_context += f"{role}: {msg.content}\n"
    
    # Create the chain
    chain = document_reference_prompt | detection_llm | StrOutputParser()
    
    # Invoke the chain
    try:
        result = chain.invoke({
            "document_list": document_list_str,
            "conversation_context": conversation_context,
            "question": question
        })
        
        # Clean up the result
        result = result.strip().strip('"\'')
        
        # Validate result
        if result.lower() != "none" and result in doc_names:
            return result
        else:
            return None
    except Exception as e:
        print(f"Error in document reference detection: {e}")
        return None

async def rag_agent(state: UnifiedState, llm, extract_document_reference: bool = False) -> UnifiedState:
    """
    RAG Agent that analyzes documents using Retrieval Augmented Generation.
    
    Args:
        state (UnifiedState): The current state of the conversation, including session_id and document_info.
        llm: The language model to use for analysis.
        extract_document_reference (bool): Whether to use document reference extraction.
        
    Returns:
        UnifiedState: Updated state with the agent's response.
    """
    # Log important fields from state instead of the entire state
    messages_count = len(state.get("messages", []))
    session_id = state.get("session_id", "unknown")
    document_count = len(state.get("document_info", []))
    rag_task = state.get("rag_task")

    logger.info(f"RAG Agent activated - Messages: {messages_count}, Session: {session_id}, Documents: {document_count}")
    if rag_task:
        task_type = rag_task.task_type if hasattr(rag_task, 'task_type') else rag_task
        logger.info(f"RAG Task: {task_type}")

    # Get the supervisor's task guidance
    current_task = state.get("current_task", {})
    task_description = current_task.get("description", "")
    task_guidance = f"Task: {task_description}" if task_description else "No specific task guidance provided"

    # Get the last message (user's question)
    last_message = state["messages"][-1]
    question = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # Get the last 6 messages for context (excluding the current message)
    conversation_context_messages = state["messages"][-7:-1]  # Last 6 messages excluding current
    logger.info(f"User question: {question}")
    
    # Get document information from state
    document_info_list = state.get("document_info", [])
    logger.info(f"Document info count: {len(document_info_list)}")
    
    # Get session ID from state
    session_id = state.get("session_id", "")
    logger.info(f"Session ID: {session_id}")
    
    # Get the RAG task from the state
    rag_task = state.get("rag_task")

    # Initialize context
    context = ""
    
    # Get the latest document info, with a fallback
    # Fix for "string indices must be integers" error - ensure document_info is a list
    document_info = state.get("document_info", [{}])
    if isinstance(document_info, str):
        # If document_info is a string, it might have been serialized incorrectly
        # Try to parse it as JSON, or use a default value
        try:
            import json
            document_info = json.loads(document_info)
        except (json.JSONDecodeError, Exception):
            # If parsing fails, use a default value
            document_info = [{}]
    
    # Ensure document_info is a list
    if not isinstance(document_info, list):
        document_info = [document_info]
    
    # Initialize target document
    target_document = None
    last_referenced_document = state.get("last_referenced_document")
    
    # Only perform document reference extraction if explicitly requested and multiple documents exist
    if extract_document_reference and len(document_info_list) > 1:
        print("Supervisor requested document reference extraction")
        
        # Initialize lightweight detection LLM
        detection_llm = initialize_detection_llm()
        
        if detection_llm:
            # Extract document reference using gemma3:1b or main LLM
            target_document = detect_document_reference(
                question, 
                document_info_list, 
                detection_llm, 
                state.get("messages", [])[-6:] if state.get("messages") else []  # Last 6 messages
            )
            
            if target_document:
                print(f"Detected document reference: {target_document}")
            else:
                print("No document reference detected")
        else:
            print("Failed to initialize detection LLM, using main LLM for detection or fallback")
            
        # Fallback to last referenced document if detection failed
        if not target_document and last_referenced_document:
            if any(isinstance(doc, dict) and doc.get("file_name") == last_referenced_document 
                   for doc in document_info_list):
                target_document = last_referenced_document
                print(f"Using fallback to last referenced document: {target_document}")
    else:
        print("Document reference extraction not requested or not needed")
        # Use last referenced document as fallback when not explicitly extracting
        if last_referenced_document:
            if any(isinstance(doc, dict) and doc.get("file_name") == last_referenced_document 
                   for doc in document_info_list):
                target_document = last_referenced_document
                print(f"Using last referenced document: {target_document}")
    
    # Get the document title for display
    if target_document:
        document_title = target_document
    else:
        # Get the last document info for fallback
        doc_info = document_info[-1] if document_info else {}
        document_title = doc_info.get("file_name", "the uploaded document") if isinstance(doc_info, dict) else "the uploaded document"
    
    # If we have a session ID and document info, try to retrieve the vector store
    if session_id and document_info_list:
        try:
            print(f"Attempting to get vector store for session {session_id}")
            vector_store = get_vector_store_by_session(session_id)
            
            if vector_store:
                print("Retrieved vector store using session ID")
                if rag_task and rag_task.task_type == "summarization":
                    # Summarization Logic (Map-Reduce)
                    print("Summarization request detected.")
                    all_chunks = vector_store.get()["documents"]
                    print(f"Retrieved {len(all_chunks)} chunks for summarization.")
                    
                    # Map step (asynchronous)
                    summary_prompt = PromptTemplate.from_template("Summarize the following text:\n\n{chunk}")
                    summary_chain = summary_prompt | llm
                    
                    print(f"Summarizing {len(all_chunks)} chunks in parallel...")
                    tasks = [summary_chain.ainvoke({"chunk": chunk}) for chunk in all_chunks]
                    results = await asyncio.gather(*tasks)
                    
                    # Handle different result types more robustly
                    chunk_summaries = []
                    for result in results:
                        if hasattr(result, 'content'):
                            chunk_summaries.append(result.content)
                        elif isinstance(result, list) and len(result) > 0:
                            # If result is a list, take the first element
                            first_element = result[0]
                            if hasattr(first_element, 'content'):
                                chunk_summaries.append(first_element.content)
                            else:
                                chunk_summaries.append(str(first_element))
                        else:
                            chunk_summaries.append(str(result))
                            
                    print("Chunk summarization complete.")
                    
                    # Reduce step
                    print("Combining chunk summaries.")
                    combined_summary = "\n".join(chunk_summaries)
                    final_summary_prompt = PromptTemplate.from_template("Combine the following summaries into a single, coherent summary:\n\n{combined_summary}")
                    final_summary_chain = final_summary_prompt | llm
                    final_summary_result = await final_summary_chain.ainvoke({"combined_summary": combined_summary})
                    
                    # Handle different result types more robustly
                    if hasattr(final_summary_result, 'content'):
                        final_summary = final_summary_result.content
                    elif isinstance(final_summary_result, list) and len(final_summary_result) > 0:
                        # If result is a list, take the first element
                        first_element = final_summary_result[0]
                        if hasattr(first_element, 'content'):
                            final_summary = first_element.content
                        else:
                            final_summary = str(first_element)
                    else:
                        final_summary = str(final_summary_result)
                        
                    response_text = f"Summary of {document_title}:\n{final_summary}"
                else:
                    # Question-Answering Logic (Existing RAG)
                    print("--- RAG AGENT: Question-Answering ---")
                    print(f"User question: {question}")
                    
                    # Expand query to improve retrieval
                    print("Expanding query for better retrieval...")
                    expanded_queries = expand_query_for_rag(question, llm)
                    print(f"Expanded queries: {expanded_queries}")
                    
                    # Retrieve chunks for each expanded query and combine results
                    all_docs = []
                    retrieved_chunks = set()  # To avoid duplicate chunks
                    
                    # Use our target_document for filtering
                    doc_filter = {"source": target_document} if target_document else None
                    
                    for i, expanded_query in enumerate(expanded_queries):
                        print(f"Retrieving chunks for expanded query {i+1}: {expanded_query}")
                        docs = retrieve_relevant_chunks(vector_store, expanded_query, filter=doc_filter)
                        for doc in docs:
                            # Use page_content and source as a unique identifier to avoid duplicates
                            doc_id = (doc.page_content, doc.metadata.get('source', ''))
                            if doc_id not in retrieved_chunks:
                                all_docs.append(doc)
                                retrieved_chunks.add(doc_id)
                    
                    print(f"Retrieved {len(all_docs)} unique chunks from {len(expanded_queries)} queries")
                    docs = all_docs
                    
                    print(f"Retrieved {len(docs)} chunks for the question.")

                    context_chunks = []
                    for i, doc in enumerate(docs):
                        source = doc.metadata.get('source', 'Unknown')
                        chunk_text = f"---\nSource: {source}\nContent: {doc.page_content}\n--- END CHUNK ---"
                        context_chunks.append(chunk_text)
                        print(f"Chunk {i+1}: Source='{source}', Length={len(doc.page_content)}")
                    context = "\n\n".join(context_chunks)
                    
                    print(f"Retrieved {len(docs)} chunks")

                    rag_chain = rag_prompt | llm
                    answer_result = await rag_chain.ainvoke({
                        "task_guidance": task_guidance,
                        "question": question,
                        "document_title": document_title,
                        "context": context
                    })
                    
                    # Handle different result types more robustly
                    if hasattr(answer_result, 'content'):
                        response_text = f"RAG Agent Response:\n{answer_result.content}"
                    elif isinstance(answer_result, list) and len(answer_result) > 0:
                        # If result is a list, take the first element
                        first_element = answer_result[0]
                        if hasattr(first_element, 'content'):
                            response_text = f"RAG Agent Response:\n{first_element.content}"
                        else:
                            response_text = f"RAG Agent Response:\n{str(first_element)}"
                    else:
                        response_text = f"RAG Agent Response:\n{str(answer_result)}"
            else:
                response_text = "Could not find vector store for this session."

        except Exception as e:
            response_text = f"Error processing document: {str(e)}"
            print(f"Error processing document: {str(e)}")
            # Let's also print the traceback for more details
            print(f"Full traceback: {traceback.format_exc()}")
    else:
        # If we don't have a session ID or document info, use fallback context
        response_text = "No document context available. Please upload a document first."
        print("Using fallback context")
    
    print(f"RAG Agent response: {response_text}")
    
    # --- NEW: Report Outcome ---
    outcome_info = {
        "agent": "rag_agent",
        "status": "completed",  # Assume success if we reach here. Could refine based on errors.
        "details": "RAG agent completed document analysis successfully"
    }
    # --- END NEW ---
    
    # Add the result to the state and return control to supervisor
    return {
        "messages": state["messages"] + [AIMessage(content=response_text, name="RAGAgent")],
        "last_agent_outcome": outcome_info,  # Add the outcome information
        "next": "supervisor"  # Return control to supervisor
    }
