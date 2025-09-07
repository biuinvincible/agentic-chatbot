from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any, Optional, List
import json
# Import unified state
from agents.state import UnifiedState

# Define the memory extraction prompt
memory_extraction_prompt = PromptTemplate.from_template(
    "You are the 'Memory Management Agent'.\n"
    "Your task is to extract important information from conversations that should be stored for future reference.\n\n"
    "Guidelines for memory extraction:\n"
    "- Extract user preferences, personal information, and important context\n"
    "- Focus on factual information that would be useful across multiple sessions\n"
    "- Format memories as clear, concise statements\n"
    "- Do NOT store sensitive personal information unless explicitly permitted\n"
    "- If no important information to store, respond with 'NO_MEMORY_NEEDED'\n\n"
    "Supervisor's Task Guidance: {task_guidance}\n\n"
    "Conversation context: {context}\n\n"
    "Extract important memories or respond with 'NO_MEMORY_NEEDED'.\n"
    "Format as a JSON array of memory statements:\n"
    "[\"User prefers detailed explanations\", \"User is a biology student\"]\n"
    "or\n"
    "\"NO_MEMORY_NEEDED\""
)

# Define the memory retrieval prompt
memory_retrieval_prompt = PromptTemplate.from_template(
    "You are the 'Memory Management Agent'.\n"
    "Your task is to determine if there are relevant memories that should be retrieved\n"
    "to help with the current query.\n\n"
    "Supervisor's Task Guidance: {task_guidance}\n\n"
    "Current query: {query}\n\n"
    "If relevant memories exist, respond with a search query to find them.\n"
    "If no relevant memories are needed, respond with 'NO_RETRIEVAL_NEEDED'.\n\n"
    "Response:"
)

def extract_memories_from_conversation(conversation_context: str, llm, task_guidance: str = "") -> list:
    """Extract memories from conversation context using LLM."""
    chain = memory_extraction_prompt | llm
    result = chain.invoke({"task_guidance": task_guidance, "context": conversation_context})
    content = result.content if hasattr(result, 'content') else str(result)
    
    # Try to parse as JSON array
    try:
        if content.strip() == "NO_MEMORY_NEEDED":
            return []
        # Try to parse JSON
        memories = json.loads(content)
        if isinstance(memories, list):
            return memories
        else:
            return [content]  # Treat as single memory string
    except json.JSONDecodeError:
        # If not JSON, treat as single memory or NO_MEMORY_NEEDED
        if "NO_MEMORY_NEEDED" in content:
            return []
        return [content]

def determine_memory_retrieval_need(query: str, llm, task_guidance: str = "") -> Optional[str]:
    """Determine if memory retrieval is needed and what to search for."""
    chain = memory_retrieval_prompt | llm
    result = chain.invoke({"task_guidance": task_guidance, "query": query})
    content = result.content if hasattr(result, 'content') else str(result)
    
    if "NO_RETRIEVAL_NEEDED" in content:
        return None
    return content.strip()

def memory_management_agent(state: UnifiedState, llm, store=None) -> UnifiedState:
    """
    Enhanced Memory Management Agent that integrates with LangGraph's store system.
    
    This agent now:
    1. Extracts important information from conversations
    2. Works with LangGraph's store system for memory persistence
    3. Provides better context to other agents
    
    This implementation follows LangMem best practices by integrating
    with LangGraph's store system rather than using background tasks.
    """
    print("[Memory] Processing memory management task...")
    
    # Get the supervisor's task guidance
    current_task = state.get("current_task", {})
    task_description = current_task.get("description", "")
    task_guidance = f"Task: {task_description}" if task_description else "No specific task guidance provided"
    
    # Get session ID
    session_id = state.get("session_id", "default_session")
    
    # Get the last message (user's task)
    last_message = state["messages"][-1]
    task = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # Get conversation context (last 6 messages)
    conversation_context = "\n".join([
        f"{msg.type if hasattr(msg, 'type') else 'unknown'}: {msg.content if hasattr(msg, 'content') else str(msg)}"
        for msg in state["messages"][-6:]  # Last 6 messages for better context
    ])
    
    # Extract memories to store
    memories_to_store = extract_memories_from_conversation(conversation_context, llm, task_guidance)
    
    # Store memories using LangGraph's store if available
    stored_memories = []
    if store and memories_to_store:
        try:
            # Use LangGraph's store directly for persistence
            # This is more in line with LangMem best practices
            for i, memory in enumerate(memories_to_store):
                # Store each memory with a unique key
                memory_key = f"memory_{i}_{hash(memory) % 10000}"
                store.put(
                    namespace=("agentic_assistant", session_id, "user_memories"),
                    key=memory_key,
                    value={"memory": memory, "timestamp": str(__import__('datetime').datetime.now())}
                )
                stored_memories.append(memory)
                
            print(f"[Memory] Stored {len(stored_memories)} memories for session {session_id}")
        except Exception as e:
            print(f"[Memory] Error storing memories: {e}")
    
    # Determine if we need to retrieve memories
    retrieval_query = determine_memory_retrieval_need(task, llm, task_guidance)
    
    # Retrieve relevant memories if needed and store is available
    retrieved_memories = []
    if store and retrieval_query:
        try:
            # For now, we'll just note that retrieval was attempted
            # A full implementation would use semantic search capabilities
            print(f"[Memory] Retrieval requested for query: {retrieval_query}")
            # In a real implementation with semantic search, you would do:
            # search_results = store.search(...)
        except Exception as e:
            print(f"[Memory] Error retrieving memories: {e}")
    
    # Format the response with memory information
    response_parts = []
    response_parts.append("Memory Management Analysis:")
    
    if stored_memories:
        response_parts.append(f"Memories stored: {stored_memories}")
    else:
        response_parts.append("No new memories stored")
    
    if retrieval_query:
        response_parts.append(f"Memory retrieval requested for query: '{retrieval_query}'")
        response_parts.append("Note: In a full implementation, relevant memories would be retrieved")
    else:
        response_parts.append("No memory retrieval needed for current query")
    
    response_text = "\n".join(response_parts)
    
    # --- NEW: Report Outcome ---
    outcome_info = {
        "agent": "memory_agent",
        "status": "completed",  # Assume success if we reach here. Could refine based on errors.
        "details": "Memory management completed successfully"
    }
    # --- END NEW ---
    
    # Add the result to the state
    response_message = AIMessage(
        content=response_text, 
        name="MemoryAgent"
    )
    
    # --- NEW: Report Outcome ---
    outcome_info = {
        "agent": "memory_agent",
        "status": "completed",  # Assume success if we reach here. Could refine based on errors.
        "details": "Memory management completed successfully"
    }
    # --- END NEW ---
    
    return {
        "messages": state["messages"] + [response_message],
        "last_agent_outcome": outcome_info,  # Add the outcome information
        "next": "supervisor"  # Return control to supervisor
    }