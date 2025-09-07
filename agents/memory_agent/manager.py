"""
Enhanced Memory Manager for LangMem integration with our agentic assistant.
This version provides direct tool access and better integration with LangGraph.
"""
import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langmem import create_manage_memory_tool, create_search_memory_tool

class MemoryManager:
    """Manages LangMem operations for the agentic assistant with direct tool access."""
    
    def __init__(self, store):
        """Initialize the memory manager with a LangMem store."""
        self.store = store
    
    def get_memory_tools(self, session_id: str):
        """Get LangMem tools for a specific session."""
        try:
            manage_tool = create_manage_memory_tool(
                namespace=("agentic_assistant", session_id, "user_memories")
            )
            search_tool = create_search_memory_tool(
                namespace=("agentic_assistant", session_id, "user_memories")
            )
            return manage_tool, search_tool
        except Exception as e:
            print(f"Error initializing LangMem tools: {e}")
            return None, None
    
    def store_memories(self, memories: List[str], session_id: str) -> bool:
        """Store memories using LangMem tools directly."""
        if not memories or not session_id:
            return False
            
        try:
            manage_tool, _ = self.get_memory_tools(session_id)
            if not manage_tool:
                return False
                
            for memory in memories:
                # Store each memory
                manage_tool.invoke({"memory": memory})
            return True
        except Exception as e:
            print(f"Error storing memories: {e}")
            return False
    
    def search_memories(self, query: str, session_id: str) -> List[str]:
        """Search for relevant memories using LangMem tools."""
        if not query or not session_id:
            return []
            
        try:
            _, search_tool = self.get_memory_tools(session_id)
            if not search_tool:
                return []
                
            # Search for relevant memories
            results = search_tool.invoke({"query": query})
            # Extract memory content from results
            if isinstance(results, list):
                return [str(result) for result in results]
            elif results:
                return [str(results)]
            return []
        except Exception as e:
            print(f"Error searching memories: {e}")
            return []
    
    def extract_memories_from_messages(self, messages: List[BaseMessage], llm) -> List[str]:
        """Extract memories from conversation messages."""
        if not messages:
            return []
        
        # Get recent conversation context
        conversation_context = "\n".join([
            f"{getattr(msg, 'type', 'unknown')}: {getattr(msg, 'content', str(msg))}"
            for msg in messages[-5:]  # Last 5 messages for better context
        ])
        
        # Use LLM to extract memories (simplified approach)
        # In a production system, you'd use a more sophisticated extraction method
        try:
            # This is a simplified extraction - in practice, you'd use a dedicated extraction prompt
            memories = []
            
            # Look for preference statements
            for msg in messages:
                content = getattr(msg, 'content', '')
                if 'prefer' in content.lower() or 'like' in content.lower():
                    memories.append(content)
                elif 'i am' in content.lower() or "i'm" in content.lower():
                    memories.append(content)
                elif 'my name' in content.lower() or 'i work' in content.lower():
                    memories.append(content)
            
            return memories
        except Exception as e:
            print(f"Error extracting memories: {e}")
            return []

# Global memory manager instance
memory_manager = None

def get_memory_manager(store=None):
    """Get or create the global memory manager."""
    global memory_manager
    if memory_manager is None and store is not None:
        memory_manager = MemoryManager(store)
    return memory_manager