"""Context management utilities for the Agentic Assistant."""

import asyncio
from typing import List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

# Context compression prompt
context_compression_prompt = PromptTemplate.from_template(
    """You are an intelligent context compressor for a conversational AI system.
    Your task is to compress the following conversation history while preserving the most important information.
    
    Guidelines:
    1. Retain key facts, decisions, and important context
    2. Remove redundant information and trivial exchanges
    3. Keep user preferences and important details
    4. Maintain the logical flow of the conversation
    5. Reduce the token count while preserving meaning
    
    Conversation History:
    {conversation_history}
    
    Compressed Version (under {max_tokens} tokens):
    """
)

async def compress_conversation_history(
    messages: List[BaseMessage], 
    llm,
    max_tokens: int = 2000,
    min_messages_to_compress: int = 6
) -> str:
    """Intelligently compress conversation history based on relevance.
    
    Args:
        messages: List of conversation messages
        llm: Language model for compression
        max_tokens: Maximum tokens for compressed output
        min_messages_to_compress: Minimum number of messages to trigger compression
        
    Returns:
        Compressed conversation history as a string
    """
    # Token-aware enhancement: Adjust compression parameters based on estimated token count
    estimated_tokens = sum(len(getattr(msg, 'content', '')) // 4 for msg in messages)
    
    # Proactive compression for large contexts
    if estimated_tokens > 5000:  # High token count
        # Compress more aggressively
        min_messages_to_compress = max(3, min_messages_to_compress // 2)
        max_tokens = min(1500, max_tokens)  # Reduce token budget
        print(f"[ContextManager] High token count detected ({estimated_tokens}), adjusting compression parameters")
    elif estimated_tokens > 10000:  # Very high token count
        # Even more aggressive compression
        min_messages_to_compress = 2
        max_tokens = 1000
        print(f"[ContextManager] Very high token count detected ({estimated_tokens}), aggressive compression")
    
    # If we don't have enough messages, return them as-is
    if len(messages) < min_messages_to_compress:
        return _format_messages_for_context(messages)
    
    # Convert messages to string format for LLM processing
    conversation_text = _format_messages_for_context(messages)
    
    # Additional safeguard: If conversation text is extremely long, truncate before processing
    if len(conversation_text) > 50000:  # ~12500 tokens
        conversation_text = "[Context Pre-Truncated] " + conversation_text[-45000:]
        print(f"[ContextManager] Extremely long conversation pre-truncated to prevent processing issues")
    
    # Use LLM to compress while preserving important context
    try:
        compression_chain = context_compression_prompt | llm
        compressed_result = await compression_chain.ainvoke({
            "conversation_history": conversation_text,
            "max_tokens": max_tokens
        })
        
        compressed_content = (
            compressed_result.content 
            if hasattr(compressed_result, 'content') 
            else str(compressed_result)
        )
        
        return compressed_content
    except Exception as e:
        # Fallback to simple truncation if compression fails
        print(f"Warning: Context compression failed: {e}. Using fallback method.")
        return _fallback_compression(conversation_text, max_tokens)

def _format_messages_for_context(messages: List[BaseMessage]) -> str:
    """Format messages for context compression."""
    formatted_messages = []
    for msg in messages:
        if hasattr(msg, 'content') and msg.content:
            # Determine role based on message type
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            elif isinstance(msg, SystemMessage):
                role = "System"
            else:
                role = getattr(msg, 'name', 'Unknown') or 'User/Assistant'
            
            formatted_messages.append(f"{role}: {msg.content}")
    
    return "\n".join(formatted_messages)

def _fallback_compression(conversation_text: str, max_tokens: int) -> str:
    """Fallback method for context compression using simple truncation."""
    # Rough estimation: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    
    if len(conversation_text) <= max_chars:
        return conversation_text
    
    # Take the most recent portion that fits within the limit
    # Add a prefix to indicate truncation
    truncated = "[Context Truncated] " + conversation_text[-(max_chars-25):]
    return truncated