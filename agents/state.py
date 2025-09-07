"""Unified state management for the Agentic Assistant."""

from typing import Annotated, Dict, Any, List, Optional, Union
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

class AgentOutcome(TypedDict):
    """Information about the outcome of an agent's execution."""
    agent: str
    status: str  # "completed", "failed", "routed", etc.
    details: str
    urls_found: Optional[int]

class UnifiedState(MessagesState):
    """Unified state schema for the entire agentic assistant system.
    
    Inherits 'messages' field from MessagesState with add_messages reducer.
    """
    
    # Core routing fields
    next: str  # Next agent to call or "END"
    
    # Document processing
    document_info: Annotated[List[Dict[str, Any]], lambda x, y: y if y is not None else x]
    
    # Session management
    session_id: Optional[str]
    
    # User control
    user_forced_agent: Optional[str]
    
    # Agent outcomes for supervisor routing decisions
    last_agent_outcome: Optional[AgentOutcome]
    
    # Research-specific fields (for deep research agent)
    supervisor_messages: Annotated[List[AnyMessage], lambda x, y: y if y is not None else x]
    research_brief: Optional[str]
    raw_notes: Annotated[List[str], lambda x, y: x + (y if isinstance(y, list) else [y] if y else [])]
    notes: Annotated[List[str], lambda x, y: x + (y if isinstance(y, list) else [y] if y else [])]
    final_report: Optional[str]
    research_strategy: Optional[Dict[str, Any]]
    current_task: Optional[Dict[str, Any]]
    completed_tasks: Annotated[List[Dict[str, Any]], lambda x, y: x + (y if isinstance(y, list) else [y] if y else [])]
    quality_assessment: Optional[Dict[str, Any]]
    
    # RAG-specific fields
    rag_task: Optional[Dict[str, Any]]
    last_referenced_document: Optional[str]