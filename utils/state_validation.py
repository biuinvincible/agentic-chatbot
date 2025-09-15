"""State validation utilities for the Agentic Assistant."""

from typing import Dict, Any, List, Optional
from agents.state import UnifiedState

def validate_state(state: UnifiedState) -> Dict[str, Any]:
    """
    Validate the state and return validation results.
    
    Args:
        state: The UnifiedState to validate
        
    Returns:
        Dictionary containing validation results with 'is_valid' and 'errors' keys
    """
    errors = []
    
    # Check required fields
    if not state.get("session_id"):
        errors.append("Missing required field: session_id")
    
    # Validate messages structure
    messages = state.get("messages", [])
    if not isinstance(messages, list):
        errors.append("Messages field must be a list")
    else:
        for i, msg in enumerate(messages):
            if not hasattr(msg, 'content'):
                errors.append(f"Message at index {i} missing content attribute")
    
    # Validate document_info if present
    document_info = state.get("document_info", [])
    if document_info:
        if not isinstance(document_info, list):
            errors.append("document_info must be a list")
        else:
            for i, doc in enumerate(document_info):
                if not isinstance(doc, dict):
                    errors.append(f"Document info at index {i} must be a dictionary")
                elif "file_name" not in doc:
                    errors.append(f"Document info at index {i} missing required field: file_name")
    
    # Validate next field
    next_agent = state.get("next")
    if next_agent and not isinstance(next_agent, str):
        errors.append("next field must be a string")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }

def get_state_summary(state: UnifiedState) -> Dict[str, Any]:
    """
    Get a summary of the state for monitoring purposes.
    
    Args:
        state: The UnifiedState to summarize
        
    Returns:
        Dictionary containing state summary information
    """
    summary = {
        "session_id": state.get("session_id", "unknown"),
        "message_count": len(state.get("messages", [])),
        "document_count": len(state.get("document_info", [])),
        "next_agent": state.get("next", "unknown"),
        "has_research_brief": bool(state.get("research_brief")),
        "note_count": len(state.get("notes", [])),
        "has_interrupt": state.get("interrupt_state", {}).get("active", False) if isinstance(state.get("interrupt_state"), dict) else False
    }
    
    # Estimate state size
    try:
        import json
        state_size = len(json.dumps(state, default=str))
        summary["estimated_size_bytes"] = state_size
    except Exception:
        summary["estimated_size_bytes"] = -1  # Unable to calculate
    
    return summary

def is_state_too_large(state: UnifiedState, max_size_bytes: int = 100000) -> bool:
    """
    Check if the state is too large which might cause performance issues.
    
    Args:
        state: The UnifiedState to check
        max_size_bytes: Maximum size in bytes before considered too large
        
    Returns:
        Boolean indicating if state is too large
    """
    try:
        import json
        state_size = len(json.dumps(state, default=str))
        return state_size > max_size_bytes
    except Exception:
        return False  # Unable to calculate, assume OK