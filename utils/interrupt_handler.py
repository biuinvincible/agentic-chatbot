"""Interrupt handling utilities for the Agentic Assistant."""

from typing import Dict, Any, Optional
from agents.state import InterruptState


def create_interrupt_state(
    interrupt_type: str, 
    message: str, 
    context: Optional[Dict[str, Any]] = None
) -> InterruptState:
    """Create a new interrupt state."""
    return InterruptState(
        active=True,
        type=interrupt_type,
        message=message,
        context=context or {}
    )


def clear_interrupt_state() -> InterruptState:
    """Create a cleared interrupt state."""
    return InterruptState(active=False)


def is_interrupt_active(state: Dict[str, Any]) -> bool:
    """Check if an interrupt is currently active."""
    interrupt_state = state.get("interrupt_state")
    if isinstance(interrupt_state, InterruptState):
        return interrupt_state.active
    elif isinstance(interrupt_state, dict):
        return interrupt_state.get("active", False)
    return False


def get_interrupt_type(state: Dict[str, Any]) -> Optional[str]:
    """Get the type of the current interrupt."""
    interrupt_state = state.get("interrupt_state")
    if isinstance(interrupt_state, InterruptState):
        return interrupt_state.type
    elif isinstance(interrupt_state, dict):
        return interrupt_state.get("type")
    return None


def update_interrupt_state(state: Dict[str, Any], new_interrupt_state: InterruptState) -> Dict[str, Any]:
    """Update the interrupt state in the agent state."""
    updated_state = state.copy()
    updated_state["interrupt_state"] = new_interrupt_state
    return updated_state