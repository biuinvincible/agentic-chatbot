"""Deep Researcher Agent Supervisor for our core system."""

import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph

# Import the local deep research implementation
from agents.deep_research_agent.deep_researcher import local_deep_researcher
from agents.deep_research_agent.progress_wrapper import progress_tracking_researcher

def get_today_str():
    """Get today's date as a string."""
    return datetime.now().strftime("%Y-%m-%d")

###################
# Main Research Agent
###################

async def deep_researcher_agent(state: dict, llm=None) -> dict:
    """Main entry point for the local deep research agent."""
    try:
        print("=== DEEP RESEARCHER AGENT: Local Deep Research Implementation ===")
        
        # Prepare input for the local deep research agent
        input_messages = state.get("messages", [])
        session_id = state.get("session_id", f"session_{int(datetime.now().timestamp())}")
        
        # Prepare configuration
        config = {
            "configurable": {
                "thread_id": session_id
            }
        }
        
        # Prepare input for local_deep_research
        research_input = {
            "messages": input_messages
        }
        
        # Add document_info if present
        if "document_info" in state:
            research_input["document_info"] = state["document_info"]
        
        # Run the local deep research agent with progress tracking
        result = await progress_tracking_researcher.ainvoke(research_input, config)
        
        # Extract the final report
        final_report = result.get("final_report", "Research completed.")
        final_message = AIMessage(content=final_report, name="DeepResearcherAgent")
        
        print("=== DEEP RESEARCHER COMPLETED ===")
        
        # Report outcome for supervisor routing
        outcome_info = {
            "agent": "deep_researcher_agent",
            "status": "completed",
            "details": "Local deep research completed successfully"
        }
        
        return {
            "messages": input_messages + [final_message],
            "next": "END",
            "last_agent_outcome": outcome_info,
            "final_report": final_report
        }
        
    except Exception as e:
        print(f"Error in deep research: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"Error during deep research: {str(e)}"
        
        # Report outcome for supervisor routing
        outcome_info = {
            "agent": "deep_researcher_agent",
            "status": "failed",
            "details": str(e)
        }
        
        return {
            "messages": state.get("messages", []) + [AIMessage(content=error_msg, name="DeepResearcherAgent")],
            "next": "END",
            "last_agent_outcome": outcome_info
        }

def create_deep_researcher_graph(llm=None, final_response_llm=None) -> StateGraph:
    """
    Create the LangGraph workflow for the local deep research agent.
    """
    print("=== DEEP RESEARCHER: Creating Local Deep Research Graph ===")
    
    workflow = StateGraph(dict)
    
    async def _deep_researcher_node(state: dict):
        return await deep_researcher_agent(state, llm)
    
    workflow.add_node("deep_researcher_agent", _deep_researcher_node)
    workflow.set_entry_point("deep_researcher_agent")
    workflow.add_edge("deep_researcher_agent", "__end__")
    
    return workflow