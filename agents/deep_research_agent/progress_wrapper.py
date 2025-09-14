"""
Wrapper for Local Deep Research Agent with Progress Tracking
"""

import asyncio
from typing import Dict, Any, Callable, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt

from agents.deep_research_agent.deep_researcher import local_deep_researcher


class ProgressTrackingResearcher:
    """Wrapper for local_deep_researcher with progress tracking capabilities"""
    
    def __init__(self):
        self.researcher = local_deep_researcher
    
    async def ainvoke(
        self, 
        input: Dict[str, Any], 
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Invoke the researcher with progress tracking
        
        Args:
            input: Input messages for the researcher
            config: Configuration for the researcher
        """
        # Extract progress callback from config if provided
        progress_callback = None
        if config and "configurable" in config:
            progress_callback = config["configurable"].get("progress_callback")
        
        # Send initial progress update
        if progress_callback:
            progress_callback("🚀 Initializing research pipeline...")
        
        try:
            # Forward to the actual researcher
            print(f"[ProgressTracking] Calling local_deep_researcher.ainvoke with input: {input}")
            result = await self.researcher.ainvoke(input, config)
            print(f"[ProgressTracking] local_deep_researcher.ainvoke returned result: {type(result)}")
            
            # Send completion update
            if progress_callback:
                progress_callback("✅ Research pipeline completed successfully!")
            
            return result
            
        except GraphInterrupt as e:
            print(f"[ProgressTracking] Caught GraphInterrupt: {e}")
            # Send interrupt update
            if progress_callback:
                progress_callback("🔄 Waiting for user input...")
            # Re-raise GraphInterrupt so it can be handled by the calling code
            raise
        except Exception as e:
            print(f"[ProgressTracking] Caught Exception: {e}")
            # Send error update for other exceptions
            if progress_callback:
                progress_callback(f"❌ Research pipeline failed: {str(e)}")
            raise


# Create the progress-tracking wrapper
progress_tracking_researcher = ProgressTrackingResearcher()