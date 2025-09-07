"""
Wrapper for Local Deep Research Agent with Progress Tracking
"""

import asyncio
from typing import Dict, Any, Callable, Optional
from langchain_core.runnables import RunnableConfig

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
            result = await self.researcher.ainvoke(input, config)
            
            # Send completion update
            if progress_callback:
                progress_callback("✅ Research pipeline completed successfully!")
            
            return result
            
        except Exception as e:
            # Send error update
            if progress_callback:
                progress_callback(f"❌ Research pipeline failed: {str(e)}")
            raise


# Create the progress-tracking wrapper
progress_tracking_researcher = ProgressTrackingResearcher()