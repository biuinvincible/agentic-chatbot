"""
Local Deep Research Agent Package
"""

from .deep_researcher import local_deep_researcher
from .progress_wrapper import progress_tracking_researcher

# Default configuration values
DEFAULT_CONFIG = {
    "max_structured_output_retries": 3,
    "allow_clarification": True,
    "max_concurrent_research_units": 2,  # Reduced from 3 to further reduce quota usage
    "max_researcher_iterations": 5,
    "max_react_tool_calls": 8,
    "summarization_model": "google:gemini-1.5-flash",
    "summarization_model_max_tokens": 8192,
    "max_content_length": 50000,
    "research_model": "google:gemini-2.0-flash",
    "research_model_max_tokens": 32768,  # Increased from 8192 to utilize the higher output limit of gemini-2.0-flash
    "compression_model": "google:gemini-2.0-flash",
    "compression_model_max_tokens": 32768,  # Increased from 8192 to utilize the higher output limit of gemini-2.0-flash
    "final_report_model": "google:gemini-2.5-flash",
    "final_report_model_max_tokens": 32768,  # Increased from 4096 to 32768 to utilize the higher output limit of gemini-2.5-flash
    "search_api": "tavily",
    "api_request_delay": 1.0,  # Increased delay to help with rate limiting
    "max_api_retries": 5,  # Increased retries for rate limit handling
    "timeout_clarification": 1800,     # 30 minutes (increased from 60)
    "timeout_research_brief": 1800,    # 30 minutes (increased from 120)
    "timeout_supervisor": 1800,        # 30 minutes (increased from 300)
    "timeout_researcher": 1800,        # 30 minutes (increased from 300)
    "timeout_tool_call": 1800,         # 30 minutes (increased from 120)
    "timeout_research_task": 1800,     # 30 minutes (increased from 300)
    "timeout_compression": 1800,       # 30 minutes (increased from 240)
    "timeout_final_report": 1800       # 30 minutes (2 hours) (increased from 600)
}

def get_default_config():
    """Get the default configuration for the Local Deep Research Agent."""
    return DEFAULT_CONFIG.copy()

def merge_config(base_config, override_config):
    """Merge two configuration dictionaries, with override_config taking precedence."""
    merged = base_config.copy()
    merged.update(override_config)
    return merged

__all__ = [
    "local_deep_researcher",
    "progress_tracking_researcher",
    "get_default_config",
    "merge_config"
]