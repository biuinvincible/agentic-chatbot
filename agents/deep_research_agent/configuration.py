"""Configuration management for the Local Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Local Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=8,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 8,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="google:gemini-1.5-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "google:gemini-1.5-flash",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 100000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="google:gemini-2.0-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "google:gemini-2.0-flash",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=32768,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 32768,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="google:gemini-2.0-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "google:gemini-2.0-flash",
                "description": "Model for research synthesis and compression"
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=32768,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 32768,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="google:gemini-2.5-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "google:gemini-2.5-flash",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=32768,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 32768,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # Rate limiting configuration
    api_request_delay: float = Field(
        default=0.5,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.5,
                "min": 0.1,
                "max": 2.0,
                "step": 0.1,
                "description": "Delay between API requests in seconds to avoid rate limiting"
            }
        }
    )
    max_api_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of retries for API requests that hit rate limits"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )
    # Timeout configuration
    timeout_clarification: int = Field(
        default=60,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 60,
                "min": 10,
                "max": 300,
                "description": "Timeout in seconds for clarification analysis"
            }
        }
    )
    timeout_research_brief: int = Field(
        default=90,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 90,
                "min": 30,
                "max": 300,
                "description": "Timeout in seconds for research brief generation"
            }
        }
    )
    timeout_supervisor: int = Field(
        default=120,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 120,
                "min": 60,
                "max": 600,
                "description": "Timeout in seconds for supervisor decision making"
            }
        }
    )
    timeout_researcher: int = Field(
        default=120,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 120,
                "min": 60,
                "max": 600,
                "description": "Timeout in seconds for researcher decision making"
            }
        }
    )
    timeout_tool_call: int = Field(
        default=90,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 90,
                "min": 30,
                "max": 300,
                "description": "Timeout in seconds for individual tool calls"
            }
        }
    )
    timeout_research_task: int = Field(
        default=180,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 180,
                "min": 60,
                "max": 900,
                "description": "Timeout in seconds for individual research tasks"
            }
        }
    )
    timeout_compression: int = Field(
        default=120,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 120,
                "min": 60,
                "max": 600,
                "description": "Timeout in seconds for research compression"
            }
        }
    )
    timeout_final_report: int = Field(
        default=240,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 240,
                "min": 120,
                "max": 1200,
                "description": "Timeout in seconds for final report generation"
            }
        }
    )

    use_interactive_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to use interactive clarification that allows natural conversation with the user during research"
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        # print(f"[Configuration] Configurable from config: {configurable}")
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        # print(f"[Configuration] Values before processing: {values}")
        # Handle enum fields
        if values.get("search_api") and isinstance(values["search_api"], str):
            try:
                values["search_api"] = SearchAPI(values["search_api"])
            except ValueError:
                # If invalid, use default
                values["search_api"] = SearchAPI.TAVILY
        
        # Handle MCP config
        if values.get("mcp_config") and isinstance(values["mcp_config"], dict):
            values["mcp_config"] = MCPConfig(**values["mcp_config"])
            
        # print(f"[Configuration] Final values: {values}")
        return cls(**{k: v for k, v in values.items() if v is not None})

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Configuration":
        """Create a Configuration instance from a dictionary."""
        # Handle enum fields
        if config_dict.get("search_api") and isinstance(config_dict["search_api"], str):
            try:
                config_dict["search_api"] = SearchAPI(config_dict["search_api"])
            except ValueError:
                # If invalid, use default
                config_dict["search_api"] = SearchAPI.TAVILY
        
        # Handle MCP config
        if config_dict.get("mcp_config") and isinstance(config_dict["mcp_config"], dict):
            config_dict["mcp_config"] = MCPConfig(**config_dict["mcp_config"])
            
        return cls(**{k: v for k, v in config_dict.items() if v is not None})

    def to_dict(self) -> dict:
        """Convert Configuration instance to a dictionary."""
        config_dict = self.model_dump()
        # Convert enum to string
        if isinstance(config_dict.get("search_api"), SearchAPI):
            config_dict["search_api"] = config_dict["search_api"].value
        return config_dict

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True