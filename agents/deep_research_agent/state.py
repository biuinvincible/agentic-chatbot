"""Graph state definitions and data structures for the Local Deep Research agent."""

import operator
from typing import Annotated, Optional, Dict, Any

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class ResearchTask(BaseModel):
    """A research task to be assigned to a specialized agent."""
    task_id: str = Field(description="Unique identifier for the task")
    task_description: str = Field(description="Detailed description of the research task")
    required_specialization: str = Field(description="Required agent specialization")
    priority: int = Field(description="Task priority (1-5, 5 being highest)")
    dependencies: list[str] = Field(default_factory=list, description="Task IDs this task depends on")

class ResearchStrategy(BaseModel):
    """Research strategy with task breakdown and specialization plan."""
    research_approach: str = Field(description="Overall research approach")
    task_breakdown: list[ResearchTask] = Field(description="List of research tasks")
    required_specializations: list[str] = Field(description="List of required agent specializations")
    complexity_assessment: str = Field(description="Assessment of research complexity")

class QualityAssessment(BaseModel):
    """Quality assessment of research findings."""
    overall_quality_score: int = Field(description="Overall quality score (1-10)")
    factual_accuracy: str = Field(description="Assessment of factual accuracy")
    source_credibility: str = Field(description="Assessment of source credibility")
    bias_assessment: str = Field(description="Assessment of potential bias")
    completeness: str = Field(description="Assessment of research completeness")
    recommendations: list[str] = Field(description="Recommendations for improvement")


###################
# State Definitions
###################
def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    research_strategy: Optional[ResearchStrategy]
    current_task: Optional[Dict[str, Any]]
    completed_tasks: Annotated[list[Dict[str, Any]], override_reducer] = []
    quality_assessment: Optional[QualityAssessment]

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    research_strategy: Optional[ResearchStrategy]
    current_task: Optional[Dict[str, Any]]
    completed_tasks: Annotated[list[Dict[str, Any]], override_reducer]
    quality_assessment: Optional[QualityAssessment]

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    research_specialization: str

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []