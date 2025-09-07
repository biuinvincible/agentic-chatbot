"""Enhanced Supervisor Agent for our agentic assistant system."""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph

# Import the local deep research implementation
from agents.deep_research_agent.deep_researcher import local_deep_researcher
from agents.deep_research_agent.progress_wrapper import progress_tracking_researcher

# Import the local deep researcher implementation
from agents.deep_research_agent.deep_researcher import local_deep_researcher as deep_researcher_local_deep_researcher
from agents.deep_research_agent.progress_wrapper import progress_tracking_researcher as deep_researcher_progress_tracking_researcher

# Import unified state
from agents.state import UnifiedState


class SupervisorRoutingDecision(BaseModel):
    """Structured output model for supervisor routing decisions."""
    next_agent: Literal[
        "web_search_agent",
        "web_scraping_agent", 
        "image_analysis_agent",
        "rag_agent",
        "deep_research_agent",
        "memory_agent",
        "final_response_agent"
    ] = Field(
        ..., 
        description="The next agent to route to in the workflow"
    )
    task: str = Field(
        ..., 
        description="Specific task instruction for the target agent to execute. Should include only the necessary context for this agent to perform its specialized function."
    )
    reasoning: str = Field(
        ..., 
        description="Explanation of why this routing decision was made"
    )

def get_today_str():
    """Get today's date as a string."""
    return datetime.now().strftime("%Y-%m-%d")

###################
# Enhanced Deep Research Agent
###################

async def deep_research_agent(state: dict, llm=None) -> dict:
    """Main entry point for the enhanced local deep research agent."""
    try:
        print("[DeepResearch] Starting enhanced local deep research...")
        
        # Prepare input for the enhanced local deep research agent
        all_input_messages = state.get("messages", [])
        input_messages = all_input_messages[-6:] if len(all_input_messages) > 6 else all_input_messages  # Last 6 messages
        session_id = state.get("session_id", f"session_{int(datetime.now().timestamp())}")
        
        # Prepare configuration
        config = {
            "configurable": {
                "thread_id": session_id
            }
        }
        
        # Prepare input for deep_researcher_local_deep_research
        research_input = {
            "messages": input_messages
        }
        
        # Add document_info if present
        if "document_info" in state:
            research_input["document_info"] = state["document_info"]
        
        # Run the enhanced local deep research agent with progress tracking
        result = await progress_tracking_researcher.ainvoke(research_input, config)
        
        # Extract the final report
        final_report = result.get("final_report", "")
        final_message = AIMessage(content=final_report, name="DeepResearchAgent")
        
        print("[DeepResearch] Research completed successfully")
        
        # Report outcome for supervisor routing
        outcome_info = {
            "agent": "deep_research_agent",
            "status": "completed",
            "details": "Enhanced local deep research completed successfully"
        }
        
        return {
            "messages": input_messages + [final_message],
            "next": "supervisor",
            "last_agent_outcome": outcome_info,
            "final_report": final_report
        }
        
    except Exception as e:
        print(f"[DeepResearch] Error: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"Error during enhanced deep research: {str(e)}"
        
        # Report outcome for supervisor routing
        outcome_info = {
            "agent": "deep_research_agent",
            "status": "failed",
            "details": str(e)
        }
        
        return {
            "messages": state.get("messages", []) + [AIMessage(content=error_msg, name="DeepResearchAgent")],
            "next": "supervisor",
            "last_agent_outcome": outcome_info
        }

###################
# Supervisor Logic
###################

# Enhanced supervisor prompt with detailed agent capabilities and routing logic
SUPERVISOR_PROMPT = """
You are the Supervisor Agent for an advanced AI assistant system. Your role is to analyze user requests and route them to the most appropriate specialized agent. You must carefully consider the query context, conversation history, document information, and agent capabilities to make optimal routing decisions.

AVAILABLE AGENTS AND THEIR CAPABILITIES:

1. web_search_agent
   - Purpose: Performs web searches and returns relevant URLs/snippets, suitable for queries that dont have links/urls
   - Use when: User needs current information from the internet, factual data, news, or general knowledge
   - Examples: "What is the capital of France?", "Latest news about AI", "Find information about climate change"

2. web_scraping_agent
   - Purpose: Extracts and summarizes content from specific URLs
   - Use when: Specific URLs have been provided in the user's query OR found by web_search_agent that need detailed content extraction
   - Examples: "what is this site about? https://example.com", "summarize this page: https://docs.example.com", After web search results are returned and user wants details from specific links

3. image_analysis_agent
   - Purpose: Analyzes images and extracts information
   - Use when: User has uploaded an image or is asking about visual content
   - Examples: "What's in this image?", "Analyze this photo", "Describe what you see"

4. rag_agent
   - Purpose: Retrieves information from uploaded documents using Retrieval Augmented Generation
   - Use when: User has uploaded documents and is asking questions about their content
   - Examples: "Summarize the PDF I uploaded", "What does the document say about X?", "Find information about Y in my file"

5. deep_research_agent
   - Purpose: Conducts comprehensive, multi-faceted research with enhanced synthesis capabilities
   - Use when: Complex, multi-step research tasks requiring deep analysis and synthesis
   - Examples: "Compare these two technologies", "Analyze the impact of social media", "Research the history of quantum computing"

6. memory_agent
   - Purpose: Manages conversation history and long-term memory
   - Use when: User is asking about previous conversations or personal preferences
   - Examples: "What did we discuss last time?", "Remember that I prefer...", "What are my preferences?"

7. final_response_agent
   - Purpose: Generates final polished responses to user queries
   - Use when: All necessary information has been gathered and it's time to provide the final answer
   - Examples: After any agent has completed its task and returned results

ROUTING DECISION FACTORS:
1. Query Complexity: Simple factual questions → web_search_agent; Complex analysis → deep_research_agent
2. Content Type: Text-based queries → web_search_agent/rag_agent; Image-based → image_analysis_agent
3. Document Context: If documents are uploaded and relevant → rag_agent
4. Conversation History: If referring to previous discussions → memory_agent
5. Task Completion: If an agent has completed its task → final_response_agent

WORKFLOW PATTERNS:
- Sequential: web_search_agent → web_scraping_agent → final_response_agent
- Parallel: Multiple agents working on different aspects of a complex query
- Iterative: Deep research requiring multiple steps and refinements

IMPORTANT: If a previous agent has completed its task (indicated in PREVIOUS AGENT OUTCOME), consider whether the gathered information is sufficient to generate a final response. If so, route to final_response_agent.

CRITICAL INSTRUCTION FOR URL DETECTION:
ALWAYS CAREFULLY EXAMINE THE USER'S QUERY FOR URLs THAT BEGIN WITH http:// OR https://
If you detect any URLs in the user's query, you MUST route to the web_scraping_agent
This is the HIGHEST PRIORITY routing rule - URLs in queries always require scraping

TASK CREATION GUIDELINES:
When creating tasks for agents, you should:
1. Provide ONLY the specific information needed for that agent to perform its function
2. Include relevant context but avoid overwhelming the agent with unnecessary history
3. Clearly specify what you want the agent to accomplish
4. Reference specific documents, URLs, or previous findings when relevant
5. Avoid duplicating work that has already been completed

Example good task for web_search_agent: "Search for laptops suitable for computer science students with dedicated GPUs and prices between $2000-$3000"
Example good task for rag_agent: "Based on the uploaded syllabus document, summarize the key topics that will be covered in the first month of the course"

Make your routing decision based on the most appropriate agent for the current step in the workflow. Consider what information is needed and which agent is best equipped to provide it.

You must respond with a structured JSON object containing:
- next_agent: The name of the agent to route to (must be one of the available agents)
- task: A clear, specific instruction for the target agent to execute based on the user's request. Should be focused and include only necessary context.
- reasoning: A clear explanation of why you chose this agent and what task you're assigning
"""

def supervisor_agent(state: UnifiedState, llm) -> dict:
    """Enhanced supervisor agent that primarily relies on LLM-based decision making with minimal essential rules."""
    print("[Supervisor] Analyzing request and routing...")
    
    # ESSENTIAL RULE-BASED CHECKS (cannot be delegated to LLM)
    
    # 1. Check if user has forced a specific agent (system requirement)
    user_forced_agent = state.get("user_forced_agent")
    if user_forced_agent and user_forced_agent != "Auto":
        print(f"[Supervisor] User forced agent: {user_forced_agent}")
        return {"next": user_forced_agent}
    
    # 2. Safety check for empty messages (prevent errors)
    all_messages = state.get("messages", [])
    messages = all_messages[-6:] if len(all_messages) > 6 else all_messages  # Last 6 messages
    if not messages:
        return {"next": "final_response_agent"}
    
    # 3. Check if we've already performed a web search and have results
    last_agent_outcome = state.get("last_agent_outcome", {})
    if (last_agent_outcome.get("agent") == "web_search_agent" and 
        last_agent_outcome.get("status") == "completed"):
        print("[Supervisor] Web search completed, moving to final response")
        return {"next": "final_response_agent"}
    
    # 4. Get the current query
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # PRIMARY LLM-BASED DECISION MAKING
    
    # Check if we've already performed a web search and have results
    last_agent_outcome = state.get("last_agent_outcome", {})
    all_messages = state.get("messages", [])
    messages = all_messages[-6:] if len(all_messages) > 6 else all_messages  # Last 6 messages
    
    # If the last agent was web_search_agent and we have messages, consider moving to final response
    if (last_agent_outcome.get("agent") == "web_search_agent" and 
        last_agent_outcome.get("status") == "completed" and
        len(messages) >= 2):
        print("[Supervisor] Web search completed, moving to final response")
        return {"next": "final_response_agent"}
    
    # Use advanced LLM-based routing with structured output as the primary approach
    try:
        routing_decision = _structured_llm_routing(state, query, llm)
        if routing_decision and hasattr(routing_decision, 'next_agent'):
            # Validate that the LLM returned a valid agent name
            valid_agents = [
                "web_search_agent", "web_scraping_agent", "image_analysis_agent",
                "rag_agent", "deep_research_agent", "memory_agent", "final_response_agent"
            ]
            
            if routing_decision.next_agent in valid_agents:
                print(f"[Supervisor] Routing to -> {routing_decision.next_agent}")
                print(f"[Supervisor] Task: {routing_decision.task}")
                # Return both the next agent and the task in the state update
                return {
                    "next": routing_decision.next_agent,
                    "current_task": {
                        "description": routing_decision.task,
                        "assigned_by": "supervisor",
                        "created_at": __import__('datetime').datetime.now().isoformat()
                    }
                }
            else:
                print(f"[Supervisor] LLM returned invalid agent '{routing_decision.next_agent}', using fallback")
    except Exception as e:
        print(f"[Supervisor] Error in structured LLM routing: {e}")
    
    # MINIMAL RULE-BASED FALLBACK (only for critical situations)
    
    # Fallback to simple rule-based routing only when LLM fails
    fallback_agent = _minimal_fallback_routing(state, query)
    print(f"[Supervisor] Routing to -> {fallback_agent} (fallback)")
    return {"next": fallback_agent}

def _minimal_fallback_routing(state: UnifiedState, query: str) -> str:
    """Minimal fallback routing with only essential rules when LLM routing fails."""
    query_lower = query.lower()
    document_info = state.get("document_info", [])
    
    # Only use essential, high-confidence rules for fallback
    if document_info and any(keyword in query_lower for keyword in ["document", "pdf", "file"]):
        return "rag_agent"
    elif any(keyword in query_lower for keyword in ["image", "picture", "photo"]):
        return "image_analysis_agent"
    elif any(keyword in query_lower for keyword in ["search", "find", "what is", "who is"]):
        return "web_search_agent"
    else:
        # Default to final response for simple queries or when uncertain
        return "final_response_agent"

# Removed _determine_next_agent_after_completion - LLM should handle this logic

# Removed _is_sufficient_for_response - LLM should handle this logic

# Removed _is_document_related_query - LLM should handle this logic

# Removed _is_follow_up_query - LLM should handle this logic

def _structured_llm_routing(state: UnifiedState, query: str, llm) -> Optional[SupervisorRoutingDecision]:
    """Use advanced LLM analysis with structured output for routing decisions."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import PromptTemplate
        
        # Prepare conversation history (last 6 messages)
        all_messages = state.get("messages", [])
        recent_messages = all_messages[-7:-1] if len(all_messages) > 7 else all_messages[:-1] if len(all_messages) > 1 else []
        conversation_history = _format_conversation_history(recent_messages)  # Last 6 messages excluding current query
        
        # Prepare document context
        document_info = state.get("document_info", [])
        document_context = _format_document_context(document_info)
        
        # Prepare agent outcome context
        last_agent_outcome = state.get("last_agent_outcome")
        outcome_context = _format_agent_outcome(last_agent_outcome) if last_agent_outcome else "No previous agent outcomes"
        
        # Create comprehensive routing prompt with full context
        comprehensive_prompt = f"""
{SUPERVISOR_PROMPT}

CONVERSATION HISTORY:
{conversation_history}

DOCUMENT CONTEXT:
{document_context}

PREVIOUS AGENT OUTCOME:
{outcome_context}

CURRENT USER QUERY:
{query}

Based on all the context provided above, determine the most appropriate agent to handle this request.
Consider the flow of conversation, document availability, and previous agent results when making your decision.

Provide your routing decision in a structured JSON format with the next agent and reasoning.
"""
        
        # Initialize routing LLM with structured output
        from langchain_google_genai import ChatGoogleGenerativeAI
        routing_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0
        ).with_structured_output(SupervisorRoutingDecision)
        
        # Create routing chain
        routing_prompt = PromptTemplate.from_template(comprehensive_prompt)
        routing_chain = routing_prompt | routing_llm
        
        # Get structured routing decision
        routing_decision = routing_chain.invoke({})
        
        # Validate the decision
        if routing_decision and hasattr(routing_decision, 'next_agent'):
            # Validate agent name is in our known agents
            valid_agents = [
                "web_search_agent", "web_scraping_agent", "image_analysis_agent",
                "rag_agent", "deep_research_agent", "memory_agent", "final_response_agent"
            ]
            
            if routing_decision.next_agent in valid_agents:
                return routing_decision
        
        return None
        
    except Exception as e:
        print(f"[Supervisor] Error in structured LLM routing: {e}")
        return None

def _format_conversation_history(messages) -> str:
    """Format conversation history for LLM context."""
    if not messages:
        return "No previous conversation history."
    
    formatted_history = []
    for msg in messages:
        if hasattr(msg, 'content') and msg.content:
            role = getattr(msg, 'name', 'User/Assistant') or 'User/Assistant'
            formatted_history.append(f"{role}: {msg.content}")
    
    history_str = "\n".join(formatted_history)
    
    # Limit length to prevent token overflow
    max_history_length = 1500
    if len(history_str) > max_history_length:
        history_str = history_str[:max_history_length] + "... (history truncated)"
    
    return history_str

def _format_document_context(document_info) -> str:
    """Format document context for LLM context."""
    if not document_info:
        return "No documents have been uploaded."
    
    doc_descriptions = []
    for doc in document_info:
        if isinstance(doc, dict):
            name = doc.get("file_name", "Unknown document")
            chunks = doc.get("chunk_count", 0)
            doc_descriptions.append(f"- {name} ({chunks} chunks processed)")
    
    return "Uploaded documents:\n" + "\n".join(doc_descriptions) if doc_descriptions else "No documents available."

def _format_agent_outcome(outcome) -> str:
    """Format agent outcome for LLM context."""
    if not outcome:
        return "No previous agent outcomes"
    
    agent = outcome.get("agent", "Unknown")
    status = outcome.get("status", "Unknown")
    details = outcome.get("details", "No details")
    
    return f"Agent: {agent}\nStatus: {status}\nDetails: {details}"

# Removed _context_aware_fallback_routing - replaced with minimal fallback

def create_agent_graph(llm, final_response_llm=None) -> StateGraph:
    """Create the enhanced LangGraph workflow with sophisticated routing and orchestration."""
    print("=== CREATING ENHANCED AGENT GRAPH WITH SOPHISTICATED ORCHESTRATION ===")
    
    from agents.state import UnifiedState
    workflow = StateGraph(UnifiedState)
    
    # Import all agents
    from agents.web_search_agent.web_search import web_search_agent
    from agents.web_scraping_agent.web_scraping import web_scraping_agent
    from agents.image_analysis_agent.agent import image_analysis_agent
    from agents.rag_agent.agent import rag_agent
    from agents.memory_agent.agent import memory_management_agent
    from agents.final_response_agent import final_response_agent
    
    # Add nodes for each agent
    async def _deep_research_node(state: dict):
        result = await deep_research_agent(state, llm)
        return result
    
    async def _supervisor_node(state: dict):
        result = supervisor_agent(state, llm)
        # supervisor_agent returns a dict directly, so we just return it
        return result
    
    async def _web_search_node(state: dict):
        result = web_search_agent(state, llm)
        return result
    
    async def _web_scraping_node(state: dict):
        result = web_scraping_agent(state, llm)
        return result
    
    async def _image_analysis_node(state: dict):
        result = image_analysis_agent(state, llm)
        return result
    
    async def _rag_node(state: dict):
        result = await rag_agent(state, llm)
        return result
    
    async def _memory_node(state: dict):
        result = memory_management_agent(state, llm)
        return result
    
    async def _final_response_node(state: dict):
        result = await final_response_agent(state, llm)
        return result
    
    # Add all nodes to the workflow
    workflow.add_node("supervisor", _supervisor_node)
    workflow.add_node("web_search_agent", _web_search_node)
    workflow.add_node("web_scraping_agent", _web_scraping_node)
    workflow.add_node("image_analysis_agent", _image_analysis_node)
    workflow.add_node("rag_agent", _rag_node)
    workflow.add_node("deep_research_agent", _deep_research_node)
    workflow.add_node("memory_agent", _memory_node)
    workflow.add_node("final_response_agent", _final_response_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Enhanced conditional edges with more sophisticated routing
    workflow.add_conditional_edges(
        "supervisor",
        _route_based_on_state,
        {
            "web_search_agent": "web_search_agent",
            "web_scraping_agent": "web_scraping_agent",
            "image_analysis_agent": "image_analysis_agent",
            "rag_agent": "rag_agent",
            "deep_research_agent": "deep_research_agent",
            "memory_agent": "memory_agent",
            "final_response_agent": "final_response_agent",
            "END": "__end__",
            "supervisor": "supervisor"  # Self-loop for complex decision making
        }
    )
    
    # Enhanced edges with conditional logic for better workflow control
    workflow.add_conditional_edges(
        "web_search_agent",
        _determine_post_search_action,
        {
            "supervisor": "supervisor",
            "web_scraping_agent": "web_scraping_agent"
        }
    )
    
    workflow.add_edge("web_scraping_agent", "supervisor")
    workflow.add_edge("image_analysis_agent", "supervisor")
    workflow.add_edge("rag_agent", "supervisor")
    workflow.add_edge("memory_agent", "supervisor")
    workflow.add_edge("deep_research_agent", "supervisor")
    workflow.add_edge("final_response_agent", "__end__")
    
    return workflow

def _route_based_on_state(state: UnifiedState) -> str:
    """Enhanced routing function that makes decisions based on comprehensive state analysis."""
    next_agent = state.get("next", "final_response_agent")
    
    # Validate the next agent is a valid option
    valid_agents = [
        "web_search_agent", "web_scraping_agent", "image_analysis_agent",
        "rag_agent", "deep_research_agent", "memory_agent", 
        "final_response_agent", "supervisor", "END"
    ]
    
    if next_agent in valid_agents:
        return next_agent
    else:
        # Fallback routing based on state analysis
        return _fallback_routing(state)

def _determine_post_search_action(state: UnifiedState) -> str:
    """Determine the next action after web search based on results and user intent."""
    last_agent_outcome = state.get("last_agent_outcome", {})
    messages = state.get("messages", [])
    
    # If web search found URLs and user seems to want details, go to scraping
    if (last_agent_outcome.get("agent") == "web_search_agent" and 
        last_agent_outcome.get("urls_found", 0) > 0 and 
        messages and len(messages) >= 2):
        
        # Check if the original query suggests wanting details
        original_query = messages[-2].content if hasattr(messages[-2], 'content') else ""
        query_indicators = ["detail", "extract", "scrape", "more about", "in-depth"]
        
        if any(indicator in original_query.lower() for indicator in query_indicators):
            return "web_scraping_agent"
    
    # Default to returning to supervisor for next decision
    return "supervisor"

def _fallback_routing(state: UnifiedState) -> str:
    """Fallback routing when the specified next agent is invalid."""
    all_messages = state.get("messages", [])
    messages = all_messages[-6:] if len(all_messages) > 6 else all_messages  # Last 6 messages
    document_info = state.get("document_info", [])
    
    if not messages:
        return "final_response_agent"
    
    # Get the last message
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # Simple rule-based fallback
    query_lower = query.lower()
    
    if document_info and any(keyword in query_lower for keyword in ["document", "pdf", "file"]):
        return "rag_agent"
    elif any(keyword in query_lower for keyword in ["image", "picture", "photo"]):
        return "image_analysis_agent"
    elif any(keyword in query_lower for keyword in ["compare", "analyze", "study", "research"]):
        return "deep_research_agent"
    elif any(keyword in query_lower for keyword in ["search", "find", "what is", "who is"]):
        return "web_search_agent"
    else:
        return "final_response_agent"