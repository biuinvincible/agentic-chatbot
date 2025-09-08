"""Final Response Agent for the Agentic Assistant."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
# Import unified state
from agents.state import UnifiedState

# Define the final response prompt with focused information
final_response_prompt = PromptTemplate.from_template(
    "You are the 'Final Response Agent'.\n"
    "Your role is to generate the final polished response to the user based on research results.\n\n"
    "Current User Question: {current_user_message}\n\n"
    "Research Results: {research_results}\n\n"
    "Instructions:\n"
    "1. Provide a clear, comprehensive response that directly addresses the user's current question\n"
    "2. Use the provided research results as your source of information\n"
    "3. If the user's question is a modification of a previous request, adjust your response accordingly\n"
    "4. Use a professional and helpful tone\n"
    "5. If the research results are from a deep research agent, they may be comprehensive and detailed\n"
    "6. If the research results indicate no relevant information was found, explain this clearly\n\n"
    "Your response should be comprehensive and well-structured.\n"
    "Your final response:"
)

async def final_response_agent(state: UnifiedState, llm) -> UnifiedState:
    """
    Final Response Agent that generates the final polished response to the user.
    """
    print("[FinalResponse] Generating final response...")
    
    try:
        # Get all messages from state
        messages = state.get("messages", [])
        if not messages:
            error_msg = "No messages found for final response."
            return {
                "messages": [AIMessage(content=error_msg, name="FinalResponseAgent")],
                "next": "END",
                "last_agent_outcome": {
                    "agent": "final_response_agent",
                    "status": "failed",
                    "details": error_msg
                }
            }
        
        # Debug: Print the conversation history being passed
        print("[FinalResponse] Conversation history:")
        for i, msg in enumerate(messages):
            msg_type = "HUMAN" if isinstance(msg, HumanMessage) else "AI" if isinstance(msg, AIMessage) else "OTHER"
            agent_name = getattr(msg, 'name', 'Unknown') if hasattr(msg, 'name') else 'Unknown'
            print(f"  [{i}] {msg_type} ({agent_name}): {msg.content[:100]}...")
        
        # Extract the most recent research results from any agent
        research_results = None
        current_user_message = None
        
        # Look through messages in reverse order to find the most recent ones
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and current_user_message is None:
                # Check if this is a message from any research agent
                agent_name = getattr(msg, 'name', '')
                if agent_name in ['WebSearchAgent', 'DeepResearchAgent', 'RAGAgent', 'WebScrapingAgent'] and research_results is None:
                    research_results = msg.content
                elif agent_name == '' and research_results is None:
                    # Fallback: if no named agent, use the last AI message
                    research_results = msg.content
            elif isinstance(msg, HumanMessage) and current_user_message is None:
                current_user_message = msg.content
        
        # Fallbacks
        if research_results is None:
            research_results = "No recent research results available."
        if current_user_message is None:
            current_user_message = "No current user message found."
        
        print(f"[FinalResponse] Current user message: {current_user_message[:100]}...")
        print(f"[FinalResponse] Most recent research results: {research_results[:100]}...")
        
        # Generate the final response using the LLM with focused context
        response_chain: Runnable = final_response_prompt | llm
        response_result = response_chain.invoke({
            "current_user_message": current_user_message,
            "research_results": research_results
        })
        
        final_response_text = response_result.content if hasattr(response_result, 'content') else str(response_result)
        
        # Debug: Print the generated response
        print(f"[FinalResponse] Generated response: {final_response_text[:200]}...")
        
        print("[FinalResponse] Response generated successfully")
        
        return {
            "messages": messages + [AIMessage(content=final_response_text, name="FinalResponseAgent")],
            "next": "END",
            "last_agent_outcome": {
                "agent": "final_response_agent",
                "status": "completed",
                "details": "Final response generated successfully"
            }
        }
        
    except Exception as e:
        print(f"[FinalResponse] Error: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"Error generating final response: {str(e)}"
        
        return {
            "messages": messages + [AIMessage(content=error_msg, name="FinalResponseAgent")],
            "next": "END",
            "last_agent_outcome": {
                "agent": "final_response_agent",
                "status": "failed",
                "details": str(e)
            }
        }