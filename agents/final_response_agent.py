"""Final Response Agent for the Agentic Assistant."""

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
# Import unified state
from agents.state import UnifiedState

# Define the final response prompt with conversation history
final_response_prompt = PromptTemplate.from_template(
    "You are the 'Final Response Agent'.\n"
    "Your role is to generate the final polished response to the user based on the complete conversation.\n\n"
    "Instructions:\n"
    "1. Read the complete conversation history carefully\n"
    "2. Identify what the user is asking RIGHT NOW\n"
    "3. Provide a clear, comprehensive, and well-structured response to the current question\n"
    "4. Use a professional and helpful tone\n\n"
    "Complete Conversation History:\n"
    "{history}\n\n"
    "What is the user asking right now and how should you respond?\n"
    "Your final response:\n"
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
        
        # Format conversation history the same way supervisor does
        formatted_history = []
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                role = getattr(msg, 'name', 'User/Assistant') or 'User/Assistant'
                formatted_history.append(f"{role}: {msg.content}")
        
        conversation_history = "\n".join(formatted_history)
        
        # Handle case where there's no prior history
        if not conversation_history:
            conversation_history = "No previous conversation history."
        
        # Limit length to prevent token overflow (same as supervisor)
        max_history_length = 1500
        if len(conversation_history) > max_history_length:
            conversation_history = conversation_history[:max_history_length] + "... (history truncated)"
        
        # Generate the final response using the LLM with enhanced context
        response_chain: Runnable = final_response_prompt | llm
        response_result = response_chain.invoke({
            "history": conversation_history
        })
        
        final_response_text = response_result.content if hasattr(response_result, 'content') else str(response_result)
        
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