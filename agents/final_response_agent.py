"""Final Response Agent for the Agentic Assistant."""

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
# Import unified state
from agents.state import UnifiedState

# Define the final response prompt
final_response_prompt = PromptTemplate.from_template(
    "You are the 'Final Response Agent'.\n"
    "Your role is to generate the final polished response to the user's query based on the conversation history.\n\n"
    "Instructions:\n"
    "- Provide a clear, comprehensive, and well-structured response\n"
    "- Synthesize information from previous agent responses if available\n"
    "- Ensure your response directly addresses the user's original query\n"
    "- Use a professional and helpful tone\n"
    "- If there were errors or limitations in previous steps, acknowledge them transparently\n\n"
    "Conversation History:\n{history}\n\n"
    "The user's original query was: '{query}'\n\n"
    "Your final response:\n"
)

async def final_response_agent(state: UnifiedState, llm) -> UnifiedState:
    """
    Final Response Agent that generates the final polished response to the user.
    """
    print("[FinalResponse] Generating final response...")
    
    try:
        # Get the last user message as the query
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
        
        # Get the last human message as the original query
        original_query = ""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == "human":
                original_query = msg.content if hasattr(msg, 'content') else str(msg)
                break
        
        if not original_query:
            # Fallback to the last message
            last_message = messages[-1]
            original_query = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Format conversation history
        conversation_history = "\n".join([
            f"{getattr(msg, 'name', 'User/Assistant')}: {msg.content}" 
            for msg in messages 
            if hasattr(msg, 'content') and msg.content
        ])
        
        # Handle case where there's no prior history
        if not conversation_history:
            conversation_history = "No previous conversation history."
        
        # Limit length to prevent token overflow
        max_history_length = 2000 
        if len(conversation_history) > max_history_length:
            conversation_history = conversation_history[:max_history_length] + "... (history truncated)"
        
        # Generate the final response using the LLM
        response_chain: Runnable = final_response_prompt | llm
        response_result = response_chain.invoke({
            "history": conversation_history,
            "query": original_query
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