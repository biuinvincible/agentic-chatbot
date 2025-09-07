from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import MessagesState
from agents.web_search_agent.tools.search_tool import get_search_tool
from agents.state import UnifiedState

# Define the query expansion prompt
query_expansion_prompt = PromptTemplate.from_template(
    "You are the 'Web Search Agent'.\n"
    "Your core capabilities are:\n"
    "1. Expanding a user's query into multiple, more effective search terms.\n"
    "2. Performing web searches using your integrated search tool.\n"
    "3. Returning a list of relevant search results (URLs and snippets).\n\n"
    "Your limitations and protocols:\n"
    "- You MUST NOT scrape or process the content of URLs; that is the Web Scraping Agent's job.\n"
    "- You should prioritize high-quality, credible sources in your results.\n"
    "- If search results are irrelevant to the ORIGINAL USER QUERY, do not perform more searches. Report findings and state that you couldn't find relevant information.\n"
    "- Always return the results of your searches, even if they are not perfect.\n\n"
    "Supervisor's Task Guidance: {task_guidance}\n\n"
    "Your goal is to maximize the quality and relevance of search results for the user's underlying information need, following the supervisor's specific guidance above.\n\n"
    "Conversation History:\n{history}\n\n"
    "The last user message (which you should expand) is: '{query}'\n\n"
    "Your task is to generate 3 *improved* search queries. These should be relevant to the original topic, seek diverse perspectives or specific details, and aim to yield comprehensive and high-quality search results.\n"
    "Format your response as a numbered list:\n"
    "1. [expanded query 1]\n"
    "2. [expanded query 2]\n"
    "3. [expanded query 3]\n"
    "Expanded queries:\n"
)

# Define the web search agent
def web_search_agent(state: UnifiedState, llm) -> UnifiedState: # Return UnifiedState
    """
    Web Search Agent that expands the query and performs the search.
    """
    print("[WebSearch] Processing query...")
    
    # Get the supervisor's task guidance
    current_task = state.get("current_task", {})
    task_description = current_task.get("description", "")
    task_guidance = f"Task: {task_description}" if task_description else "No specific task guidance provided"
    
    # Get the last message (user's query)
    last_message = state["messages"][-1]
    original_query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # --- NEW: Prepare conversation history for the LLM prompt ---
    # Get the last 6 messages except the last one (the current query)
    # Limit the history to prevent overly long prompts
    conversation_history_messages = state["messages"][-7:-1]  # Last 6 messages excluding current
    # Format history with clear role distinction
    formatted_history = []
    for msg in conversation_history_messages:
        if hasattr(msg, 'content') and msg.content:
            # Determine the role based on message type for better clarity
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            else:
                role = getattr(msg, 'name', 'User/Assistant') or 'User/Assistant'
            formatted_history.append(f"{role}: {msg.content}")
    conversation_history_str = "\n".join(formatted_history)
    # Handle case where there's no prior history
    if not conversation_history_str:
        conversation_history_str = "No previous conversation history."
    # Limit length to prevent token overflow (adjust limit as needed)
    max_history_length = 2000 
    if len(conversation_history_str) > max_history_length:
        conversation_history_str = conversation_history_str[:max_history_length] + "... (history truncated)"
    # --- END NEW ---
    
    # If this is a recursive call, extract the original query from the search results
    if original_query.startswith("Search results for"):
        # Extract the original query from the first line
        lines = original_query.split('\n')
        if lines:
            first_line = lines[0]
            # Extract the query between the quotes
            import re
            match = re.search(r"'([^']+)'", first_line)
            if match:
                original_query = match.group(1)
            else:
                # Fallback to the first line if we can't extract the query
                original_query = first_line
    
    # Get the search tool
    search_tool = get_search_tool()
    
    # Use an LLM for query expansion
    # --- MODIFIED: Pass both history and query to the expansion prompt ---
    expansion_chain = query_expansion_prompt | llm
    expansion_result = expansion_chain.invoke({
        "task_guidance": task_guidance,
        "history": conversation_history_str,
        "query": original_query
    })
    expanded_query_text = expansion_result.content if hasattr(expansion_result, 'content') else str(expansion_result)
    # --- END MODIFIED ---
    
    print(f"[WebSearch] Query expansion completed")
    
    # Parse expanded queries (simple parsing for numbered list)
    expanded_queries = []
    for line in expanded_query_text.split('\n'):
        if line.strip().startswith(('1.', '2.', '3.')):
            query = line.split('.', 1)[1].strip()
            if query:
                expanded_queries.append(query)
                
    # Fallback: if parsing fails, use basic expansion
    if not expanded_queries:
        expanded_queries = [
            original_query,
            f"{original_query} latest",
            f"{original_query} news 2024"
        ]
        print("[WebSearch] Using fallback queries")
    # Add the original query if not already present
    elif original_query not in expanded_queries:
        expanded_queries.insert(0, original_query)
    
    print(f"[WebSearch] Performing searches for {len(expanded_queries)} queries")
    
    # Perform searches for each expanded query
    search_results = []
    for query in expanded_queries:
        try:
            print(f"[WebSearch] Searching: {query}")
            results = search_tool.invoke({"query": query})
            
            # Extract the actual results from the search tool's response
            if isinstance(results, dict) and "results" in results:
                # Tavily search results
                search_results.extend(results["results"])
            elif isinstance(results, list):
                # Google search results or other list format
                search_results.extend(results)
            else:
                # Handle other formats
                search_results.append(results)
        except Exception as e:
            print(f"Error during search for query '{query}': {e}")
            search_results.append(f"Error searching for '{query}': {str(e)}")
    
    # Format the results into a response message
    response_text = f"Search results for '{original_query}' and related queries:\n\n"
    for i, result in enumerate(search_results):
        if isinstance(result, dict) and "url" in result and "content" in result:
            response_text += f"{i+1}. [{result['title']}]({result['url']})\n   {result['content'][:200]}...\n\n"
        else:
            response_text += f"{i+1}. {result}\n\n"
            
    print(f"Web Search Agent response text: {response_text}")
    
    # --- NEW: Report Outcome ---
    urls_found = len([r for r in search_results if isinstance(r, dict) and 'url' in r])
    outcome_info = {
        "agent": "web_search_agent",
        "status": "completed", # Assume success if we reach here. Could refine based on errors.
        "details": f"Found {len(search_results)} results for {len(expanded_queries)} queries.",
        "urls_found": urls_found
    }
    # --- END NEW ---
    
    # --- NEW: Update Task ---
    # Import Task model here to avoid circular imports if needed, 
    # but it's better to pass the updated task state from the supervisor.
    # For now, we assume the supervisor will update the task based on last_agent_outcome.
    # The supervisor's logic Rule 2 will see that 'search' is not in steps_completed and route accordingly.
    # The supervisor's logic Rule 3 will see the outcome and route to scrape.
    # So, the agent doesn't strictly need to update the task itself in this simple flow.
    # However, if we wanted the agent to be more autonomous, it could propose task updates.
    # For this implementation, we rely on the supervisor's explicit logic.
    # --- END NEW ---
    
    # Add the result to the state
    # Return a dict that updates both messages and last_agent_outcome
    return {
        "messages": state["messages"] + [AIMessage(content=response_text, name="WebSearchAgent")],
        "last_agent_outcome": outcome_info, # Add the outcome information
        "next": "supervisor"  # Return control to supervisor
        # Task update is handled by the supervisor based on this outcome.
    }