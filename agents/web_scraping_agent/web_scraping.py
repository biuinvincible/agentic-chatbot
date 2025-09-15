from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from agents.web_scraping_agent.tools.scraping_tool import get_scraped_content
# Import unified state
from agents.state import UnifiedState
import re

# Define the web scraping prompt (for processing scraped content)
scraping_prompt = PromptTemplate.from_template(
    "You are the 'Web Scraping Agent'.\n"
    "Your core capabilities are:\n"
    "1. Receiving a list of URLs.\n"
    "2. Retrieving the full text content from those URLs.\n"
    "3. Summarizing the key information from the scraped content.\n\n"
    "Your limitations and protocols:\n"
    "- You MUST NOT generate new search queries; that is the Web Search Agent's job.\n"
    "- You should focus on extracting relevant information from the provided URLs.\n"
    "- If content is irrelevant to the original search, do not include it in your summary.\n"
    "- Always provide a comprehensive summary of the scraped content.\n\n"
    "Supervisor's Task Guidance: {task_guidance}\n\n"
    "Your goal is to extract and summarize the most relevant information from the provided URLs, following the supervisor's specific guidance above.\n\n"
    "Original search query: {original_query}\n\n"
    "Scraped content from URLs:\n{scraped_text}\n\n"
    "Provide a detailed summary of the key information found in the scraped content that is relevant to the original query and supervisor's guidance."
)

# Define the web scraping agent
def web_scraping_agent(state: UnifiedState, llm) -> UnifiedState:  # Return UnifiedState
    """
    Web Scraping Agent that scrapes content from URLs and summarizes the information.
    """
    print("[WebScraping] Processing URLs...")
    
    # Get the supervisor's task guidance
    current_task = state.get("current_task", {})
    task_description = current_task.get("description", "")
    task_guidance = f"Task: {task_description}" if task_description else "No specific task guidance provided"
    
    # Get the last message (user's task)
    last_message = state["messages"][-1]
    original_query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # Get the last 6 messages for context (excluding the current message)
    conversation_context_messages = state["messages"][-7:-1]  # Last 6 messages excluding current
    
    # Enhanced URL extraction function that handles trailing punctuation and deduplication
    def extract_urls(text):
        """Extract URLs from text with a reliable approach."""
        if not text:
            return []
        
        # First, find all potential URLs
        potential_urls = re.findall(r'https?://[^\s]+', text)
        
        # Then clean each one by removing trailing punctuation
        cleaned_urls = []
        for url in potential_urls:
            # Remove trailing commas, periods, semicolons, and closing parentheses
            clean_url = re.sub(r'[,.;:\)]+$', '', url)
            # Only add non-empty URLs
            if clean_url:
                cleaned_urls.append(clean_url)
        
        # Deduplicate while preserving order
        seen = set()
        unique_urls = []
        for url in cleaned_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls
    
    # Extract URLs from both the last message and supervisor's task description
    urls_from_query = extract_urls(original_query)
    urls_from_task = extract_urls(task_description) if task_description else []
    
    # Combine and deduplicate URLs
    urls = list(dict.fromkeys(urls_from_query + urls_from_task))  # Preserves order and removes duplicates
    
    # Fallback: Check if it's search results we need to parse
    if not urls:
        # If no URLs found in the message, check if it's search results we need to parse
        if "Search results for" in original_query:
            # Extract URLs from search results format
            urls = re.findall(r'https?://[^\s\)]+', original_query)
    
    if not urls:
        response_text = "Web Scraping Agent Error:\nNo URLs found to scrape. Please provide URLs to scrape."
        print("[WebScraping] No URLs found to scrape")
        return {"messages": state["messages"] + [AIMessage(content=response_text, name="WebScrapingAgent")], "next": "supervisor"}
        
    print(f"[WebScraping] Found {len(urls)} URLs to scrape")
    
    # Scrape content from URLs
    scraped_contents = []
    for url in urls:
        try:
            print(f"[WebScraping] Scraping: {url}")
            # Fix: Pass URL as a list instead of single string
            content = get_scraped_content([url])
            if content:
                print(f"[WebScraping] Successfully retrieved content for {url} (length: {len(content)})")
                if len(content) < 500:
                    print(f"[WebScraping] Warning: Short content for {url}: {content[:200]}")
                scraped_contents.append(f"Content from {url}:\n{content}\n")
            else:
                print(f"[WebScraping] No content returned for {url}")
                scraped_contents.append(f"Failed to scrape content from {url}\n")
        except Exception as e:
            error_msg = f"Error scraping {url}: {str(e)}"
            print(f"[WebScraping] {error_msg}")
            import traceback
            traceback.print_exc()
            scraped_contents.append(f"Error scraping {url}: {str(e)}\n")
    
    # Combine all scraped content
    all_scraped_text = "\n".join(scraped_contents)
    
    # Limit content length to prevent token overflow
    max_content_length = 50000  # Adjust as needed
    if len(all_scraped_text) > max_content_length:
        all_scraped_text = all_scraped_text[:max_content_length] + "... (content truncated)"
    
    # Use an LLM to summarize the scraped content
    if all_scraped_text:
        try:
            print("Summarizing scraped content...")
            # Create a prompt for summarization
            scraping_chain = scraping_prompt | llm
            summary_result = scraping_chain.invoke({
                "task_guidance": task_guidance,
                "original_query": original_query,
                "scraped_text": all_scraped_text
            })
            
            summary_text = summary_result.content if hasattr(summary_result, 'content') else str(summary_result)
            response_text = f"Web Scraping Results:\n{summary_text}"
        except Exception as e:
            print(f"Error during summarization: {e}")
            response_text = f"Web Scraping Results:\nSuccessfully scraped content from {len([c for c in scraped_contents if not c.startswith('Error')])} URLs, but encountered an error during summarization: {str(e)}\n\nRaw scraped content preview:\n{all_scraped_text[:1000]}..."
    else:
        response_text = "Web Scraping Results:\nFailed to scrape content from any of the provided URLs."
    
    print("[WebScraping] Completed scraping and summarization")
    
    # --- NEW: Report Outcome ---
    urls_scraped = len([c for c in scraped_contents if not c.startswith('Error')])
    outcome_info = {
        "agent": "web_scraping_agent",
        "status": "completed",  # Assume success if we reach here. Could refine based on errors.
        "details": f"Scraped content from {urls_scraped} URLs.",
        "urls_scraped": urls_scraped
    }
    # --- END NEW ---
    
    # Add the result to the state
    # Return a dict that updates both messages and last_agent_outcome
    return {
        "messages": state["messages"] + [AIMessage(content=response_text, name="WebScrapingAgent")],
        "last_agent_outcome": outcome_info,  # Add the outcome information
        "next": "supervisor"  # Return control to supervisor
        # Task update is handled by the supervisor based on this outcome.
    }