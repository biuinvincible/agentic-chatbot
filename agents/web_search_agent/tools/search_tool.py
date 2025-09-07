import os
from ddgs import DDGS
from langchain_core.tools import Tool

# Try to import Tavily search tool
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("Tavily not available, using DuckDuckGo as primary search engine")

def get_search_tool():
    """Define the web search tool with Tavily as primary and DuckDuckGo as fallback"""
    
    def tavily_search(query: str) -> list:
        """Search Tavily for the given query and return results"""
        try:
            # Get Tavily API key from environment
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("TAVILY_API_KEY environment variable not set")
            
            client = TavilyClient(api_key=api_key)
            response = client.search(query, max_results=3)
            
            # Convert to the format expected by the web search agent
            formatted_results = []
            if "results" in response:
                for result in response["results"]:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", result.get("snippet", ""))
                    })
            return formatted_results
        except Exception as e:
            print(f"Error searching Tavily: {e}")
            return []
    
    def tavily_extract_content(urls: list) -> dict:
        """Extract content from URLs including PDFs using Tavily's extract API"""
        try:
            # Get Tavily API key from environment
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("TAVILY_API_KEY environment variable not set")
            
            client = TavilyClient(api_key=api_key)
            response = client.extract(urls=urls, format="text")
            return response
        except Exception as e:
            print(f"Error extracting content with Tavily: {e}")
            return {}
    
    def duckduckgo_search(query: str) -> list:
        """Search DuckDuckGo for the given query and return results"""
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=2)
                # Convert to the format expected by the web search agent
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "content": result.get("body", "")
                    })
                return formatted_results
        except Exception as e:
            print(f"Error searching DuckDuckGo: {e}")
            return []
    
    def fallback_search(query: str) -> list:
        """Search with Tavily as primary, fallback to DuckDuckGo"""
        # Try Tavily first if available
        if TAVILY_AVAILABLE:
            tavily_results = tavily_search(query)
            if tavily_results:
                print(f"Using Tavily search results for query: {query}")
                return tavily_results
            else:
                print(f"Tavily search failed or returned no results for: {query}, falling back to DuckDuckGo")
        
        # Fallback to DuckDuckGo
        ddg_results = duckduckgo_search(query)
        if ddg_results:
            print(f"Using DuckDuckGo search results for query: {query}")
        else:
            print(f"DuckDuckGo search also failed for: {query}")
        
        return ddg_results
    
    # Add the extract function to the tool for PDF/content extraction
    fallback_search.extract_content = tavily_extract_content
    
    return Tool(
        name="Web Search",
        func=fallback_search,
        description="Useful for when you need to search the web for information. Uses Tavily as primary search engine with DuckDuckGo as fallback. Can also extract content from URLs including PDFs."
    )