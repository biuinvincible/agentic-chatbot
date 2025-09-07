from crawl4ai import AsyncWebCrawler
import asyncio
import re

async def _scrape_in_parallel(urls: list) -> str:
    """Helper to scrape multiple URLs in parallel."""
    scraped_contents = []
    try:
        print(f"[Crawl4AI] Initializing AsyncWebCrawler")
        # Add timeout configuration to prevent hanging
        from crawl4ai import CrawlerRunConfig, BrowserConfig
        # Use default configuration to avoid parameter conflicts
        browser_config = BrowserConfig()
        async with AsyncWebCrawler(config=browser_config) as crawler:
            print(f"[Crawl4AI] Browser initialized successfully")
            # Set a reasonable timeout (30 seconds) to prevent indefinite hanging
            from crawl4ai.async_configs import CacheMode
            config = CrawlerRunConfig(
                page_timeout=30000,  # 30000 milliseconds = 30 seconds timeout
                cache_mode=CacheMode.BYPASS,  # Bypass cache to get fresh content
            )
            print(f"[Crawl4AI] Scraping {len(urls)} URLs: {urls}")
            results = await crawler.arun_many(urls, config=config)
            print(f"[Crawl4AI] Completed scraping, got {len(results)} results")
            for i, result in enumerate(results):
                print(f"[Crawl4AI] Processing result {i+1}/{len(results)}")
                if result.success:
                    print(f"[Crawl4AI] Successfully scraped {result.url}")
                    # Check if the content seems valid (not a blocked page)
                    content = result.markdown or ""
                    print(f"[Crawl4AI] Content length: {len(content)}")
                    if content:
                        print(f"[Crawl4AI] Content preview: {content[:200]}")
                    # Look for common signs of blocking
                    if _is_content_blocked(content, result.url):
                        error_message = f"Content from {result.url} appears to be blocked or restricted (IP blocking suspected)"
                        print(f"[Crawl4AI] {error_message}")
                        print(f"[Crawl4AI] Blocked content full: {content}")
                        scraped_contents.append(f"{error_message}\n---\n")
                    else:
                        scraped_contents.append(f"Content from {result.url}:\n{content}\n---\n")
                else:
                    error_message = f"Error scraping {result.url}: {result.error_message}"
                    print(f"[Crawl4AI] {error_message}")
                    # Enhance error messages with more specific information
                    if "403" in result.error_message:
                        error_message += " (Likely IP blocking or access forbidden)"
                    elif "429" in result.error_message:
                        error_message += " (Rate limit exceeded)"
                    elif "timeout" in result.error_message.lower():
                        error_message += " (Request timeout - possible IP blocking)"
                    elif "connection refused" in result.error_message.lower():
                        error_message += " (Connection refused - website may be down or blocking access)"
                    elif "net::err" in result.error_message.lower():
                        error_message += " (Network error - website may be blocking access)"
                    scraped_contents.append(f"{error_message}\n---\n")
    except asyncio.TimeoutError:
        error_message = f"Scraping operation timed out after 30 seconds for URLs: {', '.join(urls)}"
        print(error_message)
        scraped_contents.append(f"{error_message}\n---\n")
    except Exception as e:
        error_message = f"Unexpected error during scraping: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        scraped_contents.append(f"{error_message}\n---\n")
    return "\n".join(scraped_contents)

def _is_content_blocked(content: str, url: str) -> bool:
    """Check if the scraped content appears to be blocked."""
    # Common indicators of IP blocking
    blocked_indicators = [
        "access denied",
        "blocked",
        "forbidden",
        "captcha",
        "robot",
        "bot detected",
        "rate limit",
        "too many requests",
        "cloudflare",
        "security check",
        "please enable cookies",
        "javascript required",
        "connection refused",
        "err_connection_refused",
        "err_network_changed"
    ]
    
    # Convert content to lowercase for checking
    content_lower = content.lower()
    
    # Check for blocked indicators
    for indicator in blocked_indicators:
        if indicator in content_lower:
            return True
    
    # Check for extremely short content (often a sign of blocking)
    if len(content.strip()) < 100:
        # But allow short content for very short pages
        if not any(word in content_lower for word in ["home", "page", "welcome", "index"]):
            return True
    
    return False

def get_scraped_content(urls: list) -> str:
    """Get scraped content from a list of URLs in parallel."""
    # Limit to first 5 URLs to avoid overload
    urls_to_scrape = urls[:5]
    if not urls_to_scrape:
        return ""
    
    print(f"[WebScraping] Attempting to scrape URLs: {urls_to_scrape}")
    
    # Use a separate thread to run the async scraping to avoid event loop conflicts
    import concurrent.futures
    import threading
    
    def run_async_scraping():
        try:
            print(f"[WebScraping] Running async scraping in thread {threading.current_thread().name}")
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(_scrape_in_parallel(urls_to_scrape))
                print(f"[WebScraping] Async scraping completed in thread {threading.current_thread().name}")
                return result
            finally:
                loop.close()
        except Exception as e:
            error_msg = f"Error during async scraping in thread: {str(e)}"
            print(f"[WebScraping] {error_msg}")
            import traceback
            traceback.print_exc()
            return f"Error: {error_msg}"
    
    # Run the scraping in a separate thread with timeout
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async_scraping)
            result = future.result(timeout=60)  # 60 second timeout
            print(f"[WebScraping] Successfully scraped content (length: {len(result) if result else 0})")
            if result and len(result) < 500 and result.startswith("Error:"):
                print(f"[WebScraping] Error content returned: {result[:200]}")
            elif result and len(result) < 500:
                print(f"[WebScraping] Warning: Short content returned, preview: {result[:200]}")
            return result
    except concurrent.futures.TimeoutError:
        error_msg = f"Scraping operation timed out after 60 seconds for URLs: {', '.join(urls_to_scrape)}"
        print(f"[WebScraping] {error_msg}")
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Error during scraping: {str(e)}"
        print(f"[WebScraping] {error_msg}")
        import traceback
        traceback.print_exc()
        return f"Error: {error_msg}"
