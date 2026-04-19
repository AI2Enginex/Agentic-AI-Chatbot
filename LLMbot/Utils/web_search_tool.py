import requests
from typing import Optional, List, Dict
import os
import re
from dotenv import load_dotenv
from tavily import TavilyClient
from ddgs import DDGS

load_dotenv()

class WebSearchTool:
    """
    Web search tool for retrieving information when LLM is unable to provide answer
    Supports multiple search providers: Tavily (recommended), Google Search API, or DuckDuckGo
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = None):
        """
        Initialize web search tool
        
        Args:
            api_key: API key for the search provider
            provider: Search provider to use ("tavily", "google", or "duckduckgo")
                     If None, will auto-select based on available API keys
        """
        self.tavily_key = api_key or os.getenv("TAVILY_API_KEY")
        self.google_key = os.getenv("SERPAPI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        # Auto-select provider based on available keys
        if provider:
            self.provider = provider
        elif self.tavily_key:
            self.provider = "tavily"
            print(f"Using Tavily as search provider")
        elif self.google_key:
            self.provider = "google"
            print(f"Using Google Search as search provider")
        else:
            self.provider = "duckduckgo"
            print(f"Using DuckDuckGo as search provider (free, no API key)")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML tags, URLs, markdown links, and unwanted content
        """
        if not text:
            return ""
        
        # Remove HTML tags like <img>, <video>, <a href>, etc.
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove markdown links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove URLs and hyperlinks
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        
        # Remove common image/embed references
        text = re.sub(r'\[image\]|\[video\]|\[ad\]|\[embed\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\(image\)|\(video\)|\(ad\)', '', text, flags=re.IGNORECASE)
        
        # Remove mailto: and ftp: links
        text = re.sub(r'mailto:[^\s]+|ftp://[^\s]+', '', text)
        
        # Remove multiple spaces and clean up
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
        
    def _tavily_search(self, query: str, num_results: int):
        """
        Search using Tavily API (recommended - most reliable)
        Retrieves full text content summaries, filters out images and ads
        """
        results = list()
        unwanted_keywords = ['image', 'photo', 'picture', 'gallery', 'pinterest', 'instagram', 'ad', 'advertisement', 'sponsored']
        try:
            if not self.tavily_key:
                print("Tavily API key not found in TAVILY_API_KEY environment variable")
                return []
                
            client = TavilyClient(api_key=self.tavily_key)
            response = client.search(
                query, 
                max_results=num_results,
                include_answer=True,
                include_raw_content=True
            )
            
            for result in response.get('results', []):
                title = result.get('title', '').lower()
                url = result.get('url', '').lower()
                
                if any(keyword in title or keyword in url for keyword in unwanted_keywords):
                    continue
                
                content = result.get('raw_content', '')
                if not content:
                    content = result.get('content', '')
                if not content:
                    content = result.get('summary', '')
                if not content:
                    content = result.get('snippet', '')
                
                # Clean the content aggressively
                content = self._clean_text(content)
                
                if content:
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'snippet': content
                    })
                    if len(results) >= num_results:
                        break
            
            print(f"Tavily returned {len(results)} text results (filtered)")
            return results
            
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []
    
    
    def _duckduckgo_search(self, query: str, num_results: int):
        """
        Search using DuckDuckGo (free, no API key needed)
        Install: pip install ddgs (or pip install duckduckgo-search for older version)
        Filters out image results and unwanted content
        """
        try:
            ddgs = DDGS()
            results = list()
            unwanted_keywords = ['image', 'photo', 'picture', 'gallery', 'pinterest', 'instagram', 'ad', 'advertisement', 'sponsored']
            
            try:
                search_results = list(ddgs.text(query, max_results=num_results*2, timelimit='y'))
            except Exception as search_error:
                print(f"DuckDuckGo query error: {search_error}")
                print(f"Retrying with different parameters...")
                try:
                    search_results = list(ddgs.text(query, max_results=num_results*2, backend='lite'))
                except:
                    search_results = []
            
            if not search_results:
                print(f"DuckDuckGo returned no results for query")
                return []
            
            for result in search_results:
                if isinstance(result, dict):
                    title = result.get('title', '').lower()
                    url = result.get('href', '') or result.get('link', '')
                    snippet = result.get('body', '') or result.get('snippet', '')
                    
                    if any(keyword in title or keyword in url.lower() for keyword in unwanted_keywords):
                        continue
                    
                    # Clean the snippet aggressively
                    snippet = self._clean_text(snippet)
                    
                    if title and url and snippet and len(snippet.strip()) > 50:
                        results.append({
                            'title': result.get('title', ''),
                            'url': url,
                            'snippet': snippet
                        })
                        if len(results) >= num_results:
                            break
            
            print(f"DuckDuckGo returned {len(results)} text results (filtered)")
            return results
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def search(self, query: str, num_results: int):
        """
        Perform web search and return results
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of dictionaries with 'title', 'url', 'snippet'
        """
        print(f"Searching with {self.provider}...")
        try:
            # Try with selected provider
            if self.provider == "tavily" and self.tavily_key:
                results = self._tavily_search(query, num_results)
                if results:
                    return results
            elif self.provider == "google" and self.google_key:
                results = self._google_search(query, num_results)
                if results:
                    return results
            
            # Fallback to DuckDuckGo
            print(f"Primary search failed or returned empty. Trying DuckDuckGo fallback...")
            results = self._duckduckgo_search(query, num_results)
            return results if results else []
            
        except Exception as e:
            print(f"Error in web search ({self.provider}): {e}")
            # Try DuckDuckGo as last resort
            try:
                print(f"Attempting DuckDuckGo as final fallback...")
                results = self._duckduckgo_search(query, num_results)
                return results if results else []
            except Exception as fallback_e:
                print(f"All search providers failed: {fallback_e}")
                return []
    
    def format_search_results(self, results: List[Dict[str, str]]):
        """
        Format search results as a readable string with full content
        Text-only results, cleaned of unwanted markup, images, and hyperlinks
        """
        if not results:
            return "Web Search: No results found. Please try a different query or refine your search terms."
        
        formatted = "\nWeb Search Results:\n"
        
        
        for idx, result in enumerate(results, 1):
            title = result.get('title', 'N/A')
            url = result.get('url', 'N/A')
            snippet = result.get('snippet', 'N/A')
            
            # Apply additional cleaning to ensure no artifacts remain
            snippet = self._clean_text(snippet)
            
            # Trim very long snippets to reasonable length while preserving complete sentences
            if len(snippet) > 800:
                snippet = snippet[:2000]
                # Try to cut at sentence boundary
                last_period = snippet.rfind('.')
                if last_period > 1500:
                    snippet = snippet[:last_period+1]
            
            snippet = snippet.strip()
            
            formatted += f"\n{idx}. {title}\n"
            formatted += f"\nURL: {url}\n"
            formatted += f"\nSummary: {snippet}\n"
            
        
        return formatted
    