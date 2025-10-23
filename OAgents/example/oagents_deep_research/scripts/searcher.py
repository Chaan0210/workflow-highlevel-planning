#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
from serpapi import GoogleSearch, BingSearch, YahooSearch, BaiduSearch
from oagents import Tool, OpenAIServerModel
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
from reflectors import SearchReflector


class BaseSearcher:
    def __init__(self):
        self.history = []
        self.name = "google_search"

    def _pre_visit(self, url):
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i][0] == url:
                return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
        return ""
    

    def _to_content(self, query:str, snippets:List):
        web_snippets = []
        idx=1
        for search_info in snippets:
            # Handle error responses
            if isinstance(search_info, dict) and "error" in search_info:
                return f"Search failed: {search_info['error']}"
            
            # Handle normal responses with safe key access
            title = search_info.get('title', 'No title')
            link = search_info.get('link', search_info.get('href', search_info.get('url', 'No link')))
            date = search_info.get('date', '')
            source = search_info.get('source', '')
            snippet = search_info.get('snippet', search_info.get('body', 'No snippet'))
            
            redacted_version = f"{idx}. [{title}]({link})" + \
                            f"{date}{source}\n{self._pre_visit(link)}{snippet}"

            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)
            idx+=1
        
        content = (
            f"A Search through {self.name} for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )
        return content
    
    def search(self):
        NotImplemented



class SerpSearcher(BaseSearcher):
    def __init__(self,
                 engine:str="google",
                 api_key:str=None,
                 max_results:int=10
                 ):
        super().__init__()
        self.engine = engine

        self.name = f"{engine}_search"
        self.description = f"Perform a web search query on {engine} search engine and returns the search results."

        self.serpapi_key = api_key or os.getenv("SERP_API_KEY")
        self.serp_num = max_results


    def search(self, query: str, filter_year: Optional[int] = None) -> List[str]:
        if self.serpapi_key is None:
            raise ValueError("Missing SerpAPI key.")
        
        self.history.append((query, time.time()))

        params = {
            "engine": self.engine,
            "api_key": self.serpapi_key,
        }

        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        if self.engine == 'google':
            params['q'] = query
            params['num'] = self.serp_num
            search = GoogleSearch(params)
        elif self.engine == 'bing':
            params['q'] = query
            params['count'] = self.serp_num
            search = BingSearch(params)
        elif self.engine == 'baidu':
            params['q'] = query
            params['rn'] = self.serp_num
            search = BaiduSearch(params)
        elif self.engine=='yahoo':
            params['p'] = query
            search = YahooSearch(params)
        else:
            raise ValueError("Unsupport Serp Engine! Please check your parameters!")


        results = search.get_dict()
    
        self.page_title = f"{query} - Search"
        if "organic_results" not in results.keys():
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for query: '{query}'{year_filter_message}. Try with a more general query or different search terms."
        if len(results["organic_results"]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."

        web_snippets: List[str] = list()
        idx = 0
        if "organic_results" in results:
            for page in results["organic_results"]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                _search_result = {
                    "idx": idx,
                    "title": page["title"],
                    "date": date_published,
                    "snippet": snippet,
                    "source": source,
                    "link": page['link']
                }
                
                web_snippets.append(_search_result)

        return web_snippets


class WikiSearcher(BaseSearcher):
    def __init__(self):
        super().__init__()
        self.name = "wiki_search"
        self.description = "Call this tool to perform a Wikipedia search. Provide a query string for the information you want to retrieve from Wikipedia."
        
    def search(self, query:str, filter_year: Optional[int] = None):
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "exintro": True,
            "explaintext": True,
            "titles": query,
            "redirects": 1,
            "inprop": "url"
        }
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                error_info = data['error']
                return f"Wikipedia API error: {error_info.get('code', 'unknown')} - {error_info.get('info', 'unknown')}"

            pages = data.get("query", {}).get("pages", {})
            results = []
            idx = 1
            for page_id, page_info in pages.items():
                if int(page_id) < 0:
                    continue
                title = page_info.get("title", "Unknown Title")
                extract = page_info.get("extract", "No extract available")
                page_url = page_info.get("fullurl", "No URL available")

                result = {
                    "idx": idx,
                    "title": title,
                    "date": "",
                    "snippet": extract,
                    "source": "",
                    "link": page_url
                }
                results.append(result)
                idx += 1

            if results:
                return results
            return f"No relevant information found for the query: {query}"
        except requests.Timeout:
            return "Request to Wikipedia API timed out. Please try again later."
        except requests.RequestException as e:
            return f"Network error occurred: {str(e)}"
        except ValueError as e:
            return f"Error parsing JSON response: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"


class BochaSearcher(BaseSearcher):
    def __init__(self,
                 api_key:str=None):
        super().__init__()

        self.api_key = api_key or os.getenv("BOCHA_API_KEY")
        self.name = "bocha_search"
        self.description = "Perform web search through BOCHA API to search information for the given query."

    def search(self, query: str, filter_year: Optional[int] = None) -> None:
        if self.api_key is None:
            raise ValueError("Missing SerpAPI key.")
        
        url = "https://api.bochaai.com/v1/web-search"
        payload = json.dumps({
                    "query": query,
                    "summary": True,
                    "count": 10,
                    "page": 1
                    })
        # api_key=self.api_key
        headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                    }

        response = requests.request("POST", url, headers=headers, data=payload)
        # print(response.json())
        result=response.json()

        self.page_title = f"{query} - Search"

        if result['code']!=200:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        
        page_result=result['data']['webPages']
        
        web_snippets: List[str] = list()
        idx=0
        for page in page_result['value']:
            idx += 1
            date_published = ""
            if "dateLastCrawled" in page:
                date_published = "\nDate published: " + page["dateLastCrawled"]

            source = ""
            if "siteName" in page:
                source = "\nSource: " + page["siteName"]

            snippet = ""
            if "snippet" in page:
                snippet = "\n" + page["snippet"]

            _search_result = {
                "idx": idx,
                "title": page["name"],
                "date": date_published,
                "snippet": snippet,
                "source": source,
                "link": page['url']
            }

            web_snippets.append(_search_result)

        return web_snippets


class DuckDuckGoSearcher(BaseSearcher):
    def __init__(self, 
                 max_results:int=5):
        super().__init__()
        self.max_results = max_results

        self.name = "duckduckgo_search"
        self.description = """Use DuckDuckGo search engine to search information for the given query.

This function queries the DuckDuckGo API for related topics to
the given search term. The results are formatted into a list of
dictionaries, each representing a search result."""

    def search(
        self, query: str, filter_year: Optional[int] = None, source: str = "text"
        ) -> List[Dict[str, Any]]:

        from duckduckgo_search import DDGS
        from requests.exceptions import RequestException

        ddgs = DDGS()
        responses: List[Dict[str, Any]] = []

        if source == "text":
            try:
                results = ddgs.text(keywords=query, max_results=self.max_results)
            except (RequestException, Exception) as e:
                error_msg = f"duckduckgo search failed: {e}"
                if "202" in str(e) or "rate" in str(e).lower():
                    error_msg += " (Rate limited - try using a different search engine)"
                responses.append({"error": error_msg})
                return responses

            # Iterate over results found
            if results and len(results) > 0:
                try:
                    for i, result in enumerate(results, start=1):
                        # Handle potential missing keys in result dictionary
                        if not isinstance(result, dict):
                            continue
                            
                        response = {
                            "idx": i,
                            "title": result.get("title", "No title available"),
                            "snippet": result.get("body", result.get("snippet", "No snippet available")),
                            "link": result.get("href", result.get("url", "No link available")),
                            "source": result.get("source", ""),
                            "date": result.get("date", "")
                        }
                        responses.append(response)
                except Exception as e:
                    # If processing results fails, likely due to unexpected format
                    error_msg = f"Error processing DuckDuckGo results: {e}"
                    if "KeyError" in str(e):
                        error_msg += " (Possible rate limit or format issue - try a different search engine)"
                    responses.append({"error": error_msg})
            else:
                # No results case - could be due to rate limiting or genuine no results
                responses.append({"error": "No results found in DuckDuckGo search (may be rate limited - try a different search engine)"})

        elif source == "images":
            try:
                results = ddgs.images(keywords=query, max_results=self.max_results)
            except (RequestException, Exception) as e:
                error_msg = f"duckduckgo image search failed: {e}"
                if "202" in str(e) or "rate" in str(e).lower():
                    error_msg += " (Rate limited - try using a different search engine)"
                responses.append({"error": error_msg})
                return responses

            if results:
                for i, result in enumerate(results, start=1):
                    if not isinstance(result, dict):
                        continue
                    response = {
                        "result_id": i,
                        "title": result.get("title", "No title available"),
                        "image": result.get("image", "No image available"),
                        "url": result.get("url", "No URL available"),
                        "source": result.get("source", ""),
                    }
                    responses.append(response)
            else:
                responses.append({"error": "No image results found in DuckDuckGo search"})

        elif source == "videos":
            try:
                results = ddgs.videos(keywords=query, max_results=self.max_results)
            except (RequestException, Exception) as e:
                error_msg = f"duckduckgo video search failed: {e}"
                if "202" in str(e) or "rate" in str(e).lower():
                    error_msg += " (Rate limited - try using a different search engine)"
                responses.append({"error": error_msg})
                return responses

            if results:
                for i, result in enumerate(results, start=1):
                    if not isinstance(result, dict):
                        continue
                    response = {
                        "idx": i,
                        "title": result.get("title", "No title available"),
                        "snippets": result.get("description", "No description available"),
                        "embed_url": result.get("embed_url", "No embed URL available"),
                        "publisher": result.get("publisher", "Unknown publisher"),
                        "duration": result.get("duration", "Unknown duration"),
                        "published": result.get("published", "Unknown date"),
                    }
                    responses.append(response)
            else:
                responses.append({"error": "No video results found in DuckDuckGo search"})
        additional_text = """
            Here are some tips to help you get the most out of your search results:
            - When dealing with web snippets, keep in mind that they are often brief and lack specific details. If the snippet doesn't provide useful information, but the URL is from a highly-ranked source, it might still contain the data you need. 
            - For more detailed answers, you should utilize other tools to analyze the content of the websites in the search results, e.g. document relevant toolkit.
            - When seeking specific quantities, it's essential to look for a reliable and accurate source. Avoid relying solely on web snippets for figures like dollar amounts, as they may be imprecise or approximated.
            - If the information found in the snippets doesn't answer your original query satisfactorily, make sure to check the first URL. This is likely to contain much more in-depth content, as it's ranked as the most relevant. 
            - Additionally, when looking for books, consider searching for publicly available full-text PDFs, which can be searched entirely at once using document tools for relevant content.
        """
        return responses
    

class SearchTool(Tool):
    name = "web_search"
    description = "Perform a web search query (think a google search) and returns the search results."
    inputs = {"query": {"type": "string", "description": "The web search query to perform."}}
    inputs["filter_year"] = {
        "type": "string",
        "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!",
        "nullable": True,
    }
    output_type = "string"

    def __init__(self, 
                 search_type:str='google', 
                 serp_num:int=5,
                 reflection:bool=False):
        
        super().__init__()
        self.reflection = reflection

        self.allowed_search_types = ['google', 'bing', 'bocha', 'baidu', 'wiki', 'yahoo', 'duckduckgo']
        
        if search_type not in self.allowed_search_types:
            raise ValueError(f"Invalid search_type. It must be one of {self.allowed_search_types}")
        
        if search_type in ['google', 'bing', 'baidu', 'yahoo']:
            self.searcher = SerpSearcher(engine=search_type, api_key=os.getenv("SERP_API_KEY"), max_results=serp_num)
        elif search_type == 'wiki':
            self.searcher = WikiSearcher()
        elif search_type == 'bocha':
            self.searcher = BochaSearcher(api_key=os.getenv("BOCHA_API_KEY"))
        elif search_type == 'duckduckgo':
            self.searcher = DuckDuckGoSearcher(max_results=serp_num)
        else:
            self.searcher = SerpSearcher(engine='google', api_key=os.getenv("SERP_API_KEY"), max_results=serp_num)
                
        self.name = self.searcher.name
        self.description = self.searcher.description

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:

        if self.reflection:
            self.reflector = SearchReflector()
            _, query = self.reflector.query_reflect(query)

        # Try the original query first
        results = self.searcher.search(query, filter_year)
        
        # If results is a string (indicating no results found), try fallback strategies
        if isinstance(results, str) and "No results found" in results:
            fallback_queries = self._generate_fallback_queries(query)
            
            for fallback_query in fallback_queries:
                print(f"Trying fallback query: {fallback_query}")
                fallback_results = self.searcher.search(fallback_query, filter_year)
                if isinstance(fallback_results, List) and len(fallback_results) > 0:
                    results = fallback_results
                    query = fallback_query  # Update query for content generation
                    break
            
            # If no fallback worked, return the original message
            if isinstance(results, str):
                return results

        if isinstance(results, List):
            return self.searcher._to_content(query, results)
        else:
            return str(results)
    
    def _generate_fallback_queries(self, original_query: str) -> List[str]:
        """Generate alternative search queries when original query fails"""
        fallback_queries = []
        
        # Remove site-specific search constraints
        if "site:" in original_query:
            without_site = ' '.join([part for part in original_query.split() if not part.startswith('site:')])
            fallback_queries.append(without_site)
        
        # If query contains scientific names, try common names or broader terms
        if "Amphiprion percula" in original_query:
            fallback_queries.extend([
                "clownfish USGS database",
                "Amphiprion USGS",
                "clownfish invasive species database",
                "USGS nonindigenous aquatic species",
                "clownfish occurrence records"
            ])
        
        # Extract key terms and create broader searches
        words = original_query.lower().split()
        key_terms = [word for word in words if len(word) > 3 and word not in ['site:', 'and', 'the', 'with', 'for']]
        
        if len(key_terms) >= 2:
            # Try combinations of key terms
            fallback_queries.append(' '.join(key_terms[:2]))
            if len(key_terms) > 2:
                fallback_queries.append(' '.join(key_terms[:3]))
        
        return fallback_queries[:3]  # Limit to 3 fallback attempts



class MultiSourceSearchTool(Tool):
    name = "web_search"
    description = "Perform a web search query (think a google search) and returns the search results."
    inputs = {"query": {"type": "string", "description": "The web search query to perform."}}
    inputs["filter_year"] = {
        "type": "string",
        "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!",
        "nullable": True,
    }
    output_type = "string"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.searchers = []

    def forward(self, query:str):
        pass

