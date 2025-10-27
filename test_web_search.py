from tools import web_search_function
print("Web search tool (Tavily) initialized.")
from dotenv import load_dotenv
import os

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Test the web search
print("--- Testing Web Search Tool ---")
test_query_web = "AMD AI chip strategy 2024"
test_results_web = web_search_function(test_query_web)
print(f"Found {len(test_results_web)} results for query: '{test_query_web}'")
if test_results_web:
    print(f"Top result snippet: {test_results_web[0].page_content[:250]}...")