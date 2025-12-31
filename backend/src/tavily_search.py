import os
from tavily import TavilyClient
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))


def search(query):
    """Search internet using Tavily API."""
    output_search = tavily_client.search(query).get('results', [])[:3]
    search_document = "Dưới đây là các tài liệu truy xuất được từ internet: \n"
    for i, doc in enumerate(output_search):
        search_document += f"{i+1}. {doc.get('content', '')} \n"
    search_document += "Kết thúc phần tài liệu truy xuất được."
    return search_document


# Function info for tool calling (kept for compatibility)
functions_info = [
    {
        "name": "search",
        "description": "Get information from internet based on user query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "This is user query",
                },
            },
            "required": ["query"],
        },
    }
]
