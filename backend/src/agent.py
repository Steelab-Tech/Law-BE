import os
import json
from tavily import TavilyClient
import openai
from dotenv import load_dotenv, find_dotenv

from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool

_ = load_dotenv(find_dotenv())  # Read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def search(query):
    # Get first 3 search results
    output_search = tavily_client.search(query).get('results')[:3]
    # Process search results into document string
    search_document = "Below are documents retrieved from the internet: \n"
    for i, doc in enumerate(output_search):
        search_document += f"{i+1}. {doc.get('content', '')} \n"
    search_document += "End of retrieved documents."
    return search_document

search_tool = FunctionTool.from_defaults(fn=search)
llm = OpenAI(model="gpt-4o-mini")
legal_agent = ReActAgent.from_tools([search_tool], llm=llm, verbose=True)

def convert_raw_messages_to_chat_messages(messages):
    """
    Convert a list of messages to a list of ChatMessage instances.

    Args:
        messages (list): List of dictionaries with keys 'role' and 'content'.

    Returns:
        list: List of ChatMessage instances.
    """
    chat_messages = []
    for message in messages:
        role = message.get("role", MessageRole.USER)
        content = message.get("content", "")
        chat_message = ChatMessage.from_str(content=content, role=role)
        chat_messages.append(chat_message)
    return chat_messages


def react_agent_handle(history, question):
    chat_history = convert_raw_messages_to_chat_messages(history)
    response = legal_agent.chat(message=question, chat_history=chat_history)
    return response.response
