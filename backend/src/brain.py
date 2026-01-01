import json
import logging
import os
import re
from dotenv import load_dotenv

load_dotenv()
from local_llm import local_chat_complete
from tavily_search import search

logger = logging.getLogger(__name__)


def llm_chat_complete(messages, max_new_tokens=512, temperature=0.7):
    """Wrapper for local LLM chat completion."""
    return local_chat_complete(messages, max_new_tokens, temperature)


def gen_doc_prompt(docs):
    """Generate document prompt from retrieved docs."""
    doc_prompt = "Dưới đây là tài liệu về các điều luật liên quan đến câu hỏi của người dùng:"
    for i, doc in enumerate(docs):
        doc_prompt += f"{i}. {doc} \n"
    doc_prompt += "Kết thúc phần các tài liệu liên quan."
    return doc_prompt


def generate_conversation_text(conversations):
    """Convert conversation list to text format."""
    conversation_text = ""
    for conversation in conversations:
        logger.info("Generate conversation: {}".format(conversation))
        role = conversation.get("role", "user")
        content = conversation.get("content", "")
        conversation_text += f"{role}: {content}\n"
    return conversation_text


def detect_user_intent(history, message):
    """Rewrite user query based on history for better retrieval."""
    history_messages = generate_conversation_text(history)
    logger.info(f"History messages: {history_messages}")

    user_prompt = f"""
    Based on the following conversation history and the latest user query, rewrite the latest query as
    a standalone question in Vietnamese. The user may switch between different legal topics, such as
    traffic laws, economic regulations, etc., so ensure the intent of the user is accurately
    identified at the current moment to rephrase the query as precisely as possible.
    The rewritten question should be clear, complete, and understandable without additional context.

    Chat History:
    {history_messages}

    Original Question: {message}

    Answer:
    """
    messages = [
        {"role": "system", "content": "You are an amazing virtual assistant"},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"Rephrase input messages: {messages}")
    return llm_chat_complete(messages, max_new_tokens=256, temperature=0.3)


def detect_route(history, message):
    """Classify query as 'legal' or 'chitchat'."""
    logger.info(f"Detect route on history messages: {history}")

    user_prompt = f"""
    Given the following chat history and the user's latest message. Hãy phân loại xu hướng mong muốn trong tin nhắn của user là loại nào trong 2 loại sau.
    1. Mong muốn hỏi các thông tin liên quan đến luật pháp tại Việt Nam, các tình huống thực tế gặp phải liên quan đến luật
    Ví dụ: -  Nếu xe máy không đội mũ bảo hiểm thì bị phạt bao nhiêu tiền?
           -  Nếu ô tô đi ngược chiều thì bị phạt thế nào?
           -  Lập kế hoạch đấu giá quyền khai thác khoáng sản dựa trên các căn cứ nào ?
           -  Mục đích của bảo hiểm tiền gửi là gì ?
    => Loại này có nhãn là : "legal"
    2. Mong muốn chitchat thông thường.
    Ví dụ:  - Hi, xin chào, tôi cần bạn hỗ trợ,....
            - Chủ tịch nước Việt Nam là ai ,....
    => Loại này có nhãn là : "chitchat"

    Chỉ trả về một trong hai nhãn: "legal" hoặc "chitchat"

    Chat History:
    {history}

    Latest User Message:
    {message}

    Classification:
    """
    messages = [
        {"role": "system", "content": "You are a highly intelligent assistant that helps classify customer queries. Only respond with 'legal' or 'chitchat'."},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"Route output: {messages}")
    response = llm_chat_complete(messages, max_new_tokens=10, temperature=0.1)

    # Extract label from response
    response_lower = response.lower().strip()
    if "legal" in response_lower:
        return "legal"
    elif "chitchat" in response_lower:
        return "chitchat"
    else:
        # Default to legal if unclear
        return "legal"


def get_legal_agent_anwer(messages):
    """Search internet and generate response for legal questions not found in documents."""
    logger.info(f"Call tavily tool search")

    # Extract the latest user question
    user_question = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_question = msg.get("content", "")
            break

    # Search internet
    search_results = search(user_question)
    logger.info(f"Search results: {search_results[:200]}...")

    # Add search results to messages and generate response
    # Combine search results with original question to avoid consecutive user messages
    combined_content = f"Kết quả tìm kiếm từ internet:\n{search_results}\n\nDựa trên kết quả trên, hãy trả lời câu hỏi: {user_question}"
    full_messages = [
        {"role": "system", "content": "Bạn là trợ lý pháp luật thông minh. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp."},
        {"role": "user", "content": combined_content}
    ]

    response = llm_chat_complete(full_messages, max_new_tokens=512, temperature=0.7)
    return response


if __name__ == "__main__":
    history = [{"role": "system", "content": "You are an amazing virtual assistant"}]
    message = "Hello"
    output_detect = detect_route(history, message)
    print(output_detect)
