import asyncio
import logging
from copy import copy

from celery import shared_task

from utils import setup_logging
from database import get_celery_app
from brain import detect_route, llm_chat_complete, detect_user_intent, gen_doc_prompt, \
    get_legal_agent_anwer
from models import update_chat_conversation, get_conversation_messages
import requests

setup_logging()
logger = logging.getLogger(__name__)

celery_app = get_celery_app(__name__)
celery_app.autodiscover_tasks()


@shared_task()
def bot_answer_message(history, message):
    user_intent = detect_user_intent(history, message)
    logger.info(f"User intent: {user_intent}")

    # Call api retrieval relevance document
    url = "http://localhost:8002/retrieval"
    payload = {
        "query": user_intent,
        "top_k_search": 30,
        "top_k_rerank": 5
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        top_docs = response.json().get("results")
    else:
        print("Error:", response.status_code, response.text)
        top_docs = []

    # Use history as messages
    session_history = copy(history)

    messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý thông minh, hãy trở lời câu hỏi hiện tại của user dựa trên lịch sử chat và các tài liệu liên quan.
                            Câu trả lời phải ngắn gọn, chính xác nhưng vẫn đảm bảo đầy đủ các ý chính.
                NOTE:  - Hãy chỉ trả lời nếu câu trả lời nằm trong tài liệu được truy xuất ra.
                       - Nếu không tìm thấy câu trả lời trong tài liệu truy xuất ra thì hãy trả về : "no"
                        """
            },
            *session_history
        ]

    # Update documents to prompt
    rag_messages = messages +  [
            {"role": "user", "content": gen_doc_prompt(top_docs)},
            {"role": "user", "content": message},
        ]

    assistant_answer = llm_chat_complete(rag_messages, max_new_tokens=512, temperature=0.7)

    # Check if answer is "no" (not found in documents)
    if assistant_answer.strip().lower() == "no" or assistant_answer.strip() == '"no"':
        logger.info(f"Answer not found in documents, calling web search")
        messages = history + [
            {"role": "user", "content": message},
        ]
        agent_answer = get_legal_agent_anwer(messages)
        return agent_answer
    else:
        logger.info(f"Answer found via RAG")
        return assistant_answer


@shared_task()
def bot_route_answer_message(history, question):
    # detect the route
    route = detect_route(history, question)
    if route == "chitchat":
        logger.info(f"Router to chitchat")
        mess_format = [
            {"role": "system", "content": "Là một trợ lý thông minh, hãy trả lời các câu hỏi này dựa theo tri thức của bạn và hãy trả về kết quả là tiếng Việt."},
            {"role": "user", "content": question}
            ]
        output_chitchat = llm_chat_complete(mess_format, max_new_tokens=256, temperature=0.7)
        return output_chitchat

    elif route == 'legal':
        logger.info("Router to legal topic")
        return bot_answer_message(history, question)

    else:
        return "Xin lỗi, tôi không hiểu câu hỏi của bạn."


@shared_task()
def llm_handle_message(bot_id, user_id, question):
    logger.info("Start handle message")
    # Update chat conversation for new user_question
    conversation_id = update_chat_conversation(bot_id, user_id, question, True)
    logger.info("Conversation id: %s", conversation_id)

    # Convert history to list messages
    messages = get_conversation_messages(conversation_id)
    logger.info("Conversation messages: %s", messages)
    history = messages[-5:-1]
    # Use bot route to handle message
    response = bot_route_answer_message(history, question)
    logger.info(f"Chatbot response: {response}")
    # Save response to history
    update_chat_conversation(bot_id, user_id, response, False)
    # Return response
    return {"role": "assistant", "content": response}
