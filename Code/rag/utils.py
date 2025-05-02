from llm.openai_client import OpenAIClient
from prompts.system_prompts import get_system_prompt
from prompts.user_prompts import get_user_prompt
from utils.schema_loader import load_task_schema
from langchain_huggingface import HuggingFaceEmbeddings


def extract_disease_entities(text: str, client: OpenAIClient):
    system_prompt = get_system_prompt(task="disease_entity_extraction")
    user_prompt = get_user_prompt(task="disease_entity_extraction", input_text=text)
    text_data = load_task_schema(task="disease_entity_extraction")
    if not text_data:
        return None
    response = client.generate_json_response(instructions=system_prompt, input_text=user_prompt, text_data=text_data)
    return response["diseases"]


def get_embedding_function(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)


def get_text_embedding(text, embedding_function):
    return embedding_function.embed_query(text)