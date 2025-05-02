from rag.utils import *
from config.config import config

class RAG:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.embedding_function = get_embedding_function(model_name=config.EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)

    
    def retrieve(self, question):
        disease_entities = extract_disease_entities(question, self.openai_client)
        question_embedding = get_text_embedding(question, self.embedding_function)