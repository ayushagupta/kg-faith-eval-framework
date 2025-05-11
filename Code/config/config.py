import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    
	# VectorDB config
	VECTOR_DB_DISEASE_ENTITY_PATH = os.getenv(
		"VECTOR_DB_DISEASE_ENTITY_PATH", "data/disease_with_relation_to_genes.pickle"
	)
	VECTOR_DB_PATH = os.getenv(
		"VECTOR_DB_PATH", "data/vectorDB/disease_nodes_db"
	)
	VECTOR_DB_CHUNK_SIZE = int(os.getenv("VECTOR_DB_CHUNK_SIZE", 650))
	VECTOR_DB_CHUNK_OVERLAP = int(os.getenv("VECTOR_DB_CHUNK_OVERLAP", 200))
	VECTOR_DB_BATCH_SIZE = int(os.getenv("VECTOR_DB_BATCH_SIZE", 200))
	VECTOR_DB_SENTENCE_EMBEDDING_MODEL = os.getenv(
		"VECTOR_DB_SENTENCE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
	)

	# OpenAI config
	OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
	TEMPERATURE = float(os.getenv("TEMPERATURE", 0))
	MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))

	# RAG config
	EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = os.getenv(
        "EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL", "pritamdeka/S-PubMedBert-MS-MARCO"
    )
	EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = os.getenv(
        "EMBEDDING_MODEL_FOR_NODE_RETRIEVAL", "sentence-transformers/all-MiniLM-L6-v2"
    )
	CONTEXT_VOLUME = int(os.getenv("CONTEXT_VOLUME", 150))
	QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = int(os.getenv("QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD", 75))
	QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(os.getenv("QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY", 0.5))

	# SPOKE API config
	BASE_URL = os.getenv("BASE_URL", "https://spoke.rbvi.ucsf.edu")
	CUTOFF_COMPOUND_MAX_PHASE  = int(os.getenv("CUTOFF_COMPOUND_MAX_PHASE ", 3))
	CUTOFF_PROTEIN_SOURCE = os.getenv("CUTOFF_PROTEIN_SOURCE", "SwissProt").split(',')
	CUTOFF_DAG_DISEASE_SOURCES = os.getenv("CUTOFF_DAG_DISEASES_SOURCES", "knowledge,experiments").split(',')
	CUTOFF_DAG_TERMINATING = int(os.getenv("CUTOFF_DAG_TEXTMINING", 3))
	CUTOFF_CTD_PHASE = int(os.getenv("CUTOFF_CTD_PHASE", 3))
	CUTOFF_PIP_CONFIDENCE = float(os.getenv("CUTOFF_PIP_CONFIDENCE", 0.7))
	CUTOFF_ACTEG_LEVEL = os.getenv("CUTOFF_ACTEG_LEVEL", "Low,Medium,High").split(',')
	CUTOFF_DPL_AVERAGE_PREVALENCE = float(os.getenv("CUTOFF_DPL_AVERAGE_PREVALENCE", 0.001))
	DEPTH = int(os.getenv("DEPTH", 1))
 
	def __init__(self, model_name: str = "gpt-4o-mini"):
		self.MODEL_NAME = model_name


config_mini = Config(model_name="gpt-4o-mini")
config_4o = Config(model_name="gpt-4o")