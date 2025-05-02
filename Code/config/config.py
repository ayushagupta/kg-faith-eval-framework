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


config = Config()