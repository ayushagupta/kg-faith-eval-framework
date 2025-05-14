"""
Constants for path length, similarity thresholds, etc.
"""

MAX_PATH_LEN = 3                   # longest path searched when no direct edge
MIN_LINK_SCORE = 0.30              # links with scores below this threshold are considered to be incorrect - entire record score goes to zero

TRIPLE_SIM_THRESHOLD  = 0.80       # cosine similarity threshold to consider as a match for triple embeddings
ENT_JACC_THRESHOLD = 0.90          # jacard measure threshold for entity matching (token overlap - mainly to handle GENE/PROTEIN code entities)
ENT_COS_THRESHOLD  = 0.80          # embedding cosine similarity threshold for entity matching (general entity subjects)

# Scoring - negative triples
NEG_TRIPLE_ONE_ENTITY_SCORE   = 0.8   # exactly one entity found in KG
NEG_TRIPLE_BOTH_ABSENT_SCORE  = 0.5   # neither entity found (neutral score since unknown)

# Keys are prefixes (lowercase, include trailing space) on raw RAG entity strings and their corresponding types
ENTITY_TYPE_PREFIX_MAP = {
    "gene ": "gene",
    "disease ": "disease",
    "protein ": "protein",
    "compound ": "compound"
}
DEFAULT_ENTITY_TYPE = "other" # Default type if no prefix from ENTITY_TYPE_PREFIX_MAP matches

######################################
# Constants that depend on knowledge-graph - Update based on entities present and use case
# RAG entities of these types will ONLY be matched exactly against a CoT entity - no fuzzy matching if exact match fails
######################################
STRICT_RAG_ENTITY_TYPES = {"gene", "protein"}
