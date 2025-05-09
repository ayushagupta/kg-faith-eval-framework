"""
Constants for path length, similarity thresholds, etc.
"""

MAX_PATH_LEN = 3                   # longest path searched when no direct edge
MIN_LINK_SCORE = 0.30              # links with scores below this threshold are considered to be incorrect - entire record score goes to zero

TRIPLE_SIM_THRESHOLD  = 0.70       # cosine similarity threshold to consider as a match for triple embeddings
ENT_JACC_THRESHOLD = 0.80          # jacard measure threshold for entity matching (token overlap - mainly to handle GENE/PROTEIN code entities)
ENT_COS_THRESHOLD  = 0.80          # embedding cosine similarity threshold for entity matching (general entity subjects)

# Scoring - negative triples
NEG_TRIPLE_ONE_ENTITY_SCORE   = 0.8   # exactly one entity found in KG
NEG_TRIPLE_BOTH_ABSENT_SCORE  = 0.5   # neither entity found (neutral score since unknown)