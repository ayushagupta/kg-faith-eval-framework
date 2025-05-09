import numpy as np
from rag.utils import get_embedding_function, get_text_embedding
from config.config import config

from faitheval.utils import _simplify

# using same embedding fn for all comparisons (entities, triples)...
embedding_fn = get_embedding_function(model_name=config.EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)

_entity_embd_store = {}     # dict[str] = np.ndarray
_triple_embd_store = {}     # dict[(e1, r1, e2)] = np.ndarray


def _triple_to_str(triple):
    """
    Convert (source, relation, target) to a string, simplify before getting embedding
    """
    s, r, t = triple
    return f"{_simplify(s)} {r.strip().lower()} {_simplify(t)}"


def embed_triple(triple):
    """
    Creates and stores embedding for input triple
    """
    if triple not in _triple_embd_store:
        embedding  = get_text_embedding(_triple_to_str(triple), embedding_fn)
        _triple_embd_store[triple] = np.array(embedding, dtype=np.float32)
    return _triple_embd_store[triple]


def _embed_entity(entity):
    """
    Creates and stores embedding for input entity string
    """
    entity = _simplify(entity)
    if entity not in _entity_embd_store:
        embedding = get_text_embedding(entity, embedding_fn)
        _entity_embd_store[entity] = np.array(embedding, dtype=np.float32)
    return _entity_embd_store[entity]
