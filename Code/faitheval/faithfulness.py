import re
import numpy as np

import faitheval.constants as constants
from faitheval.logging_config import logger
from faitheval.utils import is_negative_relation
from faitheval.embedding_helpers import embed_triple
from faitheval.scoring_helpers import (
    prepare_rag_structures,
    fuzzy_match_entity,
    score_positive_triple,
    score_negative_triple
)


def score_record(record):
    """
    Inputs:
        record: dict - expected to be have the following keys:
        {
            ...
            question: str,
            kg_rag: List[(source, relation, target)],
            cot_kg: List[(source, relation, target)],
            ...
        }

    Outputs:
        faithfulness/groundedness score for the record (value between [0,1])
    """
    cot_triples = record["cot_kg"]
    rag_triples_raw = record["kg_rag"]
    #question_text = record["question"].lower() # TODO: check existence of entity from cot-kg in question, if entity is absent from both kg-rag and question, score 0 (hallucincation) and move to next record 
    
    hallucination_details_for_record = []

    simplified_rag_triples, edge_idx_rag, adj_rag, all_simplified_rag_entities_set, rag_entity_details = prepare_rag_structures(rag_triples_raw)

    logger.info("rag_triples: %s", simplified_rag_triples)
    logger.info("cot_triples: %s", cot_triples)

    triple_scores = []

    for s_cot_raw, rel_cot, t_cot_raw in cot_triples: 
        cot_triple_embd = embed_triple((s_cot_raw, rel_cot, t_cot_raw))
        cot_triple_raw = tuple((s_cot_raw, rel_cot, t_cot_raw))

        source_entities_rag = fuzzy_match_entity(s_cot_raw, all_simplified_rag_entities_set, rag_entity_details, constants.STRICT_RAG_ENTITY_TYPES)
        target_entities_rag = fuzzy_match_entity(t_cot_raw, all_simplified_rag_entities_set, rag_entity_details, constants.STRICT_RAG_ENTITY_TYPES)

        # For CoT triples with positive relations (standard)    
        if not is_negative_relation(rel_cot):
            score = score_positive_triple(cot_triple_embd, source_entities_rag, target_entities_rag, edge_idx_rag, adj_rag, cot_triple_raw, hallucination_details_for_record)
        # For CoT triples with negative relations (negation of relations defined in KG - Ex: "does not associate", "is not", etc.)
        else:
            score = score_negative_triple(source_entities_rag, target_entities_rag, edge_idx_rag, adj_rag, cot_triple_raw, hallucination_details_for_record)

        triple_scores.append(score)

    # Incorrect intermediate step in CoT falsifies all subsequent steps ... 
    if any(s < constants.MIN_LINK_SCORE for s in triple_scores):
        return 0.0, hallucination_details_for_record
    
    final_score = float(np.mean(triple_scores)) if triple_scores else 0.0
    return final_score, hallucination_details_for_record
