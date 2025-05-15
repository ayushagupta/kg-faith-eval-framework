import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import faitheval.constants as constants
from faitheval.utils import _simplify, _token_overlap_jaccard, get_entity_type_and_simplified_name, get_positive_relation
from faitheval.embedding_helpers import _embed_entity, embed_triple
from faitheval.graph_helpers import build_edge_index, build_adj_with_rel, find_paths
from faitheval.logging_config import logger

def prepare_rag_structures(rag_triples_raw):
    """
    Inputs: 
        rag_triples_raw: List[(source, relation, target))]
    
    Returns:
        processed_triples_for_graph: List[(simplified_s, r, simplified_t)]
        edge_idx: Dict[(simplified_s, simplified_t), relation_str]
        adj: DefaultDict[simplified_s_str, List[(simplified_t_str, relation_str)]]
        all_simplified_rag_entities_set: Set[simplified_entity_str]
        rag_entity_details: Dict[simplified_entity_str, {"type": type_str, "raw": raw_text_str}]
    """
    rag_entity_details = {} 
    processed_triples_for_graph = []

    for s_raw, r, t_raw in rag_triples_raw:
        s_simplified, s_type = get_entity_type_and_simplified_name(s_raw)
        t_simplified, t_type = get_entity_type_and_simplified_name(t_raw)
        
        processed_triples_for_graph.append((s_simplified, r, t_simplified))
        
        if s_simplified not in rag_entity_details:
            rag_entity_details[s_simplified] = {"type": s_type, "raw": s_raw}
        
        if t_simplified not in rag_entity_details:
           rag_entity_details[t_simplified] = {"type": t_type, "raw": t_raw}
    
    edge_idx = build_edge_index(processed_triples_for_graph)
    adj = build_adj_with_rel(processed_triples_for_graph)
    
    all_simplified_rag_entities_set = set(rag_entity_details.keys())
    
    return processed_triples_for_graph, edge_idx, adj, all_simplified_rag_entities_set, rag_entity_details


def _cosine_sim(v1, v2):
    return float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0])


def fuzzy_match_entity(entity, all_simplified_rag_entities_set, rag_entity_details, strict_rag_types):
    """
    Inputs: 
        entity: str
        all_simplified_rag_entities_set: Set[str] - All unique simplified RAG entity strings.
        rag_entity_details: Dict[str, Dict] - Maps simplified RAG entity strings to their details 
                                              (e.g., {"type": "gene", "raw": "Gene RNF168"}).
        strict_rag_types: Set[str] - RAG entity types that should only be matched exactly 
                                    (e.g., constants.STRICT_RAG_ENTITY_TYPES).

    Returns Set[str] - Subset of simplified RAG entity strings from all_simplified_rag_entities_set.


    Fuzzy entity match possible when:
        1. exact string match
        2. Jaccard measure over threshold
        3. embedding cosine similarity over threshold
    """
    entity_cot_simplified = _simplify(entity)

    # 1. exact string match
    exact = {e for e in all_simplified_rag_entities_set if e == entity_cot_simplified}
    if exact:
        return exact

    #  prep for fuzzy matching, and don't allow strict types to be fuzzy match candidates
    fuzzy_candidate_rag_entities = {
        e_rag_s for e_rag_s in all_simplified_rag_entities_set
        if rag_entity_details.get(e_rag_s, {}).get("type") not in strict_rag_types
    }

    # 2. Jaccard measure over threshold
    jacc = {e for e in fuzzy_candidate_rag_entities if _token_overlap_jaccard(entity_cot_simplified, e) >= constants.ENT_JACC_THRESHOLD}
    if jacc:
        return jacc

    # 3. embedding cosine similarity over threshold
    entity_cot_embd = _embed_entity(entity_cot_simplified)
    matches = set()
    for e in fuzzy_candidate_rag_entities:
        entity_rag_embd = _embed_entity(e)
        if _cosine_sim(entity_cot_embd, entity_rag_embd)  >= constants.ENT_COS_THRESHOLD:
            matches.add(e)
    return matches


#TODO: sentence embdg over relations istead of avging...
def _path_similarity(cot_embd, path_rag):
    """
    Inputs: 
        cot_embd: np.ndarray
        path: List[(source, relation, target)]
    Returns mean cosine similarity between CoT triple embedding and every triple along the given path
    """
    link_sim_scores = [
        _cosine_sim(cot_embd, embed_triple((source, relation, target)))
        for source, relation, target in path_rag
    ]
    return link_sim_scores


def score_positive_triple(cot_embd, source_entities_rag, target_entities_rag, edge_idx, adj, cot_triple_raw, hallucination_recorder):
    best = 0.0
    found_evidence = False
    reason_for_zero = "No supporting evidence found in KG for positive CoT triple" # no edge/path above similarity thresholds

    for s in source_entities_rag:
        for t in target_entities_rag:
            # direct edge - check both directions (wording differences), it's loaded the right way in edge_idx
            for pair in [(s, t), (t, s)]:
                if pair in edge_idx:
                    cosine_sim_score = _cosine_sim(cot_embd, embed_triple((pair[0], edge_idx[pair], pair[1])))
                    if cosine_sim_score >= constants.TRIPLE_SIM_THRESHOLD:
                        best = max(best, cosine_sim_score)
                        logger.info(f"Positive CoT triple: Found direct edge: ({pair[0]}, {edge_idx[pair]}, {pair[1]})")
                        found_evidence = True
            # look for paths
            for path in find_paths(adj, s, t, constants.MAX_PATH_LEN):
                #TODO: sentence embdg over relations istead of avging...
                link_sim_scores = _path_similarity(cot_embd, path)
                path_avg_sim_score = float(np.mean(link_sim_scores))
                if path_avg_sim_score >= constants.TRIPLE_SIM_THRESHOLD:
                    best = max(best, path_avg_sim_score)
                    logger.info(f"Positive CoT triple: Found path between: ({s}, {t}) in KG with averaged cosine sim score = {path_avg_sim_score:.3f}")
                    found_evidence = True
    
    if not found_evidence:
        logger.info(reason_for_zero)
        hallucination_recorder.append({
            "cot_triple": list(cot_triple_raw),
            "retrieved_sources": list(source_entities_rag),
            "retrieved_targets": list(target_entities_rag),
            "highest_sim_score": round(best, 3),
            "reason": reason_for_zero
        })
    return best


def score_negative_triple(source_entities_rag, target_entities_rag, edge_idx, adj, cot_triple_raw, hallucination_recorder):
    s_cot_raw, rel_cot_negative, t_cot_raw = cot_triple_raw
    rel_cot_positive = get_positive_relation(rel_cot_negative)
    cot_positive_form_embd = embed_triple((s_cot_raw, rel_cot_positive, t_cot_raw))

    # if edge/path actually exists in KG, it contradicts negation stated in CoT - hallucination
    for s in source_entities_rag:
        for t in target_entities_rag:
            for path in find_paths(adj, s, t, constants.MAX_PATH_LEN):
                path_avg_sim_score = round(float(np.mean(_path_similarity(cot_positive_form_embd, path))), 3)
            
                if path_avg_sim_score >= constants.TRIPLE_SIM_THRESHOLD:
                    reason = f"Negative triple in CoT: Contradicted by path in KG: ({path})"
                    logger.info(reason)
                    hallucination_recorder.append({
                        "cot_triple": list(cot_triple_raw),
                        "retrieved_sources": list(source_entities_rag),
                        "retrieved_targets": list(target_entities_rag),
                        "similarity_to_positive_cot": path_avg_sim_score,
                        "highest_sim_score": 0.0,
                        "reason": reason
                    })
                    return 0.0
                else:
                    reason = f"Negative triple in CoT: Not exactly contradicted by path in KG: ({path})"
                    logger.info(reason)
                    hallucination_recorder.append({
                        "cot_triple": list(cot_triple_raw),
                        "retrieved_sources": list(source_entities_rag),
                        "retrieved_targets": list(target_entities_rag),
                        "similarity_to_positive_cot": round(path_avg_sim_score, 3),
                        "highest_sim_score": path_avg_sim_score,
                        "reason": reason
                    })
                    return path_avg_sim_score
        
            # look for direct edge after checking for path
            for pair in [(s, t), (t, s)]:
                if pair in edge_idx:
                    edge_kg_triple = (pair[0], edge_idx[pair], pair[1])
                    
                    edge_kg_embd = embed_triple(edge_kg_triple)
                    edge_sim_score = round(_cosine_sim(cot_positive_form_embd, edge_kg_embd), 3)
                    
                    if edge_sim_score >= constants.TRIPLE_SIM_THRESHOLD: 
                        reason = f"Negative triple in CoT: Contradicted by direct edge in KG: ({pair[0]}, {edge_idx[pair]}, {pair[1]})"
                        logger.info(reason)
                        hallucination_recorder.append({
                            "cot_triple": list(cot_triple_raw),
                            "retrieved_sources": list(source_entities_rag),
                            "retrieved_targets": list(target_entities_rag),
                            "similarity_to_positive_cot": edge_sim_score,
                            "highest_sim_score": 0.0,
                            "reason": reason
                        })
                        return 0.0
                    else:
                        reason = f"Negative triple in CoT: Not exactly contradicted by direct edge in KG: ({pair[0]}, {edge_idx[pair]}, {pair[1]})"
                        logger.info(reason)
                        hallucination_recorder.append({
                            "cot_triple": list(cot_triple_raw),
                            "retrieved_sources": list(source_entities_rag),
                            "retrieved_targets": list(target_entities_rag),
                            "similarity_to_positive_cot": edge_sim_score,
                            "highest_sim_score": edge_sim_score,
                            "reason": reason
                        })
                        return edge_sim_score

    entity1_in_kg = bool(source_entities_rag)
    entity2_in_kg = bool(target_entities_rag)

    if entity1_in_kg and entity2_in_kg:
        logger.info(f"Negative triple in CoT: Found both entities in KG, link absent: ({source_entities_rag}, {target_entities_rag})")
        score = 1.0
    elif entity1_in_kg or entity2_in_kg:
        logger.info(f"Negative triple in CoT: Found only one entity in KG: ({source_entities_rag}, {target_entities_rag})")
        score = constants.NEG_TRIPLE_ONE_ENTITY_SCORE
    else:
        reason = "Negative triple in CoT: Both entities absent from KG"
        logger.info(reason)
        score = constants.NEG_TRIPLE_BOTH_ABSENT_SCORE
        hallucination_recorder.append({
            "cot_triple": list(cot_triple_raw),
            "retrieved_sources": list(source_entities_rag),
            "retrieved_targets": list(target_entities_rag),
            "highest_sim_score": score,
            "reason": reason
        })

    return score
