import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import faitheval.constants as constants
from faitheval.utils import _simplify, _token_overlap_jaccard
from faitheval.embedding_helpers import _embed_entity, embed_triple
from faitheval.graph_helpers import build_edge_index, build_adj_with_rel, find_paths

def prepare_rag_structures(rag_triples_raw):
    """
    Inputs: 
        rag_triples_raw: List[(source, relation, target))]
    
    Returns preprocessed triples, edge-index, adjacency list and entity set
    """
    rag_triples = [(_simplify(s), r, _simplify(t)) for s, r, t in rag_triples_raw]
    edge_idx = build_edge_index(rag_triples)
    adj      = build_adj_with_rel(rag_triples)
    entities = {s for (s, _, _) in rag_triples} | {t for (_, _, t) in rag_triples}
    return rag_triples, edge_idx, adj, entities


def _cosine_sim(v1, v2):
    return float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0])


def fuzzy_match_entity(entity, rag_entities):
    """
    Inputs: 
        entity: str
        rag_entities: Set[str]
    
    Returns Set[str] - subset of entities in rag_entities

    Fuzzy entity match possible when:
        1. exact string match
        2. Jaccard measure over threshold
        3. embedding cosine similarity over threshold
    """
    entity_cot = _simplify(entity)

    # 1. exact string match
    exact = {e for e in rag_entities if e == entity_cot}
    if exact:
        return exact

    # 2. Jaccard measure over threshold
    jacc = {e for e in rag_entities if _token_overlap_jaccard(entity_cot, e) >= constants.ENT_JACC_THRESHOLD}
    if jacc:
        return jacc

    # 3. embedding cosine similarity over threshold
    entity_cot_embd = _embed_entity(entity_cot)
    matches = set()
    for e in rag_entities:
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
    return link_sim_scores #float(np.mean(sims)) if sims else 0.0


def score_positive_triple(cot_embd, source_entities_rag, target_entities_rag, edge_idx, adj):
    best = 0.0
    for s in source_entities_rag:
        for t in target_entities_rag:
            # direct edge - check both directions (wording differences), it's loaded the right way in edge_idx
            for pair in [(s, t), (t, s)]:
                if pair in edge_idx:
                    cosine_sim_score = _cosine_sim(cot_embd, embed_triple((pair[0], edge_idx[pair], pair[1])))
                    if cosine_sim_score >= constants.TRIPLE_SIM_THRESHOLD:
                        best = max(best, cosine_sim_score)
                        print(f"Positive triple: Found direct edge: ({pair[0]}, {edge_idx[pair]}, {pair[1]})")
            # look for paths
            for path in find_paths(adj, s, t, constants.MAX_PATH_LEN):
                #TODO: sentence embdg over relations istead of avging...
                link_sim_scores = _path_similarity(cot_embd, path)
                if link_sim_scores:
                    best = max(best, float(np.mean(link_sim_scores)))
                    print(f"Positive CoT triple: Found path between: ({s}, {t}) in KG")
    return best


def score_negative_triple(source_entities_rag, target_entities_rag, edge_idx, adj):
    # if edge/path actually exists in KG, it contradicts negation stated in CoT - hallucination
    for s in source_entities_rag:
        for t in target_entities_rag:
            for pair in [(s, t), (t, s)]:
                if pair in edge_idx:
                    print(f"Negative triple in CoT: Found direct edge in KG: ({pair[0]}, {edge_idx[pair]}, {pair[1]})")
                    return 0.0
            for path in find_paths(adj, s, t, constants.MAX_PATH_LEN):
                print(f"Negative triple in CoT: Found path in KG: ({path})")
                return 0.0

    entity1_in_kg = bool(source_entities_rag)
    entity2_in_kg = bool(target_entities_rag)

    if entity1_in_kg and entity2_in_kg:
        print(f"Negative triple in CoT: Found both entities in KG, link absent: ({source_entities_rag}, {target_entities_rag})")
        score = 1.0
    elif entity1_in_kg or entity2_in_kg:
        print(f"Negative triple in CoT: Found only one entity in KG: ({source_entities_rag}, {target_entities_rag})")
        score = constants.NEG_TRIPLE_ONE_ENTITY_SCORE
    else:
        print(f"Negative triple in CoT: Both entities absent from KG: ({source_entities_rag}, {target_entities_rag})")
        score = constants.NEG_TRIPLE_BOTH_ABSENT_SCORE

    return score
