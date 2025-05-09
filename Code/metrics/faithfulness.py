import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from rag.utils import get_embedding_function, get_text_embedding
from metrics.graph_helpers import build_edge_index, build_adj_with_rel, find_paths
from config.config import config

MAX_PATH_LEN = 3                   # longest path searched when no direct edge
MIN_LINK_SCORE = 0.30              # links with scores below this threshold are considered to be incorrect - entire record score goes to zero

TRIPLE_SIM_THRESHOLD  = 0.70       # cosine similarity threshold to consider as a match for triple embeddings
ENT_JACC_THRESHOLD = 0.80          # jacard measure threshold for entity matching (token overlap - mainly to handle GENE/PROTEIN code entities)
ENT_COS_THRESHOLD  = 0.80          # embedding cosine similarity threshold for entity matching (general entity subjects)

_NEG_REGEX = re.compile(r"\bnot\b|\bno\b", flags=re.I)  # to detect negative relations
_SIMPLIFY_REGEX = re.compile(r"\b(?:gene|disease|mutations?)\b", flags=re.I) # removing stopwords...

def _simplify(text):
    """
    Input: string
    For uniformity - change everything to lowercase; remove stopwords, punctuation; whitespace handling

    Returns simplified string
    """
    text = _SIMPLIFY_REGEX.sub("", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def _tokens(text):
    """
    Input: string
    Tokenize input string - returns set of unique tokens
    """
    return set(_simplify(text).split())

def _token_overlap_jaccard(a, b):
    """
    Calculate Jaccard measure (token overlap) between 2 input strings a and b
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

rel_embedding_fn = get_embedding_function(model_name=config.EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
ent_embedding_fn = rel_embedding_fn         # using same embedding fn for all comparisons..
triple_embedding_fn = rel_embedding_fn

_entity_embd_store = {}     # dict[str] = np.ndarray]
_triple_embd_store = {}     # dict[(e1, r1-2, e2))] = np.ndarray

def _triple_to_str(triple):
    """
    Convert (source, relation, target) to a string, simplify before getting embedding
    """
    s, r, t = triple
    return f"{_simplify(s)} {r.strip().lower()} {_simplify(t)}"

def _embed_triple(triple):
    """
    Creates and stores embedding for input triple
    """
    if triple not in _triple_embd_store:
        embedding  = get_text_embedding(_triple_to_str(triple), triple_embedding_fn)
        _triple_embd_store[triple] = np.array(embedding, dtype=np.float32)
    return _triple_embd_store[triple]

def _embed_entity(entity):
    """
    Creates and stores embedding for input entity string
    """
    entity = _simplify(entity)
    if entity not in _entity_embd_store:
        embedding = get_text_embedding(entity, ent_embedding_fn)
        _entity_embd_store[entity] = np.array(embedding, dtype=np.float32)
    return _entity_embd_store[entity]

def _is_negative(relation):
    """
    Detect negation - reflects presence of words 'no' or 'not' in input string
    """
    return bool(_NEG_REGEX.search(relation))

def _fuzzy_match_entity(ent: str, rag_entities: Set[str]) -> Set[str]:
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
    entity_cot = _simplify(ent)

    # 1. exact string match
    exact = {e for e in rag_entities if e == entity_cot}
    if exact:
        return exact

    # 2. Jaccard measure over threshold
    jacc = {e for e in rag_entities if _token_overlap_jaccard(entity_cot, e) >= ENT_JACC_THRESHOLD}
    if jacc:
        return jacc

    # 3. embedding cosine similarity over threshold
    entity_cot_embd = _embed_entity(entity_cot)
    matches = set()
    for e in rag_entities:
        entity_rag_embd = _embed_entity(e)
        if cosine_similarity(entity_cot_embd.reshape(1, -1), entity_rag_embd.reshape(1, -1))[0][0] >= ENT_COS_THRESHOLD:
            matches.add(e)
    return matches

def score_record(record: Dict) -> float:
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

    rag_triples = [(_simplify(s), r, _simplify(t)) for (s, r, t) in rag_triples_raw]

    print("rag_triples: ", rag_triples)
    print("cot_triples: ", cot_triples)

    # setting up indices...
    edge_idx = build_edge_index(rag_triples)        # (s,t) -> relation
    adj = build_adj_with_rel(rag_triples)           # adjacency with relations
    rag_entities = {s for (s, _, _) in rag_triples} | {t for (_, _, t) in rag_triples}

    triple_scores = []

    for s_cot_raw, rel_cot, t_cot_raw in cot_triples:        
        rel_is_neg = _is_negative(rel_cot)
        #v_pred = _embed_relation(relation)
        cot_triple_embd = _embed_triple((s_cot_raw, rel_cot, t_cot_raw))

        source_entities_rag = _fuzzy_match_entity(s_cot_raw, rag_entities)
        target_entities_rag = _fuzzy_match_entity(t_cot_raw, rag_entities)

        # For CoT triples with positive relations (standard)
        if not rel_is_neg:
            best = 0.0
            
            for s in source_entities_rag:
                for t in target_entities_rag:
                    # direct edge - check both directions (wording differences), it's loaded the right way in edge_idx
                    for pair in [(s, t), (t, s)]:
                        if pair in edge_idx:
                            relation_rag = edge_idx[pair]
                            rag_triple_embd = _embed_triple((pair[0], relation_rag, pair[1]))
                            cosine_sim = cosine_similarity(
                                cot_triple_embd.reshape(1, -1),
                                rag_triple_embd.reshape(1, -1)
                            )[0][0]
                            if cosine_sim >= TRIPLE_SIM_THRESHOLD:
                                best = max(best, cosine_sim)
                                print(f"Positive triple: Found direct edge: ({s}, {rel_cot}, {t})")
                    
                    # look for paths
                    for path in find_paths(adj, s, t, MAX_PATH_LEN):
                        #TODO: sentence embdg over relations istead of avging...
                        link_sim_scores = []
                        for source, relation_rag, target in path:
                            rag_triple_embd = _embed_triple((source, relation_rag, target))
                            link_sim_scores.append(
                                cosine_similarity(
                                    cot_triple_embd.reshape(1, -1),
                                    rag_triple_embd.reshape(1, -1)
                                )[0][0]
                            )
                        if link_sim_scores:
                            best = max(best, float(np.mean(link_sim_scores)))
                            print(f"Positive CoT triple: Found path between: ({s}, {t}) in KG")
                    
                    if best == 0:
                        print(f"Postive CoT triple: Path does not exist between entities in KG: ({s}, {t})")

            score = best  # stays 0 when triple not found

        # For CoT triples with negative relations (negation of relations defined in KG - Ex: "does not associate", "is not", etc.)       
        else:
            contradiction = False

            # if edge/path actually exists in KG, it contradicts negation stated in CoT
            for s in source_entities_rag:
                for t in target_entities_rag:
                    for pair in [(s, t), (t, s)]:
                        if pair in edge_idx:
                            contradiction = True
                            print(f"Negative CoT triple: Found direct edge in KG: ({pair[0]}, {edge_idx[pair]}, {pair[1]})")
                            break
                    for path in find_paths(adj, s, t, MAX_PATH_LEN):
                        contradiction = True
                        print(f"Negative CoT triple: Found path in KG: ({path})")
                        break
                if contradiction:
                    break

            if contradiction:
                score = 0.0
            else:
                entity1_found_in_kg = bool(source_entities_rag)
                entity2_found_in_kg = bool(target_entities_rag)

                if entity1_found_in_kg and entity2_found_in_kg:
                    score = 1.0
                    print(f"Negative triple in CoT: Found both entities in KG, link absent: ({s_cot_raw}, {rel_cot}, {t_cot_raw})")
                elif entity1_found_in_kg or entity2_found_in_kg:
                    score = 0.7
                    print(f"Negative triple in CoT: Found only one entity in KG: ({s_cot_raw}, {rel_cot}, {t_cot_raw})")
                else:
                    score = 0.5
                    print(f"Negative triple in CoT: Both entities absent from KG: ({s_cot_raw}, {rel_cot}, {t_cot_raw})")

        triple_scores.append(score)

    # Incorrect intermediate step in CoT falsifies all subsequent steps ... 
    if any(s < MIN_LINK_SCORE for s in triple_scores):
        return 0.0

    return float(np.mean(triple_scores)) if triple_scores else 0.0