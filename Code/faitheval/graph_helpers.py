from collections import defaultdict, deque

def build_edge_index(triples):
    """
    Inputs: list of triples [(source, relation, target)]
    Returns dict[(source, target): relation]

    output mainly for finding link after matching entities
    """
    return {(s.lower(), t.lower()): r for (s, r, t) in triples}


def build_adj_with_rel(triples):
    """
    Inputs: list of triples [(source, relation, target)]
    Returns adjacency list (default dict) which stores relation with each directed edge
    """
    adj = defaultdict(list)
    for s, r, t in triples:
        adj[s].append((t, r))
    return adj


def find_paths(adj, start_node, final_node, max_len = 2):
    """
    Inputs: 
        adj: defaultdict
        start_node: str - starting entity
        final_node: str - ending entity
        max_len: int - max possible length of path
    
    Returns list of paths
    Each path is list of triples: [(src, r1, n1), (n1, r2, n2) ... (n, r, dst)].
    """
    
    if start_node not in adj:
        return [] # source entity unreachable

    paths = [] # List[List[(source, relation, target)]]]
    q = deque( [[(start_node, None, start_node)]] )  # seed tuple, will drop None relation

    while q:
        path = q.popleft()
        last_node = path[-1][2]

        # stop exploring if already 'max_len' links deep
        if len(path) - 1 >= max_len:
            continue

        for next_entity, rel in adj.get(last_node, []):
            new_path = path + [(last_node, rel, next_entity)]
            if next_entity == final_node:
                paths.append(new_path[1:])    # leave seed tuple out
            else:
                q.append(new_path)

    return paths
