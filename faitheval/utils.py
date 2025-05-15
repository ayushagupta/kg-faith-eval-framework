import re
import faitheval.constants as constants # only need kg-dependent ones

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


def is_negative_relation(relation):
    """
    Detect negation - reflects presence of words 'no' or 'not' in input string
    """
    return bool(_NEG_REGEX.search(relation))


def get_positive_relation(negative_relation_str):
    """
    Convert negative relation to positive form - attempt by removing negative words in _NEG_REGEX
    Example: "does not associate" -> "does associate"
             "is not a member of" -> "is a member of"
             "not linked to" -> "linked to"
    """
    positive_form = _NEG_REGEX.sub("", negative_relation_str)
    positive_form = re.sub(r"\s+", " ", positive_form).strip()
    return positive_form


def get_entity_type_and_simplified_name(raw_entity_text):
    """
    Determines entity type from raw_entity_text based on prefixes defined in 
    constants.ENTITY_TYPE_PREFIX_MAP, and returns the simplified name using 
    the global _simplify function.

    Inputs:
        raw_entity_text: str - The original, unsimplified entity string from RAG.
    Returns:
        tuple(str, str) - (simplified_name, entity_type)
    """
    raw_lower = raw_entity_text.lower()
    entity_type = constants.DEFAULT_ENTITY_TYPE

    for prefix, type_name in constants.ENTITY_TYPE_PREFIX_MAP.items():
        if raw_lower.startswith(prefix):
            entity_type = type_name
            break 
    
    simplified_name = _simplify(raw_entity_text) # _simplify uses _SIMPLIFY_REGEX
    return simplified_name, entity_type
