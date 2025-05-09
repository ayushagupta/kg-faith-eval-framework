import re

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
