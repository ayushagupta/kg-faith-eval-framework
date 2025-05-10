import re, ast
from llm.openai_client import OpenAIClient
from cot2kg.prompts import COT2KG_PROMPT

_client = OpenAIClient()

_PY_BLOCK = re.compile(r"<python>(.*?)</python>", re.DOTALL | re.IGNORECASE)

def _extract_triples(text):
    """
    Pull the list-of-triples Python code out of <python> ... </python> tags.
    """
    m = _PY_BLOCK.search(text)
    if not m:
        print("Model output:", text)
        raise ValueError("No <python> block found in model output.")
    block = m.group(1).strip()
    triples = ast.literal_eval(block) 
    if not isinstance(triples, list):
        raise ValueError("Parsed object is not a list.")
    return triples

def cot_to_kg(chain_of_thought):
    """
    Convert one chain-of-thought string to a list of triples.
    """
    raw = _client.generate_response(
        instructions = COT2KG_PROMPT.strip(),
        input_text = chain_of_thought
    )
    return _extract_triples(raw)
