"""
Modes:
    all - keep all original keys + cot_kg  (default)
    compact - keep only question_id, question, correct_answer, model_answer, kg_rag, cot_kg
"""

import argparse
from pathlib import Path
from tqdm import tqdm

import cot2kg.convert_to_kg as convert_to_kg
from cot2kg.io_utils import load_json, save_json
from cot2kg.config import DEFAULT_OUTPUT_MODE

_COMPACT_KEYS = ["question_id", "question", "correct_answer", "model_answer", "kg_rag"]

def _build_record(src: dict, kg: list, mode: str) -> dict:
    if mode == "compact":
        keep = {k: src.get(k) for k in _COMPACT_KEYS}
        return keep | {"cot_kg": kg}
    if mode == "all":
        return src | {"cot_kg": kg}
    raise ValueError(f"Unknown mode {mode}")

def _process(inp: Path, out: Path, mode: str):
    data = load_json(inp)
    out_data = []

    for rec in tqdm(data, desc="Processing records"):
        cot = rec.get("chain_of_thought")
        if cot is None or "":
            raise ValueError("Record missing 'chain_of_thought'")
        triples = convert_to_kg.cot_to_kg(cot)
        out_data.append(_build_record(rec, triples, mode))

    save_json(out_data, out)
    print(f"Wrote {out}  ({len(out_data)} records, mode={mode})")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="inp",  required=True)
    p.add_argument("--out", dest="out",  required=True)
    p.add_argument("--mode", choices=["all", "compact"],
                   default=DEFAULT_OUTPUT_MODE)
    args = p.parse_args()

    _process(Path(args.inp), Path(args.out), args.mode)

if __name__ == "__main__":
    main()
