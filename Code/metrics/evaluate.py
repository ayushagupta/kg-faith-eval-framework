import argparse, json, numpy as np
from metrics.faithfulness import score_record

def main(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    scores = [score_record(rec) for rec in data]
    for rec, sc in zip(data, scores):
        print(f"Q{rec['question_id']:>4}: faithfulness = {sc:.3f}")
    print("\nOverall average:", f"{float(np.mean(scores)):.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KG‑faithfulness evaluation")
    p.add_argument("json_file")
    main(p.parse_args().json_file)
