import argparse
import json
import numpy as np

from metrics.faithfulness import score_record


def main(filepath):
    data = json.load(open(filepath, "r", encoding="utf-8"))
    scores = [score_record(rec) for rec in data]

    for row, score in zip(data, scores):
        print(f"{row['question_id']} : faithfulness = {score:.3f}")

    print("\nOverall average:", f"{float(np.mean(scores)):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG-faithfulness evaluation")
    parser.add_argument("json_file")
    main(parser.parse_args().json_file)
