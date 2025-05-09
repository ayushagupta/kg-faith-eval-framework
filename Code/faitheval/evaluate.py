import argparse
import json
import numpy as np

from faitheval.faithfulness import score_record


def main(input_path, output_path):
    data = json.load(open(input_path, "r", encoding="utf-8"))
    scores = []
    for row in data:
        score = round(score_record(row), 3)
        row["faithfulness_score"] = score
        scores.append(score)
    
    for row in data:
        print(f"{row['question_id']} : faithfulness = {row['faithfulness_score']:.3f}")
    
    print("\nAverage faithfulness score:", f"{float(np.mean(scores)):.3f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG-faithfulness evaluation")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output test JSON")
    args = parser.parse_args()
    main(args.input_path, args.output_path)
