import argparse
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from faitheval.faithfulness import score_record
from faitheval.logging_config import logger


def main(input_path, output_path, hallucination_log_path):
    data = json.load(open(input_path, "r", encoding="utf-8"))
    scores = []
    hallucination_details = defaultdict(list)

    for row in tqdm(data, desc="Processing rows"):
        current_q_id = row['question_id']
        logger.info(f"*** Processing question {current_q_id} ***")
        raw_score, hallucinations_for_row = score_record(row)
        score = round(raw_score, 3)
        row["faithfulness_score"] = score
        scores.append(score)
        hallucination_details[current_q_id] = hallucinations_for_row
        logger.info(f"{row['question_id']} : faithfulness = {row['faithfulness_score']:.3f}")
    
    print("\nAverage faithfulness score:", f"{float(np.mean(scores)):.3f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    if hallucination_details:
        with open(hallucination_log_path, "w",  encoding="utf-8") as f:
            json.dump(hallucination_details, f, indent=4, ensure_ascii=False)
        print(f"Hallucination details logged to: {hallucination_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG-faithfulness evaluation")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--hallucination_log_path", type=str, required=True, help="Path to hallucination details JSON file")

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.hallucination_log_path)
