import argparse
import json

from tqdm import tqdm

from config.config import config
from llm.openai_client import OpenAIClient
from spoke.spoke_api_client import SpokeAPIClient
from rag.rag import RAG
from utils.dataset_loader import managed_load_dataset

parser = argparse.ArgumentParser(description="Test run KG-RAG inference and save output")
parser.add_argument('--output_path', type=str, required=True, help='Path to output test JSON')
args = parser.parse_args()

openai_client = OpenAIClient()
spoke_api_client = SpokeAPIClient()

data = managed_load_dataset()

rag = RAG(openai_client, spoke_api_client, config.CONTEXT_VOLUME, config.QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, config.QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY)

result = []
questions = []

for question in tqdm(data["mcq"]):
    try:
        question_prompt = question["prompt"]
        context, context_tables = rag.retrieve(question_prompt)
        context_tuples = []
        for context_table in context_tables:
            context_tuples.append(list(context_table[['source', 'predicate', 'target']].itertuples(index=False, name=None)))
        
        info = {
            "question_id": question["question_id"],
            "question": question["prompt"],
            "context": context_tuples, 
        }
        
        result.append(info)
    except Exception as e:
        print(f"Error for question: {question['question_id']}{question_prompt}")
        print(e)

print(f"Saving results to {args.output_path}")
with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
