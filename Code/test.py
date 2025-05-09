import argparse
import json

from tqdm import tqdm

from config.config import config
from llm.openai_client import OpenAIClient
from spoke.spoke_api_client import SpokeAPIClient
from rag.rag import RAG
from prompts.system_prompts import get_system_prompt
from utils.schema_loader import load_task_schema
from utils.dataset_loader import managed_load_dataset

parser = argparse.ArgumentParser(description="Test run KG-RAG inference and save output")
parser.add_argument('--output_path', type=str, required=True, help='Path to output test JSON')
args = parser.parse_args()

openai_client = OpenAIClient()
spoke_api_client = SpokeAPIClient()

data = managed_load_dataset(data_len=3)

rag = RAG(openai_client, spoke_api_client, config.CONTEXT_VOLUME, config.QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, config.QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY)

result = []

for question in tqdm(data["mcq"][2:3]):
    question_prompt = question["prompt"]
    context, context_tables = rag.retrieve(question_prompt)
    # print(f"Context: {context}")
    # print(f"Context table: {context_table}")
    context_tuples = []
    for context_table in context_tables:
        print(f"Context table: {context_table.shape}")
        context_tuples.append(list(context_table[['source', 'predicate', 'target']].itertuples(index=False, name=None)))

    enriched_prompt = "Context: "+ context + "\n" + "Question: "+ question_prompt
    system_prompt = get_system_prompt(task="mcq_question_cot_1")
    text_data = load_task_schema(task="mcq_question_cot")
    output = openai_client.generate_json_response(instructions=system_prompt, input_text=enriched_prompt, text_data=text_data)
    
    final_output = {
        "question_id": question["question_id"],
        "question": question_prompt,
        "chain_of_thought": output["reasoning"],
        "correct_answer": question["correct_answer"],
        "model_answer": output["answer"],
        "kg_rag": context_tuples
    }
    result.append(final_output)
    
print(result[0])

print(f"Saving results to {args.output_path}")
with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
