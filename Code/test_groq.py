import argparse
import json

from tqdm import tqdm
import traceback

from config.config import config_llama, config_mini
from llm.openai_client import OpenAIClient
from llm.groq_client import GroqClient
from spoke.spoke_api_client import SpokeAPIClient
from rag.rag import RAG
from prompts.system_prompts import get_system_prompt
from utils.schema_loader import load_task_schema
from utils.dataset_loader import managed_load_dataset
import logging

logger = logging.getLogger("retrieval_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("logs/retrieval_groq.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

parser = argparse.ArgumentParser(description="Test run KG-RAG inference and save output (Groq)")
parser.add_argument('--output_path', type=str, required=True, help='Path to output test JSON')
args = parser.parse_args()

extraction_client = OpenAIClient(config=config_mini)
inference_client = GroqClient(config=config_llama)
spoke_api_client = SpokeAPIClient()

data = managed_load_dataset()
# only run for questions that can be tested against baseline
with open("datasets/baseline.json") as file:
    baseline = json.load(file)
    baseline_qs_id = [info["question_id"] for info in baseline]
    filtered_questions = [qs for qs in data["mcq"] if qs["question_id"] in baseline_qs_id ]

rag = RAG(extraction_client, spoke_api_client, config_llama.CONTEXT_VOLUME, config_llama.QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, config_llama.QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY)

result = []

for question in tqdm(filtered_questions):
    try:
        question_prompt = question["prompt"]
        context, context_tables = rag.retrieve(question_prompt)
        context_tuples = []
        for context_table in context_tables:
            context_tuples.extend(list(context_table[['source', 'predicate', 'target']].itertuples(index=False, name=None)))

        enriched_prompt = "Context: "+ context + "\n" + "Question: "+ question_prompt
        system_prompt = get_system_prompt(task="mcq_question_cot_1")
        text_data = load_task_schema(task="mcq_question_cot")
        output = inference_client.generate_json_response(instructions=system_prompt, input_text=enriched_prompt, text_data=text_data)
        
        final_output = {
            "question_id": question["question_id"],
            "question": question_prompt,
            "chain_of_thought": output.get("reasoning", ""),
            "correct_answer": question["correct_answer"],
            "model_answer": output.get("answer", ""),
            "kg_rag": context_tuples
        }
        result.append(final_output)
    except Exception as e:
        print(f"Error occured: {question['question_id']} - {question_prompt}: {e}")
        logger.info(f"Error occured: {question['question_id']} - {question_prompt}: {e}")
        traceback.print_exc()
    

print(f"Saving results to {args.output_path}")
with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
