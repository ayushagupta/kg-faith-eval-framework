from config.config import config
from llm.openai_client import OpenAIClient
from spoke.spoke_api_client import SpokeAPIClient
from rag.rag import RAG
from prompts.system_prompts import get_system_prompt
from utils.schema_loader import load_task_schema
from utils.dataset_loader import managed_load_dataset

openai_client = OpenAIClient()
spoke_api_client = SpokeAPIClient()

data = managed_load_dataset(data_len=2)

rag = RAG(openai_client, spoke_api_client, config.CONTEXT_VOLUME, config.QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, config.QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY)

result = []

for question in data["mcq"]:
    question_prompt = question["prompt"]
    context = rag.retrieve(question_prompt)

    enriched_prompt = "Context: "+ context + "\n" + "Question: "+ question_prompt
    system_prompt = get_system_prompt(task="mcq_question_cot_1")
    text_data = load_task_schema(task="mcq_question_cot")
    output = openai_client.generate_json_response(instructions=system_prompt, input_text=enriched_prompt, text_data=text_data)
    
    final_output = {
        "question_id": question["question_id"],
        "question": question_prompt,
        "chain_of_thought": output["reasoning"],
        "correct_answer": question["correct_answer"],
        "model_answer": output["answer"]
    }
    
    result.append(final_output)
    
print(result[0])

