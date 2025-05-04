from config.config import config
from llm.openai_client import OpenAIClient
from spoke.spoke_api_client import SpokeAPIClient
from rag.rag import RAG
from prompts.system_prompts import get_system_prompt
from utils.schema_loader import load_task_schema

openai_client = OpenAIClient()
spoke_api_client = SpokeAPIClient()

# question = "Are there any drugs used for weight management in patients with Bardet-Biedl Syndrome?"
# question = "Out of the given list, which Gene is associated with head and neck cancer and uveal melanoma. Given list is: ABO, CACNA2D1, PSCA, TERT, SULT1B1"
# question = "Out of the given list, which Gene is associated with herpes zoster and psoriatic arthritis. Given list is: HLA-B, ADGRV1, CPS1, SULT1B1, ATG5"
question = "Out of the given list, which Variant is associated with collagenous colitis and autoimmune hepatitis. Given list is: rs2187668, rs6426833, rs6969780, rs3787184, rs230540"
# question = "Out of the given list, which of the following conditions is least likely to be an effect of a mutation in the FLG Gene? Given list: Dermatitis, Esophagitis, Dyspepsia, Asthma"

rag = RAG(openai_client, spoke_api_client, config.CONTEXT_VOLUME, config.QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, config.QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY)
context = rag.retrieve(question)

enriched_prompt = "Context: "+ context + "\n" + "Question: "+ question
system_prompt = get_system_prompt(task="mcq_question_cot")
text_data = load_task_schema(task="mcq_question_cot")
output = openai_client.generate_json_response(instructions=system_prompt, input_text=enriched_prompt, text_data=text_data)

print(enriched_prompt)
print("With reasoning")
print(output)
