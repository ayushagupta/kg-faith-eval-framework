from openai import OpenAI
from config.config import config
import logging
import json

openai_logger = logging.getLogger("openai_logger")
openai_logger.setLevel(logging.INFO)

if not openai_logger.handlers:
    file_handler = logging.FileHandler("logs/llm_responses.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    openai_logger.addHandler(file_handler)


class OpenAIClient:
	def __init__(self):
		self.client = OpenAI(api_key=config.OPENAI_API_KEY)


	def generate_response(self, instructions, input_text):
		try:
			response = self.client.responses.create(
				model = config.MODEL_NAME,
				instructions = instructions,
				input = input_text
			)

			self.log_response(instructions, input_text, response)

			return response.output_text
		
		except Exception as e:
			openai_logger.error(f"OpenAI API call failed: {e}")
			return "Error generating response."
		
			
	def generate_json_response(self, instructions, input_text, text_data: dict):
		try:
			response = self.client.responses.create(
				model = config.MODEL_NAME,
				input = [
					{
						"role": "system",
						"content": instructions
					},
					{
						"role": "user",
						"content": input_text
					}
				],
				text = text_data
			)

			response_text = response.output_text
			response_data = json.loads(response_text)

			self.log_response(instructions, input_text, response)

			return response_data

		except Exception as e:
			openai_logger.error(f"OpenAI API call failed: {e}")
			return "Error generating response."
		
	
	def log_response(self, instructions, input_text, response):
		log_data = {
			"model": config.MODEL_NAME,
			"instructions": instructions,
			"input_text": input_text,
			"response": response.output_text
		}
		openai_logger.info(json.dumps(log_data, indent=4))
		