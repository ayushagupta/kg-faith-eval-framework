from openai import OpenAI
# from config.config import config
import logging
import json

class OpenAIClient:
	def __init__(self, config):
		self.client = OpenAI(api_key=config.OPENAI_API_KEY)
		self.logger = logging.getLogger("openai_logger")
		self.logger.setLevel(logging.INFO)
		self.config = config

		if not self.logger.handlers:
			file_handler = logging.FileHandler("logs/llm_responses.log")
			formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
			file_handler.setFormatter(formatter)
			self.logger.addHandler(file_handler)


	def generate_response(self, instructions, input_text):
		try:
			response = self.client.responses.create(
				model = self.config.MODEL_NAME,
				instructions = instructions,
				input = input_text
			)

			self._log_response(instructions, input_text, response)

			return response.output_text
		
		except Exception as e:
			self.logger.error(f"OpenAI API call failed: {e}")
			return "Error generating response."
		
			
	def generate_json_response(self, instructions, input_text, text_data: dict):
		try:
			response = self.client.responses.create(
				model = self.config.MODEL_NAME,
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

			self._log_response(instructions, input_text, response)

			return response_data

		except Exception as e:
			self.logger.error(f"OpenAI API call failed: {e}")
			return "Error generating response."
		
	
	def _log_response(self, instructions, input_text, response):
		log_data = {
			"model": self.config.MODEL_NAME,
			"instructions": instructions,
			"input_text": input_text,
			"response": response.output_text
		}
		self.logger.info(json.dumps(log_data, indent=4))
		