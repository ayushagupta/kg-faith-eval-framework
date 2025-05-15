from groq import Groq
import logging
import json

class GroqClient:
    def __init__(self, config):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.logger = logging.getLogger("groq_logger")
        self.logger.setLevel(logging.INFO)
        self.config = config

        if not self.logger.handlers:
            file_handler = logging.FileHandler("logs/groq_responses.log")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def generate_response(self, instructions, input_text):
        try:
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_text}
                ],
                temperature=getattr(self.config, "TEMPERATURE", 0.2),
                max_tokens=getattr(self.config, "MAX_TOKENS", 1024)
            )
            output_text = response.choices[0].message.content
            self._log_response(instructions, input_text, output_text)
            return output_text
        except Exception as e:
            self.logger.error(f"Groq API call failed: {e}")
            return "Error generating response."

    def generate_json_response(self, instructions, input_text, text_data: dict):
        try:
            user_prompt = f"{input_text}\n\nRespond ONLY in valid JSON format as specified in the schema."
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=getattr(self.config, "TEMPERATURE", 0.2),
                max_tokens=getattr(self.config, "MAX_TOKENS", 1024)
            )
            output_text = response.choices[0].message.content
            try:
                response_data = json.loads(output_text)
            except Exception:
                response_data = {"raw_output": output_text}
            self._log_response(instructions, input_text, output_text)
            return response_data
        except Exception as e:
            self.logger.error(f"Groq API call failed: {e}")
            return "Error generating response."

    def _log_response(self, instructions, input_text, response_text):
        log_data = {
            "model": self.config.MODEL_NAME,
            "instructions": instructions,
            "input_text": input_text,
            "response": response_text
        }
        self.logger.info(json.dumps(log_data, indent=4))
