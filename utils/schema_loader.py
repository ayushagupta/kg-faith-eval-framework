import json
import logging

schema_loader_logger = logging.getLogger("schema_loader_logger")
schema_loader_logger.setLevel(logging.INFO)

if not schema_loader_logger.handlers:
    file_handler = logging.FileHandler("logs/schema_loader.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    schema_loader_logger.addHandler(file_handler)


def load_task_schema(task: str):
    try:
        with open("schemas/task_schemas.json", "r") as file:
            task_schemas = json.load(file)
        
        if task in task_schemas:
            return task_schemas[task]
        else:
            schema_loader_logger.error(f"Schema for task '{task}' not found.")
            return None
        
    except FileNotFoundError:
        schema_loader_logger.error("Schema file 'task_schemas.json' not found.")
        return None
    
    except json.JSONDecodeError:
        schema_loader_logger.error("Error decoding the JSON schema file.")
        return None
    
    except Exception as e:
        schema_loader_logger.error(f"Unexpected error while loading task schema: {e}")
        return None
    