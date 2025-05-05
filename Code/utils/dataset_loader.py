from datasets import load_dataset
from itertools import islice

def managed_load_dataset(data_len = None):
    """
    Loads the MCQ and True/False question
    
    param data_len: the length of dataset 
    
    returns: dictionary with key: mcq, true/false -> list of dictionaries -> question prompt, correct answer, quesiton id 
    """
    data = {"mcq": [], "tf": []}
    mcq_dataset = load_dataset("kg-rag/BiomixQA", "mcq")["train"]
    tf_dataset = load_dataset("kg-rag/BiomixQA", "true_false")["train"]
        
    for i, question in enumerate(islice(mcq_dataset, data_len)):
        info = {
            "question_id": i,
            "prompt": question["text"],
            "correct_answer": question["correct_answer"],
        }
        data["mcq"].append(info)

        
    for i, question in enumerate(islice(tf_dataset, data_len)):
        info = {
            "question_id": i,
            "prompt": f"Answer whether the following statement is true or false: {question['text']}",
            "correct_answer": question["label"],
        }
        data["tf"].append(info)

    return data