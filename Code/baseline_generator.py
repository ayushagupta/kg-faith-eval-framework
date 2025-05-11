import argparse
import json
import re

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Test run KG-RAG inference and save output")
parser.add_argument('--output_path', type=str, required=False, help='Path to output test JSON')
args = parser.parse_args()

final_output = []

with open("LLM-Reasoning-Benchmark/Code/datasets/context_tuples_mcq.json", "r") as f:
    all_data = json.load(f)
        
    for data in all_data:
    
        qs = data["question"]
        ctx = data["context"]
        options = [word.strip() for word in qs.split("Given list is:")[1].split(",")]
        print(data["question_id"], qs, options)
        
        entities = re.findall(r'Out of the given list, which \w+ is associated with (.*?) and (.*?)\.', qs)[0]
        disease1 = entities[0]
        disease2 = entities[1]
            
        gene_disease_relation = {key: [] for key in options}  # All values set to 0
        
        cot_kg_input = []
            
        for node in ctx:
            if "Gene" in qs:
                entity_type = "Gene"
            elif "Variant" in qs:
                entity_type = "Variant"
            for triples in node:
                disease = re.findall(r'Disease\s+(.*)', triples[0])
                gene_variant = re.findall(fr'{entity_type}\s+(.*)', triples[-1])            
                
                if disease and gene_variant:
                    if gene_variant[0] in options:
                        gene_disease_relation[gene_variant[0]].append(disease[0])
                        
        valid = False
                
        for gene, diseases in gene_disease_relation.items():
            if (len(diseases) == 2):
                valid = True
            if diseases:
                for disease in diseases:
                    cot_kg_input.append([gene, "ASSOCIATES", disease])
                
                if len(diseases) == 1:
                    if diseases[0] == disease1: 
                        cot_kg_input.append([gene, "DOES NOT ASSOICATE", disease2])
                    elif diseases[0] == disease2:
                        cot_kg_input.append([gene, "DOES NOT ASSOCIATE", disease1])
                
        print(gene_disease_relation)
        print(cot_kg_input)
        
        # valid = False
        
        # # determine if info in context
        # for key
         
        
        result = {
            "question_id": data["question_id"],
            "question": qs,
            "context": cot_kg_input
        }
        
        if valid:
            final_output.append(result)
            
        print(len(final_output))
    

print(f"Saving results to {args.output_path}")
with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)
