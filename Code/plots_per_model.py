import json
import difflib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set(style="whitegrid", context="talk", font_scale=1.1)
# suffix = "llama"
# model = "llama"


def plot(suffix):
    with open("datasets/baseline.json") as baseline_file:
        baseline = json.load(baseline_file)
    
    # update file name
    with open(f"datasets/{suffix}/faitheval_output_{suffix}.json") as output_file:
        output = json.load(output_file)
        
    precision_result = []
    recall_result = []
    f1_result = []
    metric = []
    for i in range(len(output)):
        baseline_ctx = baseline[i]["context"]
        output_ctx = output[i]["cot_kg"]
        evalutation_score = metric.append(output[i]["faithfulness_score"])
        
        # print(output[i]["question_id"])
        
        # print(baseline_ctx)
        # print(output_ctx)
        
        precision = 0
        relevant = 0
        # retrieved_baseline = len(baseline_ctx)
        # retrieved_output = len(output_ctx)
        # b_ctx is a tuple
        for b_ctx in baseline_ctx:
            
            found_base = False
            
            # comparing the whole list
            # instead, compare the whole- associates is not good
            # check source, predicate, target individually
            for o_ctx in output_ctx:
                # print(o_ctx, b_ctx)
                sim_0 = difflib.SequenceMatcher(None, o_ctx[0], b_ctx[0]).ratio()
                sim_1 = difflib.SequenceMatcher(None, o_ctx[1], b_ctx[1]).ratio()
                sim_2 = difflib.SequenceMatcher(None, o_ctx[2], b_ctx[2]).ratio()
                
                found_relevant = False
                
                if sim_0 >= 0.6 and sim_1 >= 0.6 and sim_2 >= 0.6:
                    relevant += 1
                    found_relevant = True
                    found_base = True
                    # print(sim_0, sim_1, sim_2)
                    # print(b_ctx, o_ctx)
                    
                if not found_relevant:
                    # swap disease and gene
                    sim_0 = difflib.SequenceMatcher(None, o_ctx[0], b_ctx[2]).ratio()
                    sim_1 = difflib.SequenceMatcher(None, o_ctx[1], b_ctx[1]).ratio()
                    sim_2 = difflib.SequenceMatcher(None, o_ctx[2], b_ctx[0]).ratio()
                    
                    if sim_0 >= 0.6 and sim_1 >= 0.6 and sim_2 >= 0.6:
                        relevant += 1
                        found_base = True
                        # print(b_ctx, o_ctx)
                
                if found_base:
                    break
                
                
                        
        precision = relevant/len(output_ctx)
        recall = relevant/len(baseline_ctx)
        if precision + recall != 0:
            f1 = (2 * precision * recall) / (precision + recall) 
        else:
            f1 = 0
        if recall == 0:
            print(output[i]["question_id"])
            print(i)            
        # print(relevant, retrieved, relevant / retrieved)
        precision_result.append(precision)
        recall_result.append(recall)
        f1_result.append(f1) 
        
    return recall_result, precision_result, f1_result 

# exit()

# scatter plot
# plt.scatter(recall_result, precision_result, edgecolor='black')
# plt.title("Precision Recall Graph")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.grid(True)
# plt.savefig("graphs/mini/precision_recall_scatter_mini.png", dpi=300)
# plt.clf()


def plot_recall_precision_scatter(recall, precision, model, suffix):
    plt.figure(figsize=(8, 6))
    plt.scatter(recall, precision, edgecolor='black', color='steelblue', alpha=0.7, s=50)
    plt.title(f"Precision-Recall Scatter Plot for {model}", fontsize=16, weight='bold')
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig(f"graphs/precision_recall_scatter_{suffix}.png", dpi=300)
    plt.clf()



def plot_f1_score(f1, model, suffix):
    plt.figure(figsize=(8, 6))
    sns.histplot(f1, bins=100, stat='percent', kde=False, color='skyblue', edgecolor='black')
    # Add vertical line for mean
    mean_val = np.mean(f1)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.2f}')
    plt.title(f"Distribution of F1 Scores (out of 1) for {model}")
    plt.xlabel("F1 Score (out of 1)")
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"graphs/f1_score_dist_{suffix}.png", dpi=300)
    plt.clf()
    

def plot_recall(recall, model, suffix):
    plt.figure(figsize=(8, 6))
    sns.histplot(recall, bins=100, stat='percent', kde=False, color='skyblue', edgecolor='black')
    # Add vertical line for mean
    mean_val = np.mean(recall)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.2f}')
    plt.title(f"Distribution of Recall Scores for {model}")
    plt.xlabel("Recall Score (out of 1)")
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"graphs/recall_dist_{suffix}.png", dpi=300)
    plt.clf()
    

def plot_precision(precision, model, suffix):
    plt.figure(figsize=(8, 6))
    sns.histplot(precision, bins=100, stat='percent', kde=False, color='skyblue', edgecolor='black')
    # Add vertical line for mean
    mean_val = np.mean(precision)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.2f}')
    plt.title(f"Distribution of Precision Scores for {model}")
    plt.xlabel("Precision Score (out of 1)")
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"graphs/precision_dist_{suffix}.png", dpi=300)
    plt.clf()
    
def plot_recall_combines(recall_mini, recall_4o, recall_llama):
    # palette = {"Mini": "#1f77b4", "GPT-4o": "#2ca02c", "LLaMA": "#d62728"}
    sns.set(style="whitegrid", context="paper", font_scale=1.2)
    df = pd.DataFrame({
        'Score': recall_llama + recall_mini + recall_4o ,
        'Model': ['Llama-3.1-8b'] * len(recall_llama) + ['GPT-mini'] * len(recall_mini) + ['GPT-4o'] * len(recall_4o)
    })
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Model', y='Score', hue='Model', palette='Set2', inner='quart')
    plt.xlabel("Model")
    plt.ylabel("Recall Score")
    plt.title("Recall Score Distribution by Model")
    plt.savefig("graphs/recall_comparison_violin.png", dpi=300)
    plt.ylim(0, 1)
    plt.clf()    


recall_mini, precision, llama = plot("mini")
recall_4o, _, _ = plot("4o")
recall_llama, _, _ = plot("llama")


suffix = "mini"
model = "GPT-mini"
plot_recall_precision_scatter(recall_mini, precision, model, suffix)

plot_recall_combines(recall_mini, recall_4o, recall_llama)

# print(recall_mini)
# print(recall_4o)
# print(recall_llama)
    