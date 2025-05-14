import json
import difflib
import matplotlib.pyplot as plt

with open("datasets/baseline.json") as baseline_file:
    baseline = json.load(baseline_file)
    
# update file name
with open("../../faitheval_output_filtered.json") as output_file:
    output = json.load(output_file)
    
precision_result = []

for i in range(len(output)):
# for i in range(14, 15):
    baseline_ctx = baseline[i]["context"]
    output_ctx = output[i]["cot_kg"]
    
    # print(output[i]["question_id"])
    
    # print(baseline_ctx)
    # print(output_ctx)
    
    precision = 0
    relevant = 0
    retrieved = len(baseline_ctx)
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
            
            if sim_0 >= 0.9 and sim_1 >= 0.9 and sim_2 >= 0.9:
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
                
                if sim_0 >= 0.9 and sim_1 >= 0.9 and sim_2 >= 0.9:
                    relevant += 1
                    found_base = True
                    # print(b_ctx, o_ctx)
            
            if found_base:
                break
            
            
                    
    precision = relevant//retrieved
    # if precision < 1:
        # print(i, relevant, retrieved)
            
    print(relevant, retrieved, relevant / retrieved)
    precision_result.append(relevant/retrieved)        


# data = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5]

# Plot histogram
plt.hist(precision_result, bins=5, edgecolor='black')
plt.title("Distribution of Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("graphs/precison_dist_mini.png", dpi=300)  
# plt.show()
    