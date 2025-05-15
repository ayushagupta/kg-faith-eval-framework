import json
import difflib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

sns.set(style="whitegrid", context="talk", font_scale=1.1)
sns.set(style="whitegrid", context="paper", font_scale=1.2)
suffix = "llama"
model = "Llama-3.1-8b"

with open("datasets/baseline.json") as baseline_file:
    baseline = json.load(baseline_file)
    
# update file name
with open(f"datasets/mini/faitheval_output_mini.json") as output_file:
    output_mini = json.load(output_file)
    

# update file name
with open(f"datasets/4o/faitheval_output_4o.json") as output_file:
    output_4o = json.load(output_file)
    

# update file name
with open(f"datasets/llama/faitheval_output_llama.json") as output_file:
    output_llama = json.load(output_file)
    
score_mini = [score["faithfulness_score"] for score in output_mini]
score_4o = [score["faithfulness_score"] for score in output_4o]
score_llama = [score["faithfulness_score"] for score in output_llama]

# print(score_4o)
# print(score_llama)
# print(score_mini)

# plt.figure(figsize=(8,6))
# sns.histplot(score_mini, color="blue", label="GPT-mini", bins=10, multiple='dodge', stat='percent', alpha=0.5, edgecolor='black')
# sns.histplot(score_4o, color="green", label="GPT-4o", bins=10, multiple='dodge', stat='percent', alpha=0.5, edgecolor='black')
# sns.histplot(score_llama, color="red", label="Llama-3.1-8b", bins=10, multiple='dodge', stat='percent', alpha=0.5, edgecolor='black')

# plt.xlabel("Value")
# plt.ylabel("Percentage (%)")
# plt.title("Overlaid Histograms")
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"graphs/reasoning_score_all.png", dpi=300)
# plt.clf()

# Combine into a long-form DataFrame for seaborn
df = pd.DataFrame({
    'Score': score_llama + score_mini + score_4o,
    'Model': (['Llama-3.1-8b'] * len(score_llama) +
                ['GPT-mini'] * len(score_mini) + 
                ['GPT-4o'] * len(score_4o)
            )
})

# Set scientific/clean style
sns.set(style="whitegrid", context="paper", font_scale=1.2)

# Plot
plt.figure(figsize=(10, 6))
palette = {"Mini": "#1f77b4", "GPT-4o": "#2ca02c", "LLaMA": "#d62728"}

# Plot histograms with slight transparency
# sns.histplot(data=df, x="Score", hue="Model", stat="percent", bins=15,
            #  element="step", multiple="dodge", fill=True, palette=palette, alpha=0.4, edgecolor=None)
# plt.figure(figsize=(10, 6))
# bins = np.arange(0, 1, 0.20)  
plot = sns.histplot(data=df, x='Score', hue='Model', stat="probability", bins=10, kde=False, multiple='dodge', palette='Set2', edgecolor='black')
# plot = sns.histplot(data=df, x='Score', hue='Model', bins=bins, kde=False, multiple='dodge', palette='Set2', edgecolor='black')

# Axis labels and title
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.xlabel("Faithfulness Score")
plt.ylabel("Percentage (%)")
plt.title("Faithfulness Score Distribution by Model")
plt.ylim(0, plt.ylim()[1] * 3)
for patch in plot.patches:
    height = patch.get_height()
    patch.set_height(height * 3)

# Format
# plt.legend(title="Model", loc="upper left")
legend_patches = [
    mpatches.Patch(color=sns.color_palette("Set2")[2], label='Llama-3.1-8b'),
    mpatches.Patch(color=sns.color_palette("Set2")[0], label='GPT-mini'),
    mpatches.Patch(color=sns.color_palette("Set2")[1], label='GPT-4o'),
    
]
# plt.legend(handles=handles, labels=labels, title="Model", loc="upper left", fontsize="x-large")
plt.legend(handles=legend_patches, title="Model", loc="upper left", fontsize="x-large")

plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save
plt.savefig("graphs/score_comparison_histogram.png", dpi=300)


plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Model', y='Score', hue='Model', palette='Set2', inner='point')
plt.xlabel("Model")
plt.ylabel("Faithfulness Score")
plt.axhline(y=0.3, color='red', linestyle='dotted', linewidth=1.5, label="MIN_LINK_SCORE=0.3")
plt.text(x=0.2, y=0.35, s="MIN_LINK_SCORE=0.3", color='black', rotation=0, va='top', ha='left', fontsize=10)
plt.title("Faithfulness Score Distribution by Model")
plt.ylim(0, 1)
plt.savefig("graphs/score_comparison_violin.png", dpi=300)
plt.clf()
