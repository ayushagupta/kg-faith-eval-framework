# LLM Reasoning Benchmark

A comprehensive benchmark suite designed to evaluate and analyze the reasoning capabilities of Knowledge Graph augmented Large Language Models (LLMs).

## Overview
The problem we tackle is assessing whether an LLM’s multi-step reasoning is correctly grounded in a biomedical knowledge graph, thereby identifying hallucinations or misuse of knowledge even if the final answer appears plausible. Our work produced a complete pipeline that not only helps a large language model answer questions with the aid of a knowledge graph, but also scrutinizes the model’s reasoning for faithfulness to that graph.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ayushagupta/LLM-Reasoning-Benchmark.git
cd LLM-Reasoning-Benchmark
```

2. Create and activate environemnt:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Vector store creation
```bash
python setup.py
```

## Usage

### Knowlede graph RAG

Retrieve the context for each question in the BioMixQA dataset from the SPOKE knowledge graph and send the question along with context to an LLM to produce an answer and the chain-of-thought behind the answer generation.

```bash
python -m test.py --out <path-to-output-file>
```

### Chain-of-Thought to knowledge graph transformation

Convert the chain-of-thought response into a knowledge graph to extract useful tiplets.

```bash
python -m cot2kg.main --in <path_to_input_file> --out <path_to_output_file>
```
Input file must be in expected JSON format - see `LLM-Reasoning-Benchmark/sample_data/cot2kg_example_input.json`

### Faithfulness evaluation

Calculate the faithfulness metric for all data points in the 

```bash
python -m faitheval.evaluate --input_path <path_to_input_file> --output_path <path_to_output_file> --hallucination_log_path <path_to_output_hallucination_log_json_file>
```
Input file must be in expected JSON format - see `LLM-Reasoning-Benchmark/sample_data/faitheval_example_input.json`