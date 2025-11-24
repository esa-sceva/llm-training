# Benchmark Suite for LLM Training

A comprehensive benchmarking system for evaluating fine-tuned language models on **Satellite Communications (SatCom)** domain-specific tasks. This suite is designed to assess model performance throughout the training process and compare different checkpoints.

## Overview

This benchmark suite enables:
- **Automated evaluation** of multiple training checkpoints
- **Multi-dataset benchmarking** across various SatCom domains
- **GPT-based comparison** between different model versions
- **Progression tracking** with visualizations across training steps
- **MCQA (Multiple Choice QA)** and **Open QA** evaluation support

Built for the **ESA-SCEVA** (SatCom Expert Virtual Assistant) project, this system evaluates models on specialized satellite communications datasets available on [HuggingFace](https://huggingface.co/esa-sceva).

## Folder Structure

```
benchmark/
├── benchmark_vllm_server_v1.py      # Main benchmarking script
├── compare_checkpoints_gpt.py       # Checkpoint comparison tool
├── plot_benchmark_progression.py    # Visualization tool
├── test_vllm_config.yaml            # Example configuration
├── requirements.txt                 # Python dependencies
├── utils/                           # Utility modules
│   ├── __init__.py                  # Package initialization
│   ├── merge_lora.py                # LoRA weight merging
│   └── convert_lit_checkpoint.py    # LitGPT to HF conversion
└── datasets/                        # Benchmark datasets
    ├── sapien_open_qas.json
    ├── sapien_open_problems.json
    ├── sapien_mcqa.json
    ├── sapien_mcqa_problems.json
    ├── eve_mcqa_transformed.json
    └── teleqna_mcqa.json
```

## Available Tools

### 1. `benchmark_vllm_server_v1.py` - Main Benchmarking Script

Evaluates multiple checkpoints using vLLM for efficient inference.

**Features:**
- Uses vLLM library for high-performance inference
- Supports LoRA checkpoint merging
- Automatic memory management between checkpoints
- Multi-dataset evaluation
- GPT-based answer quality assessment
- Comprehensive result logging (JSON, CSV)

**Usage:**
```bash
python benchmark_vllm_server_v1.py --config test_vllm_config.yaml
python benchmark_vllm_server_v1.py --config test_vllm_config.yaml --dry-run  # Test configuration
```

### 2. `compare_checkpoints_gpt.py` - Checkpoint Comparison

Compares two checkpoint results using GPT evaluation to determine which model produces better answers.

**Features:**
- Side-by-side answer comparison
- GPT-based quality judgment
- Statistical analysis with visualizations
- Detailed comparison reports

**Usage:**
```bash
python compare_checkpoints_gpt.py \
    --file1 results/checkpoint_step1.json \
    --file2 results/checkpoint_step2.json \
    --api-key YOUR_OPENAI_API_KEY

# Or use API key from config file
python compare_checkpoints_gpt.py \
    --file1 results/checkpoint_step1.json \
    --file2 results/checkpoint_step2.json \
    --config test_vllm_config.yaml
```

### 3. `plot_benchmark_progression.py` - Visualization Tool

Creates progression plots showing model performance across training checkpoints.

**Features:**
- Bar plots and scatter plots per benchmark
- Color-coded checkpoint types (base, intermediate, final)
- Automated metric visualization
- Publication-ready figures

**Usage:**
```bash
python plot_benchmark_progression.py results/experiment_summary.csv
```

### 4. `utils/merge_lora.py` - LoRA Weight Merging

Merges LoRA adapter weights with the base model for deployment or benchmarking.

**Usage:**
```bash
python -m utils.merge_lora \
    --checkpoint_dir path/to/lora_checkpoint \
    --pretrained_checkpoint_dir path/to/base_model \
    --precision bf16-true
```

Or directly:
```bash
cd utils
python merge_lora.py \
    --checkpoint_dir ../path/to/lora_checkpoint \
    --pretrained_checkpoint_dir ../path/to/base_model \
    --precision bf16-true
```

### 5. `utils/convert_lit_checkpoint.py` - LitGPT Checkpoint Converter

Converts LitGPT checkpoints to HuggingFace format for compatibility with standard tools.

**Usage:**
```bash
python -m utils.convert_lit_checkpoint \
    --checkpoint_dir path/to/litgpt_checkpoint \
    --output_dir path/to/hf_checkpoint
```

Or directly:
```bash
cd utils
python convert_lit_checkpoint.py \
    --checkpoint_dir ../path/to/litgpt_checkpoint \
    --output_dir ../path/to/hf_checkpoint
```

## Dataset Overview

The benchmark suite uses Satellite Communications (SatCom) domain-specific datasets from the [**ESA-SCEVA organization**](https://huggingface.co/esa-sceva) on HuggingFace. These datasets are developed under the ESA ARTES programme and follow open science principles.

### Available Datasets

| Dataset | Type | Description | Samples | Source |
|---------|------|-------------|---------|--------|
| `sapien_open_qas.json` | Open QA | Open-ended questions on satellite communications | ~2.5K | Local (`datasets/`) |
| `sapien_open_problems.json` | Open QA | Problem-solving scenarios | ~750 | Local (`datasets/`) |
| `sapien_mcqa.json` | MCQA | Multiple-choice questions (4 options) | ~3.8K | Local (`datasets/`) |
| `sapien_mcqa_problems.json` | MCQA | Multiple-choice problem scenarios | ~1.6K | Local (`datasets/`) |
| `eve_mcqa_transformed.json` | MCQA | EVE project questions | Variable | Local (`datasets/`) |
| `teleqna_mcqa.json` | MCQA | Telecommunications Q&A | Variable | Local (`datasets/`) |

### Dataset Formats

#### Open QA Format
```json
[
  {
    "question": "What is a link budget?",
    "expected_answer": "A link budget is an analysis that assesses...",
    "Topic": "Link budget"
  }
]
```

#### MCQA Format
```json
{
  "question 0": {
    "question": "Antenna gain increases when the effective area is:",
    "category": "Link Budget",
    "option_1": "Decreased",
    "option_2": "Increased",
    "option_3": "Constant",
    "option_4": "None of these",
    "expected_answer": "option 2: Increased"
  }
}
```

### Using HuggingFace Datasets

**Load datasets from ESA-SCEVA organization:**

```python
from datasets import load_dataset

# For public datasets
dataset = load_dataset("esa-sceva/satcom-qa")

# For gated/private datasets (requires HF token)
dataset = load_dataset("esa-sceva/satcom-synth-qa", token="YOUR_HF_TOKEN")

# Save locally for benchmarking
dataset['train'].to_json("datasets/my_dataset.json")
```

### Available Models

| Model | Description | Size |
|-------|-------------|------|
| [**esa-sceva/llama3-satcom-8b**](https://huggingface.co/esa-sceva/llama3-satcom-8b) | Fine-tuned LLaMA 3.1 8B for SatCom Q&A | 8B params |
| [**esa-sceva/llama3-satcom-70b**](https://huggingface.co/esa-sceva/llama3-satcom-70b) | Large-scale domain-specialized model (LLaMA 3.3-70B) | 70B params |

> **Note:** All datasets and models are freely accessible under permissive licenses (LLama-3.1 or equivalent).

## Configuration

### Basic Configuration File (`test_vllm_config.yaml`)

```yaml
# Base model directory (before fine-tuning)
base_model_dir: /path/to/base/model

# Experiment tracking
experiment_name: satcom_llm_benchmark

# Checkpoint selection strategy
checkpoints:
  selection: specific  # Options: all, specific, pattern
  specific_list:
    - base_model      # Evaluate base model
    - step-000200     # Intermediate checkpoints
    - step-000400
    - final           # Final checkpoint

# OpenAI API for GPT-based evaluation
openai_api:
  api_key: "YOUR_OPENAI_API_KEY"
  model: "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo"
  max_workers: 5

# vLLM inference configuration
vllm_config:
  tensor_parallel_size: 1  # Number of GPUs
  gpu_memory_utilization: 0.9
  max_model_len: 4096
  dtype: "bfloat16"

# Benchmark datasets
benchmarks:
  # Multiple Choice Questions
  - name: sapien_mcqa
    dataset_path: ./datasets/sapien_mcqa.json
    evaluation_type: mcqa_advanced
    num_samples: 1000  # null for all samples
    
    system_prompt: |
      You are SatcomLLM, an advanced AI model developed jointly by Pi School 
      and the European Space Agency (ESA). You are a satellite communications 
      expert specialized in link budgets, propagation, modulation, antennas, 
      and orbital systems.
    
    prompt_template: |
      Question: {question}
      
      Options:
      option 1) {option_1}
      option 2) {option_2}
      option 3) {option_3}
      option 4) {option_4}
      
      Answer with the correct option (option 1, 2, 3 or 4) without explanation:
    
    sampling_config:
      max_tokens: 50
      temperature: 0.1
      top_p: 0.9
  
  # Open-ended Questions with GPT evaluation
  - name: sapien_open_qa
    dataset_path: ./datasets/sapien_open_qas.json
    evaluation_type: open_qa_gpt
    num_samples: 500
    
    system_prompt: |
      You are a satellite communications expert...
    
    prompt_template: |
      Question: {question}
      
      Answer:
    
    sampling_config:
      max_tokens: 512
      temperature: 0.7
      top_p: 0.9
    
    # GPT evaluation prompt
    gpt_eval_config:
      eval_prompt: |
        You are evaluating the quality of an AI answer to a technical question.
        
        Question: {question}
        Expected Answer: {expected_answer}
        Model Answer: {model_answer}
        
        Rate the model answer on a scale of 0-10 considering:
        - Factual correctness
        - Completeness
        - Technical accuracy
        
        Respond with JSON: {{"score": X, "reasoning": "..."}}
```

### Configuration Parameters

#### Checkpoint Selection

```yaml
checkpoints:
  selection: specific  # Options:
    # all - Evaluate all checkpoints in directory
    # specific - Evaluate only listed checkpoints
    # pattern - Use regex pattern (e.g., "step-.*[05]00")
  
  specific_list:
    - base_model
    - step-000200
    - final
```

#### Evaluation Types

| Type | Description | Required Fields |
|------|-------------|-----------------|
| `mcqa_basic` | Simple multiple-choice with exact match | question, options, expected_answer |
| `mcqa_advanced` | MCQA with flexible answer parsing | question, options, expected_answer |
| `open_qa_gpt` | Open QA with GPT-based evaluation | question, expected_answer, gpt_eval_config |

#### Sampling Configuration

```yaml
sampling_config:
  max_tokens: 512        # Maximum tokens to generate
  temperature: 0.1       # Lower = more deterministic (0.0-2.0)
  top_p: 0.9            # Nucleus sampling (0.0-1.0)
  top_k: 50             # Top-k sampling
  stop_tokens:          # Custom stop sequences
    - "\n\nQuestion:"
```

## Quick Start Guide

### 1. Install Dependencies

```bash
cd benchmark
pip install -r requirements.txt

# Install vLLM (required for benchmarking)
pip install vllm

# Optional: For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Prepare Configuration

```bash
# Copy example configuration
cp test_vllm_config.yaml my_benchmark_config.yaml

# Edit configuration with your paths and API keys
nano my_benchmark_config.yaml
```

**Key settings to update:**
- `base_model_dir`: Path to your base model
- `checkpoints_dir`: Path to your training checkpoints (if not using specific_list)
- `openai_api.api_key`: Your OpenAI API key
- Dataset paths in `benchmarks` section

### 3. Run Benchmark

```bash
# Dry run to test configuration
python benchmark_vllm_server_v1.py --config my_benchmark_config.yaml --dry-run

# Full benchmark run
python benchmark_vllm_server_v1.py --config my_benchmark_config.yaml
```

### 4. Analyze Results

Results are saved in timestamped directories:

```
vllm_{experiment_name}_{timestamp}/
├── summary.csv                           # Overall results
├── detailed_results.json                 # All evaluations
├── checkpoint_base_model/
│   ├── sapien_mcqa_results.json
│   └── sapien_open_qa_results.json
├── checkpoint_step-000200/
│   └── ...
└── logs/
    └── benchmark.log
```

### 5. Compare Checkpoints

```bash
# Compare two checkpoints
python compare_checkpoints_gpt.py \
    --file1 results/checkpoint_step-000200/sapien_qa_results.json \
    --file2 results/checkpoint_step-000400/sapien_qa_results.json \
    --api-key YOUR_OPENAI_API_KEY
```

### 6. Visualize Progress

```bash
# Create progression plots
python plot_benchmark_progression.py results/summary.csv
```

This generates plots showing:
- Performance improvement across checkpoints
- Comparison between base model and fine-tuned versions
- Benchmark-specific metrics

## Output Formats

### Summary CSV (`summary.csv`)

| Column | Description |
|--------|-------------|
| `checkpoint_name` | Checkpoint identifier |
| `benchmark_name` | Dataset name |
| `metric_average_score` | Average score across all samples |
| `metric_total_samples` | Number of samples evaluated |
| `metric_correct` | Number of correct answers (MCQA) |
| `metric_accuracy` | Accuracy percentage (MCQA) |
| `timestamp` | Evaluation timestamp |

### Detailed Results JSON

```json
{
  "checkpoint_name": "step-000200",
  "benchmark_name": "sapien_mcqa",
  "config": {...},
  "metrics": {
    "average_score": 0.85,
    "accuracy": 85.0,
    "total_samples": 1000,
    "correct": 850
  },
  "responses": [
    {
      "question": "What is...",
      "expected_answer": "...",
      "model_answer": "...",
      "score": 1.0,
      "evaluation": {...}
    }
  ]
}
```

## Use Cases

### 1. Training Progression Monitoring

Track model improvement during fine-tuning:

```yaml
checkpoints:
  selection: all
  checkpoint_dir: out/finetune/llama-8b/
```

Run after each training milestone to monitor:
- Performance trends
- Overfitting indicators
- Optimal stopping points

### 2. Model Comparison

Compare different fine-tuning runs:

```bash
# Benchmark Run 1
python benchmark_vllm_server_v1.py --config config_run1.yaml

# Benchmark Run 2
python benchmark_vllm_server_v1.py --config config_run2.yaml

# Compare
python compare_checkpoints_gpt.py \
    --file1 run1/final/sapien_qa.json \
    --file2 run2/final/sapien_qa.json
```

### 3. Dataset Evaluation

Evaluate model performance on specific SatCom topics:

```yaml
benchmarks:
  - name: link_budget_eval
    dataset_path: ./datasets/sapien_mcqa.json
    category_filter: "Link Budget"  # Filter by topic
```

## Advanced Usage

### Custom Evaluation Metrics

Extend the benchmarking system with custom evaluators:

```python
# In benchmark_vllm_server_v1.py, add custom evaluation logic:

def evaluate_custom_metric(question, expected, model_answer):
    """Custom evaluation function."""
    # Your custom logic here
    score = calculate_custom_score(expected, model_answer)
    return {
        'score': score,
        'details': {...}
    }
```

### Batch Processing Multiple Experiments

```bash
#!/bin/bash
# benchmark_all.sh

CONFIGS=("config1.yaml" "config2.yaml" "config3.yaml")

for config in "${CONFIGS[@]}"; do
    echo "Running benchmark: $config"
    python benchmark_vllm_server_v1.py --config "$config"
done

# Generate comparison report
python generate_comparison_report.py --results results/
```

### Memory Optimization

For large models or limited GPU memory:

```yaml
vllm_config:
  tensor_parallel_size: 2      # Split across 2 GPUs
  gpu_memory_utilization: 0.8  # Reduce memory usage
  max_model_len: 2048          # Limit context length
  quantization: "awq"          # Use quantization
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```yaml
vllm_config:
  gpu_memory_utilization: 0.7  # Reduce from 0.9
  max_model_len: 2048           # Reduce context
  tensor_parallel_size: 2       # Use multiple GPUs
```

**2. vLLM Import Error**

```bash
pip install vllm
# Or with specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

**3. OpenAI API Rate Limits**

```yaml
openai_api:
  max_workers: 3  # Reduce concurrent requests
  
# Or implement exponential backoff in config
```

**4. Checkpoint Loading Failures**

Ensure checkpoints are properly merged (for LoRA):

```bash
python merge_lora.py \
    --checkpoint_dir path/to/lora_checkpoint \
    --pretrained_checkpoint_dir path/to/base_model
```

**5. Dataset Format Errors**

Validate your dataset format matches expected structure:

```python
import json

# Check MCQA format
with open('dataset.json') as f:
    data = json.load(f)
    for key, item in data.items():
        assert 'question' in item
        assert 'expected_answer' in item
        assert all(f'option_{i}' in item for i in range(1, 5))
```

## References

### ESA-SCEVA Project

- **HuggingFace Organization**: [https://huggingface.co/esa-sceva](https://huggingface.co/esa-sceva)
- **Project Description**: SatCom Expert Virtual Assistant (SCEVA) - Assessment of Open-Source LLMs within the SatCom Sector
- **Programme**: ESA ARTES FP 1A.128 (Connectivity & Secure Communications)
- **Consortium**: Pi School (Lead), RINA Consulting, European Space Agency

### Technical Documentation

- **vLLM Documentation**: [https://docs.vllm.ai/](https://docs.vllm.ai/)
- **LitGPT Repository**: [https://github.com/Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt)
- **HuggingFace Datasets**: [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)

### Research Papers

- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Evaluation Benchmarks**: See ESA-SCEVA publications

## Contributing

This benchmark suite is part of the open-source **ESA-SCEVA** project. Contributions are welcome:

1. Add new evaluation metrics
2. Integrate additional datasets
3. Improve visualization tools
4. Optimize memory usage
5. Enhance documentation

## License

This project uses open-source components under permissive licenses:
- **LitGPT**: Apache License 2.0
- **vLLM**: Apache License 2.0
- **ESA-SCEVA Datasets**: Apache License 2.0 or equivalent

---

**Part of the ESA-SCEVA project: advancing open, reliable, and domain-specialized AI for Europe's space communications sector.**

For questions or support, please refer to the [main repository documentation](../README.md) or visit the [ESA-SCEVA HuggingFace organization](https://huggingface.co/esa-sceva).

