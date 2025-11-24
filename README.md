# llm-training

A streamlined repository for fine-tuning large language models using **[LitGPT](https://github.com/Lightning-AI/litgpt)** - a high-performance library by Lightning AI for training and deploying 20+ LLMs.

> **Part of the [ESA-SCEVA](https://huggingface.co/esa-sceva) project** - Developed under the **European Space Agency (ESA) ARTES programme** for advancing open, domain-specialized AI in Satellite Communications.

## Features

- **Easy Setup**: Simple configuration using `.env` files and Makefile commands
- **Multiple Fine-tuning Methods**: Support for LoRA (memory-efficient) and full fine-tuning
- **Production Ready**: Built on LitGPT's battle-tested training infrastructure
- **Flexible Configuration**: YAML-based configuration for all hyperparameters
- **Experiment Tracking**: Built-in Weights & Biases integration
- **Memory Efficient**: Support for LoRA, QLoRA, and quantization techniques
- **Multi-GPU Support**: Distributed training with Lightning Fabric

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- HuggingFace account and access token ([Get one here](https://huggingface.co/settings/tokens))
- (Optional) Weights & Biases account for experiment tracking

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/esa-satcomllm/llm-training.git
cd llm-training
```

### 2. Configure Environment Variables

Create a `.env` file in the repository root:

```bash
JSON_PATH=./dataset.json
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
HF_TOKEN=<YOUR_HF_TOKEN>
CONFIG_PATH=config/lora.yaml
WANDB_TOKEN=<YOUR_WANDB_TOKEN>
WANDB_NAME=llama-8B
VAL_SPLIT=0.05
OUT_DIR=out/finetune/meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Configuration Parameters:**

| Parameter | Description |
|-----------|-------------|
| `JSON_PATH` | Path to your Alpaca-formatted dataset |
| `MODEL_NAME` | HuggingFace model identifier (see [Available Models](#available-models)) |
| `HF_TOKEN` | Your HuggingFace access token |
| `CONFIG_PATH` | Training configuration file (`config/lora.yaml` or `config/full.yaml`) |
| `WANDB_TOKEN` | (Optional) Weights & Biases API token |
| `WANDB_NAME` | (Optional) Experiment name for W&B |
| `VAL_SPLIT` | Validation split fraction (e.g., 0.05 = 5%) |
| `OUT_DIR` | Output directory for checkpoints |

### 3. Prepare Your Dataset

Your dataset must be in **Alpaca format** (JSON):

```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  },
  {
    "instruction": "Translate the following to Spanish:",
    "input": "Hello, how are you?",
    "output": "Hola, ¿cómo estás?"
  }
]
```

Save your dataset to the path specified in `JSON_PATH`.

**Need a dataset?** Check out these HuggingFace datasets:
- `yahma/alpaca-cleaned`
- `esa-sceva/satcom-synth-qa`
- Or create your own following the format above

### 4. Install Dependencies

```bash
make install
```

This creates a virtual environment and installs required packages.

### 5. Download Model

```bash
make model
```

Downloads the pre-trained model specified in your `.env` file.

### 6. (Optional) Setup Weights & Biases

```bash
make wandb
```

### 7. Start Training

```bash
make train
```

Training will begin using the configuration specified in `CONFIG_PATH`. Checkpoints will be saved to `OUT_DIR`.

## Available Models

To see all supported models:

```bash
litgpt download list
```

**Popular models:**
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `mistralai/Mistral-Small-24B-Instruct-2501`
- `microsoft/phi-2`
- `google/gemma-2b`

## Configuration

### LoRA vs Full Fine-tuning

The repository supports two fine-tuning approaches:

#### 1. LoRA (Low-Rank Adaptation) - **Recommended**

**Config:** `config/lora.yaml`

**Advantages:**
- Requires much less GPU memory (2-8GB)
- Faster training
- Smaller checkpoint files (~100MB)
- Often achieves similar performance to full fine-tuning

**Trade-offs:**
- Only trains adapter weights (not the full model)

**Best for:** Most use cases, limited GPU memory, rapid iteration

#### 2. Full Fine-tuning

**Config:** `config/full.yaml`

**Advantages:**
- Potentially better task-specific performance
- Maximum model adaptation flexibility

**Trade-offs:**
- Requires significant GPU memory (24GB+)
- Slower training
- Large checkpoint files (several GBs)

**Best for:** High-end GPUs, maximum performance requirements

### Key Configuration Parameters

#### LoRA Parameters (`config/lora.yaml`)

```yaml
# LoRA configuration
lora_r: 32              # LoRA rank (8-64 typical)
lora_alpha: 16          # Scaling factor (usually lora_r or 2*lora_r)
lora_dropout: 0.05      # Dropout rate

# Training configuration
train:
  epochs: 2             # Number of training epochs
  global_batch_size: 128
  micro_batch_size: 4   # Decrease if out of memory
  max_seq_length: 512   # Maximum token length
  save_interval: 200    # Save checkpoint every N steps
  
# Optimizer
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.0001          # Learning rate
    weight_decay: 0.0
    
# Hardware
devices: 1              # Number of GPUs
precision: bf16-true    # Training precision
quantize: null          # Optional: nf4, fp4, int8-training
```

### Common Adjustments

| Problem | Solution |
|---------|----------|
| **Out of GPU memory** | ↓ Decrease `micro_batch_size`, `max_seq_length`, or `lora_r` |
| **Underfitting** | ↑ Increase `epochs`, `lora_r`, or `lr` |
| **Overfitting** | ↓ Decrease `epochs`, increase `lora_dropout` or `weight_decay` |
| **Training too slow** | ↑ Increase `micro_batch_size` or decrease `save_interval` |
| **Unstable training** | ↓ Decrease `lr` or increase `lr_warmup_steps` |
| **Need longer context** | ↑ Increase `max_seq_length` (uses more memory) |

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Create virtual environment and install dependencies |
| `make model` | Download the pre-trained model from HuggingFace |
| `make wandb` | Login to Weights & Biases |
| `make train` | Start fine-tuning with current configuration |
| `make evaluate` | Evaluate the fine-tuned model on benchmarks |
| `make all` | Run complete pipeline (install → model → train) |

## Evaluation

### Standard Benchmarks

After training, evaluate your model on standard benchmarks:

```bash
make evaluate
```

This runs evaluation on:
- HellaSwag (commonsense reasoning)
- TruthfulQA (truthfulness)
- MMLU (multitask language understanding)

Results are saved to `evaluate_model/`.

### SatCom-Specific Benchmarks

For comprehensive evaluation on **Satellite Communications domain-specific tasks**, use the benchmark suite in the [`benchmark/`](benchmark/) directory:

```bash
cd benchmark
python benchmark_vllm_server_v1.py --config test_vllm_config.yaml
```

**Features:**
- Evaluate multiple checkpoints automatically
- Domain-specific SatCom datasets (Link Budget, Propagation, etc.)
- GPT-based answer quality assessment
- Progression tracking with visualizations
- Multiple-choice and open-ended question support

**Available Datasets:**
- `sapien_mcqa` - Multiple-choice questions (3.8K samples)
- `sapien_open_qas` - Open-ended questions (2.5K samples)
- [ESA-SCEVA datasets on HuggingFace](https://huggingface.co/esa-sceva) - Official SatCom datasets

See the [**Benchmark README**](benchmark/README.md) for detailed instructions and configuration options.

## Using the Fine-tuned Model

### Interactive Chat

```bash
litgpt chat out/finetune/meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Generate Text

```bash
litgpt generate out/finetune/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --prompt "What is machine learning?"
```

### Deploy as API Server

```bash
litgpt serve out/finetune/meta-llama/Meta-Llama-3.1-8B-Instruct
```

## Tutorial Notebook

For a detailed step-by-step guide, see [`model_finetuning_guide.ipynb`](model_finetuning_guide.ipynb), which covers:
- Downloading datasets from HuggingFace
- Converting datasets to Alpaca format
- Understanding configuration parameters
- Monitoring training progress
- Best practices and tips

## Advanced Usage

### Custom Configuration

To create a custom configuration:

```bash
cp config/lora.yaml config/my_config.yaml
# Edit my_config.yaml with your parameters
```

Update `.env`:
```bash
CONFIG_PATH=config/my_config.yaml
```

### Multi-GPU Training

In your config file, set:

```yaml
devices: 4              # Use 4 GPUs
num_nodes: 1            # Single machine
```

### Quantization (Further Memory Reduction)

In your config file:

```yaml
quantize: nf4           # Options: nf4, nf4-dq, fp4, fp4-dq, int8-training
```

### Merging LoRA Weights

After LoRA training, merge adapters with base model:

```bash
litgpt merge_lora \
    --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --lora_path out/finetune/meta-llama/Meta-Llama-3.1-8B-Instruct/final \
    --out_dir merged_model/
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `micro_batch_size` in config
- Reduce `max_seq_length`
- Enable quantization (`quantize: nf4`)
- Use a smaller model

**2. HuggingFace Authentication Error**
- Ensure `HF_TOKEN` is set correctly in `.env`
- Check token has access to gated models (like Llama)
- Visit the model page and accept license terms

**3. Validation Loss Not Decreasing**
- Increase learning rate
- Train for more epochs
- Increase `lora_r` for more capacity
- Check dataset quality

**4. Training Too Slow**
- Increase `micro_batch_size` (if memory allows)
- Use multiple GPUs
- Enable bf16 precision: `precision: bf16-true`

## Resources

### LitGPT Documentation
- [Official LitGPT Repository](https://github.com/Lightning-AI/litgpt)
- [LoRA Fine-tuning Tutorial](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/finetune_lora.md)
- [Quantization Guide](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md)
- [All LitGPT Tutorials](https://github.com/Lightning-AI/litgpt/tree/main/tutorials)

### Papers & References
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project uses [LitGPT](https://github.com/Lightning-AI/litgpt) which is licensed under the Apache License 2.0.

---

**Powered by [LitGPT](https://github.com/Lightning-AI/litgpt)**

For detailed tutorials and examples, explore the [`tutorials/` folder](https://github.com/Lightning-AI/litgpt/tree/main/tutorials) in the LitGPT repository.
