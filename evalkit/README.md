# Evalkit

This repository contains the evaluation tasks and metrics for evaluating models on Earth Observation specific benchmarks.

Evalkit is built on top of the [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), a unified framework for testing generative language models on a large number of different evaluation tasks.

For more information refer to the complete guide of the project: [EVE Guide](eve-esa.github.io/eve-guide/).

## Earth Virtual Expert (EVE)

**Earth Virtual Expert (EVE)** aims to advance the use of Large Language Models (LLMs) within the Earth Observation (EO) and Earth Science (ES) community.

- Website: https://eve.philab.esa.int/  
- HuggingFace: https://huggingface.co/eve-esa
- Other repositories: https://github.com/eve-esa
  
## Installation

### 1. Clone the repository

```bash
git clone -b dev https://github.com/eve-esa/eve-evaluation.git
cd eve-evaluation
```

### 2. Set up Python environment

We recommend using Python 3.10 or higher. Install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment:

```bash
uv venv
source .venv/bin/activate  
```

### 3. Install the package

Install the package and all dependencies using uv:

```bash
uv sync
```

This will install:
- The EVE evaluation framework with all custom tasks and metrics
- The lm-evaluation-harness framework
- All required dependencies (PyTorch, transformers, datasets, etc.)

## Usage

### Running Evaluations with Configuration File

The recommended way to run evaluations is using the YAML configuration file `scripts/config/evals.yaml`. This allows you to configure multiple models, tasks, and evaluation parameters in one place.

#### Configuration File Structure

The `evals.yaml` file has the following structure:

```yaml
constants:
  hf_token: your-huggingface-token
  judge_api_key: your-judge-api-key
  judge_base_url: https://api.mistral.ai/v1  # Base URL only
  judge_name: mistral-medium-latest

  # Define tasks with different model types
  tasks:
    # MCQA tasks using local-completions (non-chat API)
    - name: mcqa_single_answer_0_shot
      task_name: mcqa_single_answer
      model_type: local-completions  # Uses /completions endpoint
      apply_chat_template: false
      num_fewshot: 0
      max_tokens: 5
      batch_size: 8

    - name: mcqa_multiple_answer_2_shot
      task_name: mcqa_multiple_answer
      model_type: local-completions  # Uses /completions endpoint
      apply_chat_template: false
      num_fewshot: 2
      max_tokens: 10
      batch_size: 8

    # Open-ended tasks using openai-chat-completions (chat API)
    - name: open_ended_0_shot
      task_name: open_ended
      model_type: openai-chat-completions  # Uses /chat/completions endpoint
      apply_chat_template: false  # API handles templates
      num_fewshot: 0
      max_tokens: 10000
      batch_size: 1
      judge_api_key: !ref judge_api_key
      judge_base_url: !ref judge_base_url
      judge_name: !ref judge_name

    - name: hallucination_detection_5_shot
      task_name: hallucination_detection
      model_type: openai-chat-completions  # Uses /chat/completions endpoint
      apply_chat_template: false
      num_fewshot: 5
      max_tokens: 100
      batch_size: 4

wandb:
  enabled: true
  project: eve-evaluations
  entity: your-wandb-entity
  run_name: my-evaluation-run
  api_key: your-wandb-api-key

models:
  - name: mistralai/mistral-small-3.2-24b-instruct
    base_url: https://openrouter.ai/api/v1  # Base URL only (no /chat/completions)
    api_key: sk-or-v1-...
    temperature: 0.1
    num_concurrent: 15
    timeout: 180
    tokenizer: mistralai/Mistral-7B-Instruct-v0.2  # Optional tokenizer
    tasks: !ref tasks

output_dir: evals_outputs
```

#### Configuration Options

**Constants Section:**
- Define reusable values that can be referenced throughout the config using `!ref`
- Common constants: API keys, base URLs, model names, task lists

**Tasks Configuration:**
- `name`: User-defined task configuration name (used for output folders and logging)
- `task_name`: Actual lm-eval task name (defaults to `name` if not provided)
- `model_type`: Model API type - determines which API interface to use (default: `"openai-chat-completions"`)
  - `openai-chat-completions`: OpenAI-compatible chat API (uses `/chat/completions` endpoint)
  - `local-completions`: Local completions API (uses `/completions` endpoint)
  - `anthropic`: Anthropic API
  - `vllm`: vLLM API
  - Any other lm-eval supported model type
- `apply_chat_template`: Whether to apply chat template locally (default: `true`)
  - Set to `false` for API models that handle chat templates internally
  - Set to `true` to apply chat templates locally using the tokenizer
- `num_fewshot`: Number of few-shot examples (default: 0)
- `max_tokens`: Maximum tokens for model generation (default: 512)
- `temperature`: Sampling temperature for this task (default: 0.0, uses model temperature if > 0)
- `limit`: Optional limit on number of samples to evaluate
- `judge_api_key`: API key for LLM-as-judge evaluation (if applicable)
- `judge_base_url`: Base URL for judge model API
- `judge_name`: Name/ID of judge model
- `judges`: List of multiple judge models for multi-judge evaluation (optional)
  - Each judge has: `name`, `model`, `api_key`, `base_url`
  - Enables voting-based evaluation and judge agreement metrics

**Models Configuration:**
- `name`: Model name/identifier
- `base_url`: **Base API URL only** (e.g., `https://api.provider.com/v1`)
  - System automatically appends `/chat/completions` or `/completions` based on `model_type`
  - Do not include the endpoint path in the URL
- `api_key`: Authentication key for the model API
- `tokenizer`: Optional HuggingFace tokenizer name/path 
  - Can be a HuggingFace model ID (e.g., `mistralai/Mistral-7B-Instruct-v0.2`) or local path
- `temperature`: Default sampling temperature (default: 0.0)
- `num_concurrent`: Number of concurrent API requests (default: 3)
- `timeout`: Request timeout in seconds (default: 300)
- `tasks`: List of tasks to run on this model (can reference `!ref tasks`)

**Weights & Biases (wandb) Configuration:**
- `enabled`: Enable/disable wandb logging
- `project`: Wandb project name
- `entity`: Wandb entity/organization
- `run_name`: Custom run name prefix (optional)
- `api_key`: Wandb API key

**Output Configuration:**
- `output_dir`: Directory where evaluation results will be saved

#### Running Evaluations

To run evaluations using the configuration file:

```bash
python evaluate.py evals.yaml
```

The script will:
1. Parse the configuration file and resolve all `!ref` references
2. Initialize the evaluation framework using Python API (no CLI subprocess calls)
3. For each model and task combination:
   - Automatically construct the correct API endpoint based on `model_type`
   - Apply chat templates if configured (`apply_chat_template: true`)
   - Run evaluation using `simple_evaluate()` from lm-eval
   - Save results to `{output_dir}/{task_name}/results_{timestamp}.json`
   - Save samples to `{output_dir}/{task_name}/samples_{task}_{timestamp}.jsonl`
4. Optionally log metrics and samples to Weights & Biases
5. Print a summary of all evaluations

**Key Features:**
- **Pure Python API**: All evaluations run through the Python API, no CLI subprocess calls
- **Per-task configuration**: Each task can use a different model API type
- **Automatic URL construction**: Base URL + model type → full endpoint URL
- **Flexible chat templates**: Control whether to apply templates locally or let the API handle them
- **Custom tokenizers**: Optionally specify a tokenizer for chat template application

### Multi-Judge Evaluation

EVE-evaluation supports **multi-judge evaluation** for open-ended tasks, where multiple LLM judges evaluate each answer independently. This approach provides more robust and reliable evaluation through consensus-based scoring.

**Key Benefits:**
- **Reduced bias**: Multiple judges reduce impact of individual judge biases
- **Voting metric**: Majority vote provides robust final scores
- **Agreement tracking**: Monitor judge consensus to identify ambiguous samples
- **Individual judge analysis**: Compare judges to identify systematic differences

**Supported Tasks:**
- `open_ended` - Open-ended question answering
- `open_ended_w_context` - Open-ended QA with context documents
- `open_ended_w_context_full` - Open-ended QA with full context

**Configuration Example:**

```yaml
constants:
  judges:
    - name: qwen3
      model: qwen/qwen3-235b-a22b-2507
      api_key: your_openrouter_api_key
      base_url: https://openrouter.ai/api/v1/
    - name: mistral_large
      model: mistralai/mistral-large-2411
      api_key: your_openrouter_api_key
      base_url: https://openrouter.ai/api/v1/
    - name: claude_sonnet
      model: anthropic/claude-3.5-sonnet
      api_key: your_openrouter_api_key
      base_url: https://openrouter.ai/api/v1/

  tasks:
    - name: open_ended_0_shot_multi_judge
      task_name: open_ended
      model_type: local-chat-completions
      num_fewshot: 0
      max_tokens: 10000
      judges: !ref judges  # Use all judges defined above
      batch_size: 15
```

**Metrics Produced:**
- `llm_as_judge_{judge_name}`: Individual score from each judge (0 or 1)
- `judge_voting`: **Majority vote result** (recommended primary metric)
- `llm_as_judge_avg`: Average score across all judges (0.0 to 1.0)
- `judge_agreement`: Percentage of samples with unanimous agreement (0.0 to 1.0)

See the [Multi-Judge Setup Guide](MULTI_JUDGE_SETUP.md) for detailed configuration examples and recommendations.

## EO Tasks

The EVE-evaluation framework includes the following Earth Observation-specific evaluation tasks:

### Core Tasks

1. **mcqa_single_answer** - Multiple choice questions with a single correct answer
   - Dataset: `eve-esa/mcqa-single-answer`
   - Metrics: Accuracy
   - Evaluates model's ability to select the correct answer from multiple options

2. **mcqa_multiple_answer** - Multiple choice questions with multiple correct answers
   - Dataset: `eve-esa/mcqa-multiple-answers`
   - Metrics: Intersection over Union (IoU), Accuracy
   - Tests model's ability to identify all correct answers among options

3. **open_ended** - Open-ended question answering
   - Dataset: `eve-esa/open-ended`
   - Metrics: LLM-as-judge
   - Tests model's ability to generate comprehensive answers to EO questions

4. **open_ended_w_context** - Open-ended QA with context documents
   - Dataset: `eve-esa/open-ended-w-context`
   - Metrics: LLM-as-judge
   - Evaluates RAG-style question answering using provided context

5. **hallucination_detection** - Detection of hallucinations in EO answers
   - Dataset: `eve-esa/hallucination-detection`
   - Metrics: Accuracy, Precision, Recall, F1
   - Tests model's ability to identify false or unsupported information

6. **refusal** - Appropriate refusal when context is insufficient
   - Dataset: `eve-esa/refusal`
   - Metrics: LLM-as-judge
   - Evaluates model's ability to refuse answering when information is unavailable

All tasks support few-shot evaluation and can be configured with custom parameters in the `evals.yaml` configuration file.

## Development

### Adding New Tasks

To add a new evaluation task:

1. Create a new directory in `tasks/` with your task name
2. Add a YAML configuration file defining the task
3. Implement any custom metrics in `metrics/`
4. Add the task to your `evals.yaml` configuration

### Custom Metrics

Custom metrics are implemented in the `metrics/` package. See existing metrics for examples of how to implement new evaluation metrics.

## Funding

This project is supported by the European Space Agency (ESA) Φ-lab through the Large Language Model for Earth Observation and Earth Science project, as part of the Foresight Element within FutureEO Block 4 programme.

## Citation
If you use this project in academic or research settings, please cite:

## License

This project is released under the Apache 2.0 License - see the [LICENSE](LICENSE) file for more details.




