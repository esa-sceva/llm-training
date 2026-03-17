# Examples

This page provides comprehensive examples of how to configure and run evaluations with Eve-evalkit. All examples use the YAML configuration format with the `evaluate.py` script.

## Basic Structure

Every configuration file has the following structure:

```yaml
constants:          # Define reusable values
  # ...

wandb:             # Optional: WandB integration
  # ...

models:            # One or more models to evaluate
  # ...

output_dir:        # Where to save results
```

---

## Example 1: EVE Earth Observation Tasks

Evaluate a model on Earth Observation-specific tasks:

```yaml
constants:
  judge_api_key: sk-or-v1-xxxxx
  judge_base_url: https://openrouter.ai/api/v1
  judge_name: mistralai/mistral-large-2411

  tasks:

    - name: mcqa_single_answer
      num_fewshot: 2
      max_tokens: 10000

    - name: hallucination_detection
      num_fewshot: 0
      max_tokens: 100

    - name: open_ended
      num_fewshot: 5
      max_tokens: 40000
      judge_api_key: !ref judge_api_key
      judge_base_url: !ref judge_base_url
      judge_name: !ref judge_name

wandb:
  enabled: true
  project: eve-evaluations
  entity: LLM4EO
  run_name: eve-model-v1

models:
  - name: eve-esa/eve_v0.1
    base_url: https://api.runpod.ai/v2/endpoint-id/openai/v1/chat/completions
    api_key: your-runpod-api-key
    temperature: 0.1
    num_concurrent: 10
    timeout: 600
    tasks: !ref tasks

output_dir: evals_outputs
```

---

## Example 2: Using LM-Eval-Harness Tasks

Eve-evalkit supports **all tasks from lm-evaluation-harness**. Here's an example using MMLU-Pro:

```yaml
constants:
  tasks:
    - name: mmlu_pro
      num_fewshot: 5
      max_tokens: 1000

    - name: gsm8k
      num_fewshot: 8
      max_tokens: 512

    - name: hellaswag
      num_fewshot: 10
      max_tokens: 100

models:
  - name: gpt-4
    base_url: https://api.openai.com/v1/chat/completions
    api_key: your-openai-api-key
    temperature: 0.0
    num_concurrent: 3
    tasks: !ref tasks

output_dir: evals_outputs
```

---

## Example 3: Mixed EVE and Standard Tasks

Combine Earth Observation tasks with standard benchmarks:

```yaml
constants:
  judge_api_key: your-judge-api-key
  judge_base_url: https://openrouter.ai/api/v1
  judge_name: mistralai/mistral-large-2411

  tasks:
    # EVE Earth Observation Tasks
    - name: hallucination_detection
      num_fewshot: 0
      max_tokens: 100

    # Standard Benchmark Tasks
    - name: mmlu_pro
      num_fewshot: 5
      max_tokens: 1000

    - name: arc_challenge
      num_fewshot: 25
      max_tokens: 100

wandb:
  enabled: true
  project: comprehensive-eval
  entity: your-org

models:
  - name: your-model
    base_url: https://api.provider.com/v1/chat/completions
    api_key: your-api-key
    temperature: 0.1
    num_concurrent: 5
    tasks: !ref tasks

output_dir: evals_outputs
```

---

## Example 4: Multiple Models

Evaluate multiple models on the same tasks:

```yaml
constants:
  tasks:
    - name: hallucination_detection
      num_fewshot: 0
      max_tokens: 100

    - name: mcqa_single_answer
      num_fewshot: 2
      max_tokens: 1000

wandb:
  enabled: true
  project: model-comparison

models:
  - name: model-a
    base_url: https://api.provider-a.com/v1/chat/completions
    api_key: api-key-a
    temperature: 0.1
    num_concurrent: 5
    tasks: !ref tasks

  - name: model-b
    base_url: https://api.provider-b.com/v1/chat/completions
    api_key: api-key-b
    temperature: 0.1
    num_concurrent: 5
    tasks: !ref tasks

output_dir: evals_outputs
```

---

## Example 5: Using Environment Variables

Instead of hardcoding API keys, use environment variables:

```yaml
constants:
  judge_api_key: ${JUDGE_API_KEY}

  tasks:
    - name: mcqa_single_answer
      num_fewshot: 0
      max_tokens: 20000
      judge_api_key: !ref judge_api_key
      judge_base_url: https://openrouter.ai/api/v1
      judge_name: mistralai/mistral-large-2411

models:
  - name: my-model
    base_url: https://api.provider.com/v1/chat/completions
    api_key: ${MODEL_API_KEY}
    tasks: !ref tasks

wandb:
  enabled: true
  api_key: ${WANDB_API_KEY}
  project: my-project

output_dir: evals_outputs
```

Set environment variables before running:

```bash
export JUDGE_API_KEY=your-judge-key
export MODEL_API_KEY=your-model-key
export WANDB_API_KEY=your-wandb-key
python evaluate.py evals.yaml
```

---

## Example 6: Testing with Limited Samples

Test your configuration on a small subset before running full evaluation:

```yaml
constants:
  tasks:
    - name: hallucination_detection
      num_fewshot: 0
      max_tokens: 100
      limit: 10  # Only evaluate first 10 samples

    - name: mcqa_single_answer
      num_fewshot: 2
      max_tokens: 1000
      limit: 5   # Only evaluate first 5 samples

models:
  - name: test-model
    base_url: https://api.provider.com/v1/chat/completions
    api_key: your-api-key
    temperature: 0.1
    num_concurrent: 2
    tasks: !ref tasks

output_dir: test_outputs
```

---

## Running Examples

To run any of these examples:

1. Save the configuration to a file (e.g., `evals.yaml`)
2. Replace placeholder values (API keys, URLs, etc.) with your actual values
3. Run the evaluation:

```bash
python evaluate.py evals.yaml
```

---

## Next Steps

- Learn more about [EO Tasks](eo_tasks.md)
- Review the [Getting Started](getting_started.md) guide for detailed configuration options
- Check the [Code Reference](evalkit.md) for API documentation
