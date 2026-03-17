# Eve-evalkit

Welcome to the Wve-evalkit documentation. This framework provides comprehensive tools for evaluating language models on Earth Observation (EO) specific tasks and benchmarks.

## What is Eve-evalkit?

Eve-evalkit is built on top of the [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), providing:

- **Custom EO Tasks**: Specialized evaluation tasks for Earth Observation domain, including MCQA, hallucination detection, and more
- **Full LM-Eval-Harness Support**: Access to all standard benchmarks (MMLU-Pro, GSM8K, HellaSwag, etc.)
- **WandB Integration**: Automatic experiment tracking and metric logging
- **Flexible Configuration**: YAML-based configuration for easy experiment management
- **Production Ready**: Built for evaluating models via API endpoints with concurrent requests

## Quick Links

- **[Getting Started](getting_started.md)**: Installation, configuration, and running your first evaluation
- **[Examples](examples.md)**: Comprehensive configuration examples and use cases
- **[EO Tasks](eo_tasks.md)**: Detailed information about Earth Observation evaluation tasks
- **[Code Reference](evalkit.md)**: API documentation and code examples

## Key Features

### Earth Observation Tasks

Evaluate models on specialized EO capabilities:

- **Multiple-Choice QA**: Single and multiple-answer questions from EO curricula
- **Open-Ended QA**: Free-form question answering with and without context
- **Hallucination Detection**: Identify fabricated or unsupported information
- **Refusal Testing**: Assess appropriate refusal behavior when context is insufficient

### Comprehensive Metrics

- **LLM-as-Judge**: Sophisticated evaluation using judge models
- **Traditional Metrics**: Accuracy, F1, Precision, Recall, IoU
- **Semantic Metrics**: BERTScore, Cosine Similarity
- **Generation Metrics**: BLEU, ROUGE

### Production Features

- **API Model Support**: Evaluate models via OpenAI-compatible endpoints
- **Concurrent Requests**: Speed up evaluations with parallel API calls
- **Timeout Handling**: Graceful handling of slow or failed requests
- **Result Logging**: Comprehensive JSON and JSONL outputs
- **WandB Integration**: Track experiments and visualize metrics


## Example Use Cases

### Research & Development

- Benchmark Earth Observation models against established tasks
- Compare model performance across different architectures
- Identify strengths and weaknesses in EO domain understanding

### Model Selection

- Evaluate multiple models on EO-specific capabilities
- Compare general-purpose models vs. domain-specific models
- Assess trade-offs between performance and cost


## Documentation Structure

- **Getting Started**: Installation, configuration, and basic usage
- **Examples**: Comprehensive configuration examples and use cases
- **EO Tasks**: Detailed task descriptions, metrics, and examples
- **Code Reference**: API documentation and programmatic usage

## Support & Contributing

- **GitHub**: [eve-esa/eve-evaluation](https://github.com/eve-esa/eve-evaluation)
- **Issues**: Report bugs or request features on GitHub

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@misc{eve2025,
  title={EVE: Earth Virtual Expert},
  author={ESA},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/eve-esa/eve_v0.1}
}
```

For the underlying evaluation harness:

```bibtex
@software{eval-harness,
  author       = {Gao, Leo and others},
  title        = {A framework for few-shot language model evaluation},
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
