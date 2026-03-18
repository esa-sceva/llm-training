import argparse
import dataclasses
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import traceback as tb_module

import yaml
import wandb
import logging
from lm_eval.utils import setup_logging

# Set up logging to INFO level
setup_logging(verbosity=logging.INFO)

# Suppress verbose per-request API logging from lm_eval
logging.getLogger("lm_eval.models.api_models").setLevel(logging.WARNING)

# Set up file logger so errors survive terminal buffer overflow
_file_logger = logging.getLogger("evalkit")
_file_logger.setLevel(logging.DEBUG)
_log_handler = logging.FileHandler("evaluate.log", mode="a")
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_file_logger.addHandler(_log_handler)


@dataclasses.dataclass
class FieldReference:
    reference_name: str


@dataclasses.dataclass
class FieldWithMaybeType:
    new_type: type
    name: str


@dataclasses.dataclass
class TaskConfig:
    name: str  # User-defined name for this task configuration (used for output folders and wandb)
    task_name: str | None = None  # Actual lm_eval task name (defaults to name if not provided)
    model_type: str = (
        "openai-chat-completions"  # Model API type (e.g., "openai-chat-completions", "local-completions", etc.)
    )
    max_tokens: int = 512
    num_fewshot: int = 0
    temperature: float = 0.0
    apply_chat_template: bool = True  # Whether to apply chat template to prompts
    judge_api_key: str = ""
    judge_base_url: str = ""
    judge_name: str = ""
    judges: list[dict] | None = None  # Multiple judges for multi-judge evaluation
    # Each judge dict can have: name, model, api_key, base_url, max_tokens, prompt
    batch_size: int = 1
    limit: int | None = None  # Limit number of samples to evaluate
    sample_offset: int = 0  # Skip first N samples (use with limit to evaluate e.g. last 500 samples)

    def __post_init__(self):
        # If task_name is not provided, use name as task_name
        if self.task_name is None:
            self.task_name = self.name


@dataclasses.dataclass
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    tasks: list[TaskConfig]
    temperature: float = 0.0
    num_concurrent: int = 3  # Number of concurrent API requests
    timeout: int = 300  # Timeout in seconds (default: 300s / 5 minutes)
    tokenizer: str | None = None  # HuggingFace tokenizer name for chat template (optional)
    # Eve API specific parameters
    email: str | None = None
    password: str | None = None
    public_collections: list[str] | None = None
    k: int | None = None
    threshold: float | None = None


@dataclasses.dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "eve-evalkit"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = dataclasses.field(default_factory=list)
    api_key: str | None = None


@dataclasses.dataclass
class EvaluationConfig:
    models: list[ModelConfig]
    output_dir: str = "eval_results"
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    hf_token: str | None = None


def _register_yaml_tags():
    def yaml_reference_tag_constructor(_: yaml.SafeLoader, node: yaml.Node) -> FieldReference:
        if not isinstance(node, yaml.ScalarNode):
            raise NotImplementedError
        if node.tag != "!ref":
            raise NotImplementedError(f"Tag {node.tag} is not supported")
        return FieldReference(reference_name=node.value)

    def yaml_type_conversion_tag_constructor(
        _: yaml.SafeLoader, node: yaml.Node
    ) -> FieldWithMaybeType:
        """Converts a field from one type to another as specified by the tag in the YAML
        file."""
        if not isinstance(node, yaml.ScalarNode):
            raise NotImplementedError
        tag = node.tag[1:].lower()
        if tag == "str":
            new_type = str
        elif tag == "int":
            new_type = int
        elif tag == "float":
            new_type = float
        elif tag == "bool":
            new_type = bool
        else:
            raise NotImplementedError(f"Converting to {tag} is not supported")
        return FieldWithMaybeType(new_type=new_type, name=node.value)

    yaml.SafeLoader.add_constructor("!str", yaml_type_conversion_tag_constructor)
    yaml.SafeLoader.add_constructor("!int", yaml_type_conversion_tag_constructor)
    yaml.SafeLoader.add_constructor("!ref", yaml_reference_tag_constructor)


def resolve_refs(config_dict: dict) -> dict:
    """Resolve FieldReference objects in the config."""
    constants = config_dict.get("constants", {})

    def _resolve(obj):
        if isinstance(obj, FieldReference):
            # Resolve the reference from constants and recursively resolve the result
            resolved = constants.get(obj.reference_name, obj)
            # If we got another reference or complex object, resolve it too
            if resolved != obj:
                return _resolve(resolved)
            return resolved
        elif isinstance(obj, FieldWithMaybeType):
            # Apply type conversion if needed
            value = constants.get(obj.name, obj.name)
            try:
                return obj.new_type(value)
            except (ValueError, TypeError):
                return value
        elif isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    # First resolve the entire config
    resolved = _resolve(config_dict)

    # Make sure no FieldReference objects remain anywhere
    def _check_resolved(obj):
        if isinstance(obj, (FieldReference, FieldWithMaybeType)):
            raise ValueError(f"Unresolved reference: {obj}")
        elif isinstance(obj, dict):
            for v in obj.values():
                _check_resolved(v)
        elif isinstance(obj, list):
            for item in obj:
                _check_resolved(item)

    _check_resolved(resolved)
    return resolved


def parse_task_config(task_data) -> TaskConfig:
    """Parse task data into TaskConfig object."""
    if isinstance(task_data, str):
        # Simple task name string
        return TaskConfig(name=task_data)
    elif isinstance(task_data, dict):
        # Task with parameters - use dataclass defaults for missing values
        # Build kwargs dict only with provided values
        kwargs = {"name": task_data["name"]}

        if "task_name" in task_data:
            kwargs["task_name"] = task_data["task_name"]
        if "model_type" in task_data:
            kwargs["model_type"] = task_data["model_type"]
        if "max_tokens" in task_data:
            kwargs["max_tokens"] = task_data["max_tokens"]
        if "num_fewshot" in task_data:
            kwargs["num_fewshot"] = task_data["num_fewshot"]
        if "temperature" in task_data:
            kwargs["temperature"] = task_data["temperature"]
        if "apply_chat_template" in task_data:
            kwargs["apply_chat_template"] = task_data["apply_chat_template"]
        if "judge_api_key" in task_data:
            kwargs["judge_api_key"] = task_data["judge_api_key"]
        if "judge_base_url" in task_data:
            kwargs["judge_base_url"] = task_data["judge_base_url"]
        if "judge_name" in task_data:
            kwargs["judge_name"] = task_data["judge_name"]
        if "judges" in task_data:
            kwargs["judges"] = task_data["judges"]
        if "limit" in task_data:
            kwargs["limit"] = task_data["limit"]
        if "sample_offset" in task_data:
            kwargs["sample_offset"] = task_data["sample_offset"]

        return TaskConfig(**kwargs)
    else:
        raise ValueError(f"Invalid task data: {task_data}")


def load_samples(output_dir: str, model_name: str, task_name: str) -> list[dict] | None:
    """Load evaluation samples from JSONL files, selecting the newest version of each file."""
    task_output_dir = Path(output_dir) / task_name / model_name.replace("/", "__")

    print(f"Loading samples from: {task_output_dir}")

    # Find all JSONL files
    jsonl_files = list(task_output_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No JSONL files found for {model_name} / {task_name}")
        return None

    # Pattern to extract base name and timestamp from filename
    # Example: samples_lidar_2025-11-17T13-05-01.938804.jsonl
    # Timestamp pattern: YYYY-MM-DDTHH-MM-SS.microseconds
    timestamp_pattern = re.compile(r"(.+)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.jsonl$")

    # Group files by base name and track the newest version
    file_groups = defaultdict(list)

    for jsonl_file in jsonl_files:
        match = timestamp_pattern.match(jsonl_file.name)
        if match:
            base_name = match.group(1)
            timestamp_str = match.group(2)

            # Parse timestamp (convert '-' to ':' for time part only)
            try:
                # Format: 2025-11-17T13-05-01.938804 -> 2025-11-17T13:05:01.938804
                # Split on 'T' to separate date and time, then replace '-' with ':' only in time part
                date_part, time_part = timestamp_str.split("T")
                time_normalized = time_part.replace("-", ":")
                timestamp_normalized = f"{date_part}T{time_normalized}"
                timestamp = datetime.fromisoformat(timestamp_normalized)
                file_groups[base_name].append((timestamp, jsonl_file))
            except ValueError as e:
                print(f"Warning: Failed to parse timestamp from {jsonl_file.name}: {e}")
                # If timestamp parsing fails, still include the file with a default timestamp
                file_groups[base_name].append((datetime.min, jsonl_file))
        else:
            # If no timestamp pattern found, use the file anyway with a default base name
            file_groups[jsonl_file.stem].append((datetime.min, jsonl_file))

    # Select the newest file for each base name
    selected_files = []
    for base_name, files in file_groups.items():
        # Sort by timestamp (newest first) and select the first one
        files.sort(key=lambda x: x[0], reverse=True)
        newest_file = files[0][1]
        selected_files.append(newest_file)
        if len(files) > 1:
            print(
                f"  Found {len(files)} versions of '{base_name}', using newest: {newest_file.name}"
            )

    # Load all samples from selected files
    all_samples = []
    for sample_file in selected_files:
        try:
            with open(sample_file, "r") as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        all_samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to load samples from {sample_file}: {e}")

    print(f"  Loaded {len(all_samples)} samples from {len(selected_files)} file(s)")
    return all_samples if all_samples else None


def calculate_weighted_stderr(task_stderrs, task_sizes, total_size):
    """
    Calculate weighted standard error for aggregated metric.

    Uses the formula for combining standard errors of independent samples:
    SE_combined = sqrt(sum((n_i/N)^2 * SE_i^2))
    """
    weighted_var = sum(
        ((size / total_size) ** 2) * (stderr**2) for stderr, size in zip(task_stderrs, task_sizes)
    )
    return math.sqrt(weighted_var)


def add_aggregate_metrics_to_results(results_file: Path) -> bool:
    """
    Add aggregate metrics to group tasks if they're missing.

    Calculates weighted averages (by sample size) for group tasks according
    to their aggregate_metric_list configuration.

    Returns True if any metrics were added, False otherwise.
    """
    try:
        # Load results
        with open(results_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Warning: Failed to load results file: {e}")
        return False

    # Check if there are any groups
    groups = data.get("groups", {})
    if not groups:
        return False

    modified = False

    # Process each group
    for group_name, group_data in groups.items():
        # Get subtasks for this group
        subtasks = data.get("group_subtasks", {}).get(group_name, [])
        if not subtasks:
            continue

        # Check if aggregate metric is already present
        group_results = data["results"].get(group_name, {})
        if "exact_match,get_response" in group_results:
            # Metric already exists
            continue

        # Calculate aggregate metrics
        task_scores = []
        task_stderrs = []
        task_sizes = []

        for task_name in subtasks:
            # Get exact_match score
            task_result = data["results"].get(task_name, {})
            score = task_result.get("exact_match,get_response")
            stderr = task_result.get("exact_match_stderr,get_response")

            # Get sample size
            n_samples_data = data.get("n-samples", {}).get(task_name, {})
            size = n_samples_data.get("effective", n_samples_data.get("original", 0))

            if score is not None and size > 0:
                task_scores.append(score)
                task_stderrs.append(stderr if stderr is not None else 0)
                task_sizes.append(size)

        if not task_scores:
            continue

        # Calculate weighted average (weight by size)
        total_size = sum(task_sizes)
        weighted_avg = sum(
            score * (size / total_size) for score, size in zip(task_scores, task_sizes)
        )

        # Calculate weighted standard error
        weighted_stderr = calculate_weighted_stderr(task_stderrs, task_sizes, total_size)

        # Update the group results
        if group_name not in data["results"]:
            data["results"][group_name] = {}

        data["results"][group_name]["exact_match,get_response"] = weighted_avg
        data["results"][group_name]["exact_match_stderr,get_response"] = weighted_stderr

        # Also update groups section
        if isinstance(data["groups"][group_name], dict):
            data["groups"][group_name]["exact_match,get_response"] = weighted_avg
            data["groups"][group_name]["exact_match_stderr,get_response"] = weighted_stderr
        else:
            # If groups entry is just a string, convert to dict
            data["groups"][group_name] = {
                "alias": group_name,
                "exact_match,get_response": weighted_avg,
                "exact_match_stderr,get_response": weighted_stderr,
            }

        print(
            f"  ✓ Added aggregate metrics for {group_name}: {weighted_avg:.4f} ± {weighted_stderr:.4f}"
        )
        modified = True

    # Write back to file if modified
    if modified:
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

    return modified


def load_eval_results(output_dir: str, model_name: str, task_name: str) -> dict | None:
    """Load evaluation results from lm_eval output directory, selecting the newest file."""
    task_output_dir = Path(output_dir) / task_name / model_name.replace("/", "__")

    print(f"Loading results from: {task_output_dir}")

    # Find results.json file
    results_files = list(task_output_dir.glob("**/*.json"))
    if not results_files:
        print(f"Warning: No json found for {model_name} / {task_name}")
        return None

    # Sort by modification time (newest first) to ensure we get the most recent results
    results_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    results_file = results_files[0]

    if len(results_files) > 1:
        print(f"  Found {len(results_files)} result files, using newest: {results_file.name}")

    with open(results_file, "r") as f:
        data = json.load(f)

    return data.get("results", {})


def flatten_metrics(results: dict, original_task_name: str) -> dict:
    """Flatten nested metrics dictionary for wandb logging."""
    flattened = {}

    for task_name, metrics in results.items():
        # Skip 'alias' and 'group' keys
        for key, value in metrics.items():
            if key in ["alias", "group"]:
                continue

            # Create hierarchical metric name: original_task/subtask/metric
            # Remove any commas from metric names (lm_eval uses commas for variants)
            clean_key = key.split(",")[0] if "," in key else key

            # If task_name is different from original_task_name, it's a subtask
            if task_name != original_task_name:
                metric_name = f"{original_task_name}/{task_name}/{clean_key}"
            else:
                metric_name = f"{original_task_name}/{clean_key}"

            # Only log numeric values
            if isinstance(value, (int, float)):
                flattened[metric_name] = value

    return flattened


def sanitize_config_for_wandb(config: dict) -> dict:
    """
    Sanitize configuration by removing sensitive information like API keys and tokens.
    Returns a deep copy with sensitive fields redacted.
    """
    import copy

    # Create a deep copy to avoid modifying the original
    sanitized = copy.deepcopy(config)

    # List of sensitive field names to redact
    sensitive_fields = [
        "api_key",
        "judge_api_key",
        "hf_token",
        "token",
        "password",
        "secret",
        "credential",
    ]

    def redact_sensitive(obj, path=""):
        """Recursively redact sensitive fields."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = key.lower()
                # Check if this key contains any sensitive field name
                if any(sensitive in key_lower for sensitive in sensitive_fields):
                    # Redact the value
                    if value and isinstance(value, str) and len(value) > 0:
                        obj[key] = "***REDACTED***"
                elif isinstance(value, (dict, list)):
                    # Recursively process nested structures
                    redact_sensitive(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    redact_sensitive(item, f"{path}[{i}]")

    redact_sensitive(sanitized)
    return sanitized


def init_wandb_for_task(
    wandb_config: WandbConfig, model: ModelConfig, task: TaskConfig, full_config: dict | None = None
):
    """Initialize wandb run for a specific model and task."""
    if not wandb_config.enabled:
        return None

    try:
        # Login to wandb with API key if provided (only once)
        if wandb_config.api_key and not wandb.run:
            wandb.login(key=wandb_config.api_key)

        # Create run name: prefix_model_task
        model_short_name = model.name.split("/")[-1]
        run_name_parts = []
        if wandb_config.run_name:
            run_name_parts.append(wandb_config.run_name)
        run_name_parts.extend([model_short_name, task.name])
        run_name = "_".join(run_name_parts)

        # Initialize wandb run with comprehensive configuration
        run_config = {
            "model": {
                "name": model.name,
                "base_url": model.base_url,
                "temperature": model.temperature,
                "num_concurrent": model.num_concurrent,
                "timeout": model.timeout,
                "tokenizer": model.tokenizer,
            },
            "task": {
                "name": task.name,  # User-defined task configuration name
                "task_name": task.task_name,  # Actual lm_eval task name
                "model_type": task.model_type,
                "max_tokens": task.max_tokens,
                "num_fewshot": task.num_fewshot,
                "temperature": task.temperature,
                "apply_chat_template": task.apply_chat_template,
                "batch_size": task.batch_size,
                "limit": task.limit,
                "sample_offset": task.sample_offset,
            },
        }

        # Add judge configuration if present
        if task.judge_name:
            run_config["task"]["judge"] = {
                "judge_name": task.judge_name,
                "judge_base_url": task.judge_base_url,
            }

        # Add full YAML configuration if provided (sanitized)
        if full_config:
            # Create a sanitized copy of the config
            sanitized_config = sanitize_config_for_wandb(full_config)
            run_config["yaml_config"] = sanitized_config

        wandb_init_kwargs = {
            "project": wandb_config.project,
            "name": run_name,
            "config": run_config,
        }

        if wandb_config.entity:
            wandb_init_kwargs["entity"] = wandb_config.entity

        if wandb_config.tags:
            wandb_init_kwargs["tags"] = wandb_config.tags

        run = wandb.init(**wandb_init_kwargs)
        return run

    except Exception as e:
        print(f"\n✗ Failed to initialize W&B: {e}")
        print("Continuing without W&B logging...")
        return None


def create_samples_table(samples: list[dict]) -> wandb.Table:
    """Create a wandb table from evaluation samples, dynamically handling all fields."""
    if not samples:
        return wandb.Table(columns=["empty"], data=[])

    # First pass: collect all unique column names across all samples
    all_columns = set()

    for sample in samples:
        # Add top-level fields
        all_columns.update(sample.keys())

        # Add doc fields with 'doc.' prefix
        doc = sample.get("doc", {})
        if isinstance(doc, dict):
            for key in doc.keys():
                all_columns.add(f"doc.{key}")

    # Sort columns for consistent ordering, with important fields first
    priority_fields = ["doc_id", "doc.Topic", "doc.Question", "target", "filtered_resps"]
    columns = []

    # Add priority fields first if they exist
    for field in priority_fields:
        if field in all_columns:
            columns.append(field)
            all_columns.remove(field)

    # Add remaining fields sorted alphabetically
    columns.extend(sorted(all_columns))

    # Create table
    table = wandb.Table(columns=columns)

    # Populate table
    for sample in samples:
        row = []
        doc = sample.get("doc", {})

        for col in columns:
            if col.startswith("doc."):
                # Handle doc fields
                field_name = col[4:]  # Remove 'doc.' prefix
                value = doc.get(field_name, "")
            else:
                # Handle top-level fields
                value = sample.get(col, "")

            # Convert value to string representation for table
            if value is None:
                row.append("")
            elif isinstance(value, list):
                # For lists, join elements or show first element
                if col == "filtered_resps" and value:
                    # For model responses, take first one
                    row.append(str(value[0]))
                else:
                    # For other lists, show as JSON
                    row.append(json.dumps(value))
            elif isinstance(value, dict):
                # For dicts, show as JSON
                row.append(json.dumps(value))
            else:
                row.append(str(value))

        table.add_data(*row)

    return table


def log_task_metrics_to_wandb(task_metrics: dict, model: ModelConfig, task: TaskConfig):
    """Log metrics for a task to the active wandb run."""
    # Create flat metrics dict (without model name prefix since each run is per model/task)
    flat_metrics = {}
    summary_table = wandb.Table(columns=["model_name", "task", "subtask", "metric", "value"])

    for metric_name, value in sorted(task_metrics.items()):
        # Split metric_name (format: task/metric or task/subtask/metric)
        parts = metric_name.split("/")

        if len(parts) == 2:
            # Format: task/metric - just use the metric name
            _, metric = parts
            flat_metrics[metric] = value
            summary_table.add_data(model.name, task.name, "", metric, value)
        elif len(parts) == 3:
            # Format: task/subtask/metric - use subtask/metric
            _, subtask, metric = parts
            metric_key = f"{subtask}/{metric}"
            flat_metrics[metric_key] = value
            # Add subtask in separate column
            summary_table.add_data(model.name, task.name, subtask, metric, value)
        else:
            # Fallback for unexpected format
            flat_metrics[metric_name] = value
            summary_table.add_data(model.name, task.name, "", metric_name, value)

    # Log metrics
    wandb.log(flat_metrics)
    wandb.log({"metrics_summary": summary_table})


def run_evaluation(
    model: ModelConfig, task: TaskConfig, output_dir: str, hf_token: str | None = None
) -> tuple[int, dict | None]:
    """Run lm_eval for a specific model and task using simple_evaluate.

    Returns:
        tuple: (return_code, results_dict) where results_dict is None if evaluation failed
    """
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager, get_task_dict

    # Create output directory for this specific evaluation (use user-defined name)
    task_output_dir = Path(output_dir) / task.name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables before initializing the model
    env_backup = {}

    # Set API key as environment variable (more secure than command line)
    env_backup["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = model.api_key

    # Set HuggingFace token if provided in config, otherwise use environment variable
    if hf_token:
        env_backup["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        env_backup["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    # If not in config, check if it exists in environment and preserve it
    elif "HUGGING_FACE_HUB_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
    elif "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # Set judge environment variables if provided
    if task.judges:
        # Multi-judge mode: serialize judges list as JSON
        env_backup["TASK_JUDGES"] = os.environ.get("TASK_JUDGES")
        os.environ["TASK_JUDGES"] = json.dumps(task.judges)
        print(f"  [INFO] Multi-judge mode enabled with {len(task.judges)} judges")
    elif task.judge_api_key:
        # Single judge mode (backward compatibility)
        env_backup["JUDGE_API_KEY"] = os.environ.get("JUDGE_API_KEY")
        os.environ["JUDGE_API_KEY"] = task.judge_api_key
    if task.judge_base_url:
        env_backup["JUDGE_BASE_URL"] = os.environ.get("JUDGE_BASE_URL")
        os.environ["JUDGE_BASE_URL"] = task.judge_base_url
    if task.judge_name:
        env_backup["JUDGE_NAME"] = os.environ.get("JUDGE_NAME")
        os.environ["JUDGE_NAME"] = task.judge_name

    print(f"\nRunning evaluation:")
    print(f"  Model: {model.name}")
    print(f"  Model Type: {task.model_type}")
    print(f"  Task Config: {task.name}")
    if task.name != task.task_name:
        print(f"  LM-Eval Task: {task.task_name}")
    print(f"  Output: {task_output_dir}")
    try:
        _tmp_tm = TaskManager(include_path="tasks")
        _tmp_td = get_task_dict([task.task_name], _tmp_tm)
        for _tname, _tobj in _tmp_td.items():
            if hasattr(_tobj, 'dataset'):
                _split = getattr(_tobj.config, 'test_split', None) or 'test'
                if _split in _tobj.dataset:
                    _n = len(_tobj.dataset[_split])
                    _effective = min(_n, task.limit) if task.limit and task.limit > 0 else _n
                    print(f"  Questions: {_effective}" + (f" (limited from {_n})" if _effective != _n else ""))
    except Exception:
        pass
    print("-" * 80)
    _file_logger.info(f"Starting evaluation: model={model.name}, task={task.name}, output={task_output_dir}")

    try:
        # Construct model arguments based on model type
        if task.model_type == "eve-api":
            # Eve API specific arguments
            base_url = model.base_url.rstrip("/")  # Remove trailing slashes
            model_args = (
                f"email={model.email},"
                f"password={model.password},"
                f"base_url={base_url},"
                f"num_concurrent={model.num_concurrent},"
                f"timeout={model.timeout},"
                f"batch_size={task.batch_size}"
            )

            # Add Eve-specific RAG parameters
            if model.public_collections:
                # Join collections with pipe separator to avoid comma conflicts with lm_eval's parser
                # Will be split back into a list in eve_api.py
                collections_str = "|".join(model.public_collections)
                model_args += f",public_collections={collections_str}"
                print(f"[DEBUG] public_collections value: {model.public_collections}")
                print(f"[DEBUG] Joined with pipe: {collections_str}")
                print(f"[DEBUG] model_args after adding collections: {model_args}")
            if model.k is not None:
                model_args += f",k={model.k}"
            if model.threshold is not None:
                model_args += f",threshold={model.threshold}"

            print(f"\n[DEBUG] Final model_args for eve-api:")
            print(f"  {model_args}\n")
        else:
            # Standard OpenAI-compatible API arguments
            # Construct full API URL based on model type
            # Base URL should be just the base (e.g., https://api.openai.com/v1/)
            # We append either 'completions' or 'chat/completions' based on model type
            base_url = model.base_url.rstrip("/")  # Remove trailing slashes

            # Check if "chat" is in the model type to determine the endpoint
            if "chat" in task.model_type.lower():
                full_url = f"{base_url}/chat/completions"
            else:
                full_url = f"{base_url}/completions"

            # Construct model arguments as a string (lm_eval expects comma-separated key=value pairs)
            model_args = (
                f"base_url={full_url},"
                f"model={model.name},"
                f"num_concurrent={model.num_concurrent},"
                f"max_gen_toks={task.max_tokens},"
                f"temperature={task.temperature if task.temperature > 0 else model.temperature},"
                f"timeout={model.timeout},"
                f"batch_size={task.batch_size}"
            )

            # Add tokenizer if provided
            if model.tokenizer:
                model_args += f",tokenizer={model.tokenizer}"

        # Create TaskManager to include custom tasks directory
        # This is equivalent to the CLI's --include tasks option
        task_manager = TaskManager(include_path="tasks")

        # If sample_offset is set, load the task and slice its dataset
        # This allows evaluating e.g. the last 500 samples by setting offset=492
        eval_tasks = [task.task_name]
        if task.sample_offset and task.sample_offset > 0:
            print(f"  [INFO] Applying sample_offset={task.sample_offset} to dataset")
            task_dict = get_task_dict([task.task_name], task_manager)
            for tname, task_obj in task_dict.items():
                if hasattr(task_obj, 'dataset'):
                    split = task_obj.config.test_split or 'test'
                    if split in task_obj.dataset:
                        total = len(task_obj.dataset[split])
                        if task.sample_offset >= total:
                            raise ValueError(
                                f"sample_offset ({task.sample_offset}) >= dataset size ({total})"
                            )
                        task_obj.dataset[split] = task_obj.dataset[split].select(
                            range(task.sample_offset, total)
                        )
                        new_total = len(task_obj.dataset[split])
                        print(f"  [INFO] Dataset sliced: {total} -> {new_total} samples (offset {task.sample_offset})")
            eval_tasks = list(task_dict.values())

        # Run evaluation using simple_evaluate
        # Pass model type as string and let simple_evaluate handle model initialization
        results = simple_evaluate(
            model=task.model_type,
            model_args=model_args,
            tasks=eval_tasks,
            num_fewshot=task.num_fewshot,
            limit=task.limit if task.limit and task.limit > 0 else None,
            log_samples=True,
            apply_chat_template=task.apply_chat_template,
            task_manager=task_manager,
        )

        if results is None:
            raise RuntimeError("simple_evaluate returned None")

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")

        def default_serializer(obj):
            if isinstance(obj, set):
                return list(obj)
            return str(obj)

        # Save samples as JSONL files
        if "samples" in results:
            for subtask_name, samples_list in results["samples"].items():
                samples_path = task_output_dir / f"samples_{subtask_name}_{timestamp}.jsonl"
                print(f"  Saving samples to: {samples_path}")
                with open(samples_path, "w") as f:
                    for sample in samples_list:
                        f.write(json.dumps(sample, default=default_serializer) + "\n")

        # Save results without samples (samples are in JSONL files)
        results_to_save = results.copy()
        if "samples" in results_to_save:
            del results_to_save["samples"]

        results_path = task_output_dir / f"results_{timestamp}.json"
        print(f"  Saving results to: {results_path}")
        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=2, default=default_serializer)

        print(f"✓ Completed: {model.name} on {task.name}")
        _file_logger.info(f"Completed: {model.name} on {task.name}")
        return_code = 0
        eval_results = results

    except Exception as e:
        error_msg = f"✗ Failed: {model.name} on {task.name}\n  Error: {e}"
        full_traceback = tb_module.format_exc()
        print(error_msg)
        print(full_traceback)

        # Always log to file so errors survive terminal buffer overflow
        _file_logger.error(error_msg)
        _file_logger.error(full_traceback)

        # Also save error to output directory for easy access
        try:
            error_path = task_output_dir / "last_error.log"
            with open(error_path, "w") as f:
                f.write(f"{error_msg}\n\n{full_traceback}")
            print(f"  Error details saved to: {error_path}")
        except Exception:
            pass

        return_code = 1
        eval_results = None

    finally:
        # Restore environment variables
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return return_code, eval_results


def evaluate_model(
    model: ModelConfig,
    output_dir: str,
    wandb_config: WandbConfig,
    hf_token: str | None = None,
    full_config: dict | None = None,
) -> dict[str, int]:
    """Evaluate a model on all its tasks."""
    results = {}

    for task in model.tasks:
        return_code, eval_results = run_evaluation(model, task, output_dir, hf_token)
        results[task.name] = return_code

        # Log to wandb immediately after each task completes
        _file_logger.info(f"Task {task.name} finished with return_code={return_code}")
        if return_code == 0 and eval_results:  # Only if task succeeded and we have results
            # Log to wandb if enabled
            if wandb_config.enabled:
                print(f"\nPreparing W&B logging for {task.name}...")
                _file_logger.info(f"Preparing W&B logging for {task.name}")

                # Extract metrics from results dict
                task_results = eval_results.get("results", {})

                if task_results:
                    # Initialize a new wandb run for this model/task combination
                    wandb_run = init_wandb_for_task(wandb_config, model, task, full_config)

                    if wandb_run:
                        # Flatten and log metrics for this task
                        task_metrics = flatten_metrics(task_results, task.name)
                        log_task_metrics_to_wandb(task_metrics, model, task)

                        # Log samples if available
                        if "samples" in eval_results and eval_results["samples"]:
                            print(f"Logging samples for W&B ({task.name})...")
                            # samples is a dict with subtask_name -> list of samples
                            all_samples = []
                            for subtask_samples in eval_results["samples"].values():
                                all_samples.extend(subtask_samples)

                            if all_samples:
                                samples_table = create_samples_table(all_samples)
                                wandb.log({"samples": samples_table})
                                print(f"✓ Logged {len(all_samples)} samples to W&B")
                        else:
                            print(f"Warning: No samples found for {task.name}")

                        print(f"✓ Logged {len(task_metrics)} metrics to W&B: {wandb_run.url}")

                        # Finish this run (wrapped because wandb 0.25 can crash on cleanup with Python 3.14)
                        try:
                            wandb.finish()
                        except Exception as e:
                            print(f"Warning: wandb.finish() failed ({e}), but data was already synced.")
                else:
                    print(f"Warning: No metrics found for {task.name}")
        else:
            print(f"\nSkipping W&B logging for {task.name} (evaluation failed)")
            _file_logger.warning(f"Skipping W&B logging for {task.name} (return_code={return_code})")

    return results


def main(config_file: str):
    """Main entry point for the evaluation script."""

    # Register custom YAML tags
    _register_yaml_tags()

    # Load and parse config
    with open(config_file, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Resolve !ref references
    config_dict = resolve_refs(config_dict)

    # Parse wandb config if present
    wandb_config = WandbConfig()
    if "wandb" in config_dict:
        wandb_data = config_dict["wandb"]
        wandb_config = WandbConfig(
            enabled=wandb_data.get("enabled", False),
            project=wandb_data.get("project", "eve-evaluation"),
            entity=wandb_data.get("entity"),
            run_name=wandb_data.get("run_name"),
            tags=wandb_data.get("tags", []),
            api_key=wandb_data.get("api_key"),
        )

    # Parse into dataclass structure
    eval_config = EvaluationConfig(
        models=[
            ModelConfig(
                name=model["name"],
                base_url=model["base_url"],
                api_key=model.get("api_key", ""),
                temperature=model.get("temperature", 0.0),
                num_concurrent=model.get("num_concurrent", 3),
                timeout=model.get("timeout", 300),
                tokenizer=model.get("tokenizer"),
                email=model.get("email"),
                password=model.get("password"),
                public_collections=model.get("public_collections"),
                k=model.get("k"),
                threshold=model.get("threshold"),
                tasks=[parse_task_config(task) for task in model["tasks"]],
            )
            for model in config_dict["models"]
        ],
        output_dir=config_dict.get("output_dir", "eval_results"),
        wandb=wandb_config,
        hf_token=config_dict.get("constants", {}).get("hf_token"),
    )

    # Create output directory
    Path(eval_config.output_dir).mkdir(parents=True, exist_ok=True)

    # Track all results
    all_results = {}

    # Evaluate each model
    for model in eval_config.models:
        print(f"\n{'=' * 80}")
        print(f"Evaluating model: {model.name}")
        print(f"Tasks: {[task.name for task in model.tasks]}")
        print(f"{'=' * 80}")

        model_results = evaluate_model(
            model, eval_config.output_dir, eval_config.wandb, eval_config.hf_token, config_dict
        )
        all_results[model.name] = model_results

    # Print summary
    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}")

    total_evaluations = 0
    failed_evaluations = 0

    for model_name, tasks_results in all_results.items():
        print(f"\n{model_name}:")
        for task_name, return_code in tasks_results.items():
            status = "✓ PASSED" if return_code == 0 else "✗ FAILED"
            print(f"  {task_name}: {status}")
            total_evaluations += 1
            if return_code != 0:
                failed_evaluations += 1

    print(f"\n{'=' * 80}")
    print(f"Total: {total_evaluations} evaluations")
    print(f"Passed: {total_evaluations - failed_evaluations}")
    print(f"Failed: {failed_evaluations}")
    print(f"{'=' * 80}\n")

    return 0 if failed_evaluations == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lm_eval harness based on a YAML config file")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")

    args = parser.parse_args()

    exit_code = main(args.config)
    exit(exit_code)
