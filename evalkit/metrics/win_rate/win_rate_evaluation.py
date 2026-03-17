#!/usr/bin/env python3
"""
Win Rate Evaluation Script

Compares two model outputs using multiple LLM judges to determine win rates.
Configurable via YAML file and logs results to Weights & Biases.

Usage:
    python win_rate_evaluation.py --config win_rate_config.yaml
"""

import argparse
import os
import random
import re
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Add the notebooks/win_rate directory to the path to import utils
from utils import (
    calculate_win_rates,
    create_judge_prompt,
    evaluate_parallel,
    results_to_dataframe,
)


def load_config(config_path: str) -> Dict:
    """Load and parse YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Expand environment variables in API keys
    for judge in config.get("judges", []):
        if "api_key" in judge:
            judge["api_key"] = expand_env_vars(judge["api_key"])

    return config


def expand_env_vars(value: str) -> str:
    """Expand environment variables in strings like ${VAR_NAME}."""
    pattern = re.compile(r"\$\{([^}]+)\}")

    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return pattern.sub(replacer, value)


def generate_model_combinations(config: Dict) -> List[Tuple[Dict, Dict]]:
    """Generate all combinations of model_a and model_b from config.

    Supports both single model configs and lists of models.
    Returns list of tuples: [(model_a_config, model_b_config), ...]
    """
    # Check if model_a and model_b are lists or single configs
    model_a_configs = config.get("models_a") or config.get("model_a")
    model_b_configs = config.get("models_b") or config.get("model_b")

    # Normalize to lists
    if isinstance(model_a_configs, dict):
        model_a_configs = [model_a_configs]
    if isinstance(model_b_configs, dict):
        model_b_configs = [model_b_configs]

    # Generate all combinations
    combinations = list(product(model_a_configs, model_b_configs))

    print(
        f"\nFound {len(model_a_configs)} model_a config(s) and {len(model_b_configs)} model_b config(s)"
    )
    print(f"Will run {len(combinations)} evaluation(s) total\n")

    return combinations


def sanitize_name(name: str) -> str:
    """Sanitize model name for use in filenames."""
    return name.lower().replace(" ", "_").replace(".", "").replace("-", "_").replace("/", "_")


def format_filename(template: str, model_a_name: str, model_b_name: str) -> str:
    """Format filename template with model names."""
    return template.format(model_a=sanitize_name(model_a_name), model_b=sanitize_name(model_b_name))


def load_and_align_data(config: Dict) -> pd.DataFrame:
    """Load CSV files and align them by question."""
    model_a_file = config["model_a"]["file"]
    model_b_file = config["model_b"]["file"]
    model_a_name = config["model_a"]["name"]
    model_b_name = config["model_b"]["name"]

    print(f"\nLoading data...")
    print(f"  Model A: {model_a_file}")
    print(f"  Model B: {model_b_file}")

    # Load CSV files
    df_model_a = pd.read_csv(model_a_file)
    df_model_b = pd.read_csv(model_b_file)

    print(f"  Loaded {len(df_model_a)} rows from Model A")
    print(f"  Loaded {len(df_model_b)} rows from Model B")

    # Sanitize model names for column naming
    model_a_col = sanitize_name(model_a_name)
    model_b_col = sanitize_name(model_b_name)

    # Detect question column name - handle both "Question" and "doc\\.Question"
    question_col_a = None
    question_col_b = None

    for possible_col in ["doc\\\\.Question", "Question"]:
        if possible_col in df_model_a.columns:
            question_col_a = possible_col
        if possible_col in df_model_b.columns:
            question_col_b = possible_col

    if question_col_a is None:
        raise ValueError(
            f"Could not find question column in {model_a_file}. Expected 'Question' or 'doc\\\\.Question'"
        )
    if question_col_b is None:
        raise ValueError(
            f"Could not find question column in {model_b_file}. Expected 'Question' or 'doc\\\\.Question'"
        )

    print(
        f"  Detected question column: Model A uses '{question_col_a}', Model B uses '{question_col_b}'"
    )

    # Merge dataframes on Question field
    comparison_df = pd.merge(
        df_model_a[[question_col_a, "target", "filtered_resps"]],
        df_model_b[[question_col_b, "filtered_resps"]],
        left_on=question_col_a,
        right_on=question_col_b,
        suffixes=(f"_{model_a_col}", f"_{model_b_col}"),
    )

    # Rename columns for clarity with actual model names
    rename_dict = {
        f"filtered_resps_{model_a_col}": f"answer_{model_a_col}",
        f"filtered_resps_{model_b_col}": f"answer_{model_b_col}",
    }

    # Add question column renaming (keep only one, rename to "question")
    if question_col_a == question_col_b:
        rename_dict[question_col_a] = "question"
    else:
        # Different column names - drop one and rename the other
        rename_dict[question_col_a] = "question"
        if question_col_b in comparison_df.columns:
            comparison_df.drop(columns=[question_col_b], inplace=True)

    comparison_df.rename(columns=rename_dict, inplace=True)

    print(f"  Aligned {len(comparison_df)} questions for comparison\n")

    return comparison_df


def run_evaluation(config: Dict, comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Run parallel evaluation with all judges."""
    eval_config = config["evaluation"]
    judges = config["judges"]
    model_a_name = config["model_a"]["name"]
    model_b_name = config["model_b"]["name"]

    # Set random seed if specified
    if eval_config.get("random_seed") is not None:
        random.seed(eval_config["random_seed"])
        print(f"Random seed set to: {eval_config['random_seed']}")

    # Limit evaluation if specified
    eval_limit = eval_config.get("limit")
    eval_df = comparison_df.head(eval_limit) if eval_limit else comparison_df

    print(f"\nStarting parallel evaluation...")
    print(f"  Comparison: {model_a_name} vs {model_b_name}")
    print(f"  Questions: {len(eval_df)}")
    print(f"  Judges: {len(judges)}")
    print(f"  Total API calls: {len(eval_df) * len(judges)}")
    print(f"  Max parallel workers: {eval_config['max_workers']}")
    print(f"  Rate limit delay: {eval_config['rate_limit_delay']}s per thread")
    print(f"\nNote: Answer order is randomized per sample to prevent position bias\n")

    # Run parallel evaluation
    start_time = time.time()
    results = evaluate_parallel(
        eval_df,
        judges,
        model_a_name,
        model_b_name,
        max_workers=eval_config["max_workers"],
        rate_limit_delay=eval_config["rate_limit_delay"],
        progress_bar=True,
    )
    elapsed_time = time.time() - start_time

    # Convert results to DataFrame with actual model names
    comparison_df = results_to_dataframe(results, comparison_df, model_a_name, model_b_name)

    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"  Time elapsed: {elapsed_time:.1f}s")
    print(f"  Average time per call: {elapsed_time / (len(eval_df) * len(judges)):.2f}s")
    print(
        f"  Speedup vs sequential (est.): {(len(eval_df) * len(judges) * 0.5) / elapsed_time:.1f}x"
    )
    print(f"{'='*80}\n")

    return comparison_df


def calculate_accuracy_rates(
    df: pd.DataFrame,
    judges: List[Dict],
    model_a_name: str,
    model_b_name: str,
    eval_limit: Optional[int] = None,
) -> Dict:
    """Calculate accuracy rates for each model across all judges."""
    eval_df = df.head(eval_limit) if eval_limit else df

    # Sanitize model names for column lookups
    model_a_col = sanitize_name(model_a_name)
    model_b_col = sanitize_name(model_b_name)

    accuracy_rates = {}
    for judge in judges:
        judge_name = judge["name"]
        correct_a_col = f"judge_{judge_name}_{model_a_col}_correct"
        correct_b_col = f"judge_{judge_name}_{model_b_col}_correct"

        if correct_a_col in eval_df.columns and correct_b_col in eval_df.columns:
            # Count correct answers
            valid_rows = eval_df[eval_df[correct_a_col].notna() & eval_df[correct_b_col].notna()]

            correct_a = valid_rows[correct_a_col].sum()
            correct_b = valid_rows[correct_b_col].sum()
            total = len(valid_rows)

            accuracy_a = (correct_a / total * 100) if total > 0 else 0
            accuracy_b = (correct_b / total * 100) if total > 0 else 0

            accuracy_rates[judge_name] = {
                "model_a_accuracy": accuracy_a,
                "model_b_accuracy": accuracy_b,
                "model_a_correct_count": int(correct_a),
                "model_b_correct_count": int(correct_b),
                "n_samples": total,
            }

    return accuracy_rates


def calculate_position_bias(
    df: pd.DataFrame, judges: List[Dict], model_a_name: str, eval_limit: Optional[int] = None
) -> Dict:
    """Analyze position bias in judge decisions."""
    eval_df = df.head(eval_limit) if eval_limit else df

    # Sanitize model name for column lookups
    model_a_col = sanitize_name(model_a_name)

    position_bias = {}
    for judge in judges:
        judge_name = judge["name"]
        judge_col = f"judge_{judge_name}_decision"
        position_col = f"{model_a_col}_position"

        # Get decisions when model_a was in position A vs position B
        model_a_at_A = eval_df[eval_df[position_col] == "A"]
        model_a_at_B = eval_df[eval_df[position_col] == "B"]

        # Calculate win rates for each position - check if judge chose model_a_name
        a_wins_when_at_A = (model_a_at_A[judge_col] == model_a_name).sum()
        a_wins_when_at_B = (model_a_at_B[judge_col] == model_a_name).sum()

        total_at_A = len(model_a_at_A[model_a_at_A[judge_col].notna()])
        total_at_B = len(model_a_at_B[model_a_at_B[judge_col].notna()])

        if total_at_A > 0 and total_at_B > 0:
            win_rate_at_A = a_wins_when_at_A / total_at_A * 100
            win_rate_at_B = a_wins_when_at_B / total_at_B * 100

            position_bias[judge_name] = {
                "win_rate_at_A": win_rate_at_A,
                "win_rate_at_B": win_rate_at_B,
                "difference": abs(win_rate_at_A - win_rate_at_B),
                "a_wins_when_at_A": int(a_wins_when_at_A),
                "a_wins_when_at_B": int(a_wins_when_at_B),
                "total_at_A": total_at_A,
                "total_at_B": total_at_B,
            }

    return position_bias


def calculate_judge_agreement(
    df: pd.DataFrame,
    judges: List[Dict],
    model_a_name: str,
    model_b_name: str,
    eval_limit: Optional[int] = None,
) -> Dict:
    """Analyze how often judges agree with each other."""
    eval_df = df.head(eval_limit) if eval_limit else df
    judge_cols = [f"judge_{judge['name']}_decision" for judge in judges]

    unanimous_decisions = 0
    majority_model_a = 0
    majority_model_b = 0
    split_decisions = 0

    for idx, row in eval_df.iterrows():
        judgments = [
            row[col]
            for col in judge_cols
            if pd.notna(row[col]) and row[col] not in ["ERROR", "INVALID"]
        ]

        if len(judgments) < len(judges):
            continue

        if len(set(judgments)) == 1:
            unanimous_decisions += 1

        a_votes = judgments.count(model_a_name)
        b_votes = judgments.count(model_b_name)

        if a_votes > b_votes:
            majority_model_a += 1
        elif b_votes > a_votes:
            majority_model_b += 1
        else:
            split_decisions += 1

    total = len(eval_df)
    return {
        "unanimous_decisions": unanimous_decisions,
        "unanimous_rate": unanimous_decisions / total * 100 if total > 0 else 0,
        "majority_model_a": majority_model_a,
        "majority_model_a_rate": majority_model_a / total * 100 if total > 0 else 0,
        "majority_model_b": majority_model_b,
        "majority_model_b_rate": majority_model_b / total * 100 if total > 0 else 0,
        "split_decisions": split_decisions,
        "split_rate": split_decisions / total * 100 if total > 0 else 0,
    }


def create_visualizations(
    win_rates: Dict,
    model_a_name: str,
    model_b_name: str,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Create visualizations of win rates across judges."""
    judge_names = list(win_rates.keys())
    model_a_rates = [win_rates[j]["model_a_win_rate"] for j in judge_names]
    model_b_rates = [win_rates[j]["model_b_win_rate"] for j in judge_names]
    tie_rates = [win_rates[j]["tie_rate"] for j in judge_names]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Stacked bar chart
    x = np.arange(len(judge_names))
    width = 0.6

    axes[0].bar(x, model_a_rates, width, label=model_a_name, color="#3498db")
    axes[0].bar(
        x,
        model_b_rates,
        width,
        bottom=model_a_rates,
        label=model_b_name,
        color="#e74c3c",
    )
    axes[0].bar(
        x,
        tie_rates,
        width,
        bottom=np.array(model_a_rates) + np.array(model_b_rates),
        label="Ties",
        color="#95a5a6",
    )

    axes[0].set_ylabel("Percentage (%)", fontsize=12)
    axes[0].set_title(
        f"Win Rate Distribution by Judge\n{model_a_name} vs {model_b_name}",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(judge_names, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Plot 2: Grouped bar chart for comparison
    x = np.arange(len(judge_names))
    width = 0.35

    axes[1].bar(x - width / 2, model_a_rates, width, label=model_a_name, color="#3498db")
    axes[1].bar(x + width / 2, model_b_rates, width, label=model_b_name, color="#e74c3c")

    axes[1].set_ylabel("Win Rate (%)", fontsize=12)
    axes[1].set_title(
        f"Head-to-Head Win Rates by Judge\n{model_a_name} vs {model_b_name}",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(judge_names, rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")

    return fig


def get_sample_rationales(
    df: pd.DataFrame, judges: List[Dict], model_a_name: str, model_b_name: str, n_samples: int = 5
) -> List[Dict]:
    """Get sample rationales from judges for logging."""
    samples = []

    # Sanitize model names for column lookups
    model_a_col = sanitize_name(model_a_name)
    model_b_col = sanitize_name(model_b_name)

    for idx in df.head(n_samples).index:
        row = df.loc[idx]
        position_col = f"{model_a_col}_position"
        answer_a_col = f"answer_{model_a_col}"
        answer_b_col = f"answer_{model_b_col}"

        sample = {
            "question": row["question"],
            "reference_answer": row["target"],
            "answer_model_a": row.get(answer_a_col, "N/A"),
            "answer_model_b": row.get(answer_b_col, "N/A"),
            "model_a_position": row.get(position_col, "N/A"),
            "rationales": {},
        }

        for judge in judges:
            judge_name = judge["name"]
            decision = row.get(f"judge_{judge_name}_decision", "N/A")
            rationale = row.get(f"judge_{judge_name}_rationale", "No rationale")

            sample["rationales"][judge_name] = {
                "decision": decision,
                "rationale": rationale,
            }

        samples.append(sample)

    return samples


def log_to_wandb(
    config: Dict,
    win_rates: Dict,
    accuracy_rates: Dict,
    position_bias: Dict,
    judge_agreement: Dict,
    comparison_df: pd.DataFrame,
    results_text: str,
    metrics_table: Optional[pd.DataFrame] = None,
    figure: Optional[plt.Figure] = None,
    output_dir: Optional[Path] = None,
):
    """Log results to Weights & Biases."""
    try:
        import wandb
    except ImportError:
        print("Warning: wandb not installed. Skipping wandb logging.")
        return

    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        print("wandb logging disabled in config")
        return

    model_a_name = config["model_a"]["name"]
    model_b_name = config["model_b"]["name"]

    # Initialize wandb
    run_name = format_filename(
        wandb_config.get("run_name", "{model_a}_vs_{model_b}"),
        model_a_name,
        model_b_name,
    )

    wandb.init(
        project=wandb_config.get("project", "win-rate-evaluation"),
        entity=wandb_config.get("entity"),
        name=run_name,
        config={
            "model_a": config["model_a"],
            "model_b": config["model_b"],
            "judges": [j["name"] for j in config["judges"]],
            "evaluation": config["evaluation"],
        },
        tags=wandb_config.get("tags", []),
        notes=wandb_config.get("notes", ""),
    )

    log_config = wandb_config.get("log", {})

    # Sanitize model names for wandb metric names
    model_a_sanitized = sanitize_name(model_a_name)
    model_b_sanitized = sanitize_name(model_b_name)

    # Log win rates
    if log_config.get("win_rates", True):
        for judge_name, stats in win_rates.items():
            wandb.log(
                {
                    f"win_rate/{judge_name}/{model_a_sanitized}": stats["model_a_win_rate"] / 100,
                    f"win_rate/{judge_name}/{model_b_sanitized}": stats["model_b_win_rate"] / 100,
                    f"win_rate/{judge_name}/tie": stats["tie_rate"] / 100,
                    f"win_rate/{judge_name}/{model_a_sanitized}_wins": stats["model_a_wins"],
                    f"win_rate/{judge_name}/{model_b_sanitized}_wins": stats["model_b_wins"],
                    f"win_rate/{judge_name}/ties": stats["ties"],
                }
            )

        # Aggregate win rates
        total_model_a_wins = sum(stats["model_a_wins"] for stats in win_rates.values())
        total_model_b_wins = sum(stats["model_b_wins"] for stats in win_rates.values())
        total_ties = sum(stats["ties"] for stats in win_rates.values())
        total_evaluations = sum(stats["total"] for stats in win_rates.values())

        # Calculate average win rates for wandb
        avg_model_a_win_rate = np.mean([stats["model_a_win_rate"] for stats in win_rates.values()])
        avg_model_b_win_rate = np.mean([stats["model_b_win_rate"] for stats in win_rates.values()])

        wandb.log(
            {
                f"aggregate/{model_a_sanitized}_wins": total_model_a_wins,
                f"aggregate/{model_b_sanitized}_wins": total_model_b_wins,
                "aggregate/ties": total_ties,
                f"aggregate/{model_a_sanitized}_win_rate": (
                    total_model_a_wins / total_evaluations if total_evaluations > 0 else 0
                ),
                f"aggregate/{model_b_sanitized}_win_rate": (
                    total_model_b_wins / total_evaluations if total_evaluations > 0 else 0
                ),
                f"aggregate/avg_{model_a_sanitized}_win_rate": avg_model_a_win_rate / 100,
                f"aggregate/avg_{model_b_sanitized}_win_rate": avg_model_b_win_rate / 100,
                "aggregate/win_rate_difference": (avg_model_b_win_rate - avg_model_a_win_rate)
                / 100,
            }
        )

    # Log accuracy rates
    if log_config.get("accuracy_rates", True):
        for judge_name, stats in accuracy_rates.items():
            wandb.log(
                {
                    f"accuracy/{judge_name}/{model_a_sanitized}": stats["model_a_accuracy"] / 100,
                    f"accuracy/{judge_name}/{model_b_sanitized}": stats["model_b_accuracy"] / 100,
                    f"accuracy/{judge_name}/{model_a_sanitized}_correct": stats[
                        "model_a_correct_count"
                    ],
                    f"accuracy/{judge_name}/{model_b_sanitized}_correct": stats[
                        "model_b_correct_count"
                    ],
                }
            )

        # Overall accuracy
        overall_accuracy_a = np.mean(
            [stats["model_a_accuracy"] for stats in accuracy_rates.values()]
        )
        overall_accuracy_b = np.mean(
            [stats["model_b_accuracy"] for stats in accuracy_rates.values()]
        )

        wandb.log(
            {
                f"aggregate/{model_a_sanitized}_accuracy": overall_accuracy_a / 100,
                f"aggregate/{model_b_sanitized}_accuracy": overall_accuracy_b / 100,
                "aggregate/accuracy_difference": (overall_accuracy_b - overall_accuracy_a) / 100,
            }
        )

    # Log position bias
    if log_config.get("position_bias", True):
        for judge_name, stats in position_bias.items():
            wandb.log(
                {
                    f"position_bias/{judge_name}/{model_a_sanitized}_win_rate_at_A": stats[
                        "win_rate_at_A"
                    ]
                    / 100,
                    f"position_bias/{judge_name}/{model_a_sanitized}_win_rate_at_B": stats[
                        "win_rate_at_B"
                    ]
                    / 100,
                    f"position_bias/{judge_name}/difference": stats["difference"] / 100,
                }
            )

    # Log judge agreement
    if log_config.get("judge_agreement", True):
        wandb.log(
            {
                "judge_agreement/unanimous_decisions": judge_agreement["unanimous_decisions"],
                "judge_agreement/unanimous_rate": judge_agreement["unanimous_rate"] / 100,
                f"judge_agreement/majority_{model_a_sanitized}": judge_agreement[
                    "majority_model_a"
                ],
                f"judge_agreement/majority_{model_b_sanitized}": judge_agreement[
                    "majority_model_b"
                ],
                "judge_agreement/split_decisions": judge_agreement["split_decisions"],
            }
        )

    # Log visualizations
    if log_config.get("visualizations", True) and figure:
        wandb.log({"visualizations/win_rates": wandb.Image(figure)})

    # Log sample rationales
    if log_config.get("sample_rationales", True):
        samples = get_sample_rationales(
            comparison_df,
            config["judges"],
            model_a_name,
            model_b_name,
            log_config.get("sample_count", 5),
        )
        # Create a wandb Table
        columns = [
            "question",
            "reference",
            "model_a_answer",
            "model_b_answer",
            "position",
        ]
        for judge in config["judges"]:
            columns.extend([f"{judge['name']}_decision", f"{judge['name']}_rationale"])

        table_data = []
        for sample in samples:
            # Convert to string and handle cases where values might be floats/NaN
            answer_a = str(sample["answer_model_a"]) if sample["answer_model_a"] != "N/A" else "N/A"
            answer_b = str(sample["answer_model_b"]) if sample["answer_model_b"] != "N/A" else "N/A"

            row = [
                str(sample["question"])[:100] + "...",
                str(sample["reference_answer"])[:100] + "...",
                answer_a[:80] + "...",
                answer_b[:80] + "...",
                sample["model_a_position"],
            ]
            for judge in config["judges"]:
                judge_name = judge["name"]
                row.extend(
                    [
                        sample["rationales"][judge_name]["decision"],
                        sample["rationales"][judge_name]["rationale"][:200] + "...",
                    ]
                )
            table_data.append(row)

        wandb.log({"sample_rationales": wandb.Table(columns=columns, data=table_data)})

    # Log console output as text
    if results_text:
        import html

        escaped_text = html.escape(results_text)
        wandb.log({"results_summary": wandb.Html(f"<pre>{escaped_text}</pre>")})

    # Log metrics table
    if log_config.get("metrics_table", True) and metrics_table is not None:
        # Convert metrics table to wandb Table
        wandb.log({"metrics_table": wandb.Table(dataframe=metrics_table)})

    # Log raw results
    if log_config.get("raw_results", True) and output_dir:
        results_file = output_dir / format_filename(
            config["output"]["results_filename"], model_a_name, model_b_name
        )
        summary_file = output_dir / format_filename(
            config["output"]["summary_filename"], model_a_name, model_b_name
        )
        metrics_file = output_dir / format_filename(
            config["output"].get("metrics_filename", "metrics_{model_a}_vs_{model_b}.csv"),
            model_a_name,
            model_b_name,
        )

        if results_file.exists():
            wandb.save(str(results_file))
        if summary_file.exists():
            wandb.save(str(summary_file))
        if metrics_file.exists():
            wandb.save(str(metrics_file))

    # Get URL before finishing (wandb.run becomes None after finish())
    run_url = wandb.run.get_url() if wandb.run else None
    wandb.finish()

    if run_url:
        print(f"\nResults logged to wandb: {run_url}")


def create_metrics_table(
    config: Dict,
    win_rates: Dict,
    accuracy_rates: Dict,
    position_bias: Dict,
    judge_agreement: Dict,
) -> pd.DataFrame:
    """Create a standardized metrics table with model-level metrics.

    Returns a DataFrame with columns: model_name, task, subtask, opponent, metric, value
    """
    model_a_name = config["model_a"]["name"]
    model_b_name = config["model_b"]["name"]

    metrics_rows = []

    # Task type should be specified in config (e.g., "open_ended", "open_ended_w_context", etc.)
    task = config.get("task", "open_ended")
    subtask = "win_rate"
    opponent = model_a_name

    # Calculate aggregate metrics
    total_model_a_wins = sum(stats["model_a_wins"] for stats in win_rates.values())
    total_model_b_wins = sum(stats["model_b_wins"] for stats in win_rates.values())
    total_ties = sum(stats["ties"] for stats in win_rates.values())
    total_evaluations = sum(stats["total"] for stats in win_rates.values())

    avg_model_a_win_rate = np.mean([stats["model_a_win_rate"] for stats in win_rates.values()])
    avg_model_b_win_rate = np.mean([stats["model_b_win_rate"] for stats in win_rates.values()])

    # Aggregate accuracy
    overall_accuracy_a = np.mean([stats["model_a_accuracy"] for stats in accuracy_rates.values()])
    overall_accuracy_b = np.mean([stats["model_b_accuracy"] for stats in accuracy_rates.values()])

    # Add metrics for model_b (the model being evaluated against model_a)
    metrics_rows.extend(
        [
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": "avg_win_rate",
                "value": avg_model_b_win_rate / 100,
            },
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": "total_wins",
                "value": total_model_b_wins,
            },
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": "total_ties",
                "value": total_ties,
            },
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": "total_evaluations",
                "value": total_evaluations,
            },
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": "avg_accuracy",
                "value": overall_accuracy_b / 100,
            },
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": "win_rate_difference_vs_opponent",
                "value": (avg_model_b_win_rate - avg_model_a_win_rate) / 100,
            },
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": "accuracy_difference_vs_opponent",
                "value": (overall_accuracy_b - overall_accuracy_a) / 100,
            },
        ]
    )

    # Add per-judge metrics for model_b
    for judge_name, stats in win_rates.items():
        metrics_rows.append(
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": f"win_rate_judge_{judge_name}",
                "value": stats["model_b_win_rate"] / 100,
            }
        )

    for judge_name, stats in accuracy_rates.items():
        metrics_rows.append(
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": f"accuracy_judge_{judge_name}",
                "value": stats["model_b_accuracy"] / 100,
            }
        )

    # Add judge agreement metrics
    metrics_rows.extend(
        [
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": "unanimous_decisions_rate",
                "value": judge_agreement["unanimous_rate"] / 100,
            },
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": f"majority_wins_rate",
                "value": judge_agreement["majority_model_b_rate"] / 100,
            },
        ]
    )

    # Add average position bias if available
    if position_bias:
        avg_position_diff = np.mean([stats["difference"] for stats in position_bias.values()])
        metrics_rows.append(
            {
                "model_name": model_b_name,
                "task": task,
                "subtask": subtask,
                "opponent": opponent,
                "metric": "avg_position_bias_difference",
                "value": avg_position_diff / 100,
            }
        )

    return pd.DataFrame(metrics_rows)


def print_results(
    config: Dict,
    win_rates: Dict,
    accuracy_rates: Dict,
    position_bias: Dict,
    judge_agreement: Dict,
) -> str:
    """Print results to console and return as string for wandb logging."""
    model_a_name = config["model_a"]["name"]
    model_b_name = config["model_b"]["name"]

    # Collect output in a list
    output_lines = []

    def print_and_log(text=""):
        """Helper to both print and collect output."""
        print(text)
        output_lines.append(text)

    # Win rates
    print_and_log("\n" + "=" * 80)
    print_and_log(f"WIN RATE RESULTS: {model_a_name} vs {model_b_name}")
    print_and_log("=" * 80 + "\n")

    for judge_name, stats in win_rates.items():
        print_and_log(f"Judge: {judge_name}")
        print_and_log(
            f"  {model_a_name} wins: {stats['model_a_wins']:4d} ({stats['model_a_win_rate']/100:.3f})"
        )
        print_and_log(
            f"  {model_b_name} wins: {stats['model_b_wins']:4d} ({stats['model_b_win_rate']/100:.3f})"
        )
        print_and_log(f"  Ties:         {stats['ties']:4d} ({stats['tie_rate']/100:.3f})")
        print_and_log(f"  Errors:       {stats['errors']:4d}")
        print_and_log(f"  Total:        {stats['total']:4d}")
        print_and_log()

    # Aggregate win rates
    total_model_a_wins = sum(stats["model_a_wins"] for stats in win_rates.values())
    total_model_b_wins = sum(stats["model_b_wins"] for stats in win_rates.values())
    total_ties = sum(stats["ties"] for stats in win_rates.values())
    total_evaluations = sum(stats["total"] for stats in win_rates.values())

    print_and_log("\n" + "=" * 80)
    print_and_log(f"AGGREGATE RESULTS (Across All Judges)")
    print_and_log("=" * 80 + "\n")
    print_and_log(
        f"{model_a_name} wins: {total_model_a_wins:4d} ({total_model_a_wins/total_evaluations:.3f})"
    )
    print_and_log(
        f"{model_b_name} wins: {total_model_b_wins:4d} ({total_model_b_wins/total_evaluations:.3f})"
    )
    print_and_log(f"Ties:         {total_ties:4d} ({total_ties/total_evaluations:.3f})")
    print_and_log(f"Total:        {total_evaluations:4d}")

    # Calculate average win rate difference across judges
    avg_model_a_win_rate = np.mean([stats["model_a_win_rate"] for stats in win_rates.values()])
    avg_model_b_win_rate = np.mean([stats["model_b_win_rate"] for stats in win_rates.values()])
    win_rate_difference = (avg_model_b_win_rate - avg_model_a_win_rate) / 100

    print_and_log(f"\nAverage win rate across judges:")
    print_and_log(f"  {model_a_name}: {avg_model_a_win_rate/100:.3f}")
    print_and_log(f"  {model_b_name}: {avg_model_b_win_rate/100:.3f}")
    print_and_log(f"  Difference ({model_b_name} - {model_a_name}): {win_rate_difference:+.3f}")

    # Accuracy rates
    print_and_log("\n" + "=" * 80)
    print_and_log(f"ACCURACY RATE ANALYSIS")
    print_and_log("=" * 80 + "\n")

    for judge_name, stats in accuracy_rates.items():
        print_and_log(f"Judge: {judge_name}")
        print_and_log(
            f"  {model_a_name} accuracy: {stats['model_a_accuracy']/100:.3f} ({stats['model_a_correct_count']}/{stats['n_samples']} correct)"
        )
        print_and_log(
            f"  {model_b_name} accuracy: {stats['model_b_accuracy']/100:.3f} ({stats['model_b_correct_count']}/{stats['n_samples']} correct)"
        )
        print_and_log(
            f"  Difference: {(stats['model_a_accuracy'] - stats['model_b_accuracy'])/100:+.3f}"
        )
        print_and_log()

    # Overall accuracy
    overall_accuracy_a = np.mean([stats["model_a_accuracy"] for stats in accuracy_rates.values()])
    overall_accuracy_b = np.mean([stats["model_b_accuracy"] for stats in accuracy_rates.values()])

    print_and_log("=" * 80)
    print_and_log("OVERALL ACCURACY (Across All Judges)")
    print_and_log("=" * 80)
    print_and_log(f"  {model_a_name} accuracy: {overall_accuracy_a/100:.3f}")
    print_and_log(f"  {model_b_name} accuracy: {overall_accuracy_b/100:.3f}")
    print_and_log(f"  Difference: {(overall_accuracy_a - overall_accuracy_b)/100:+.3f}")
    print_and_log()

    # Judge agreement
    print_and_log("\n" + "=" * 80)
    print_and_log("JUDGE AGREEMENT ANALYSIS")
    print_and_log("=" * 80 + "\n")
    print_and_log(
        f"Unanimous decisions: {judge_agreement['unanimous_decisions']} ({judge_agreement['unanimous_rate']/100:.3f})"
    )
    print_and_log(
        f"Majority for {model_a_name}: {judge_agreement['majority_model_a']} ({judge_agreement['majority_model_a_rate']/100:.3f})"
    )
    print_and_log(
        f"Majority for {model_b_name}: {judge_agreement['majority_model_b']} ({judge_agreement['majority_model_b_rate']/100:.3f})"
    )
    print_and_log(
        f"Split decisions:     {judge_agreement['split_decisions']} ({judge_agreement['split_rate']/100:.3f})"
    )

    # Position bias
    print_and_log("\n" + "=" * 80)
    print_and_log("POSITION BIAS ANALYSIS")
    print_and_log("=" * 80 + "\n")

    for judge_name, stats in position_bias.items():
        print_and_log(f"Judge: {judge_name}")
        print_and_log(
            f"  {model_a_name} win rate at position A: {stats['win_rate_at_A']/100:.3f} ({stats['a_wins_when_at_A']}/{stats['total_at_A']})"
        )
        print_and_log(
            f"  {model_a_name} win rate at position B: {stats['win_rate_at_B']/100:.3f} ({stats['a_wins_when_at_B']}/{stats['total_at_B']})"
        )
        print_and_log(f"  Difference: {stats['difference']/100:.3f}")
        print_and_log()

    # Return the collected output as a single string
    return "\n".join(output_lines)


def run_single_evaluation(
    config: Dict,
    model_a_config: Dict,
    model_b_config: Dict,
    combination_idx: int,
    total_combinations: int,
) -> None:
    """Run evaluation for a single model_a vs model_b combination.

    Args:
        config: Full configuration dictionary
        model_a_config: Configuration for model A
        model_b_config: Configuration for model B
        combination_idx: Current combination index (1-based)
        total_combinations: Total number of combinations to run
    """
    model_a_name = model_a_config["name"]
    model_b_name = model_b_config["name"]

    print("\n" + "=" * 80)
    print(f"EVALUATION {combination_idx}/{total_combinations}: {model_a_name} vs {model_b_name}")
    print("=" * 80)

    # Create a temporary config with the current model combination
    eval_config = config.copy()
    eval_config["model_a"] = model_a_config
    eval_config["model_b"] = model_b_config

    # Load and align data
    comparison_df = load_and_align_data(eval_config)

    # Run evaluation
    comparison_df = run_evaluation(eval_config, comparison_df)

    # Calculate metrics
    eval_limit = config["evaluation"].get("limit")

    print("Calculating win rates...")
    win_rates = calculate_win_rates(
        comparison_df.head(eval_limit) if eval_limit else comparison_df,
        config["judges"],
        model_a_name,
        model_b_name,
    )

    print("Calculating accuracy rates...")
    accuracy_rates = calculate_accuracy_rates(
        comparison_df, config["judges"], model_a_name, model_b_name, eval_limit
    )

    print("Analyzing position bias...")
    position_bias = calculate_position_bias(
        comparison_df, config["judges"], model_a_name, eval_limit
    )

    print("Analyzing judge agreement...")
    judge_agreement = calculate_judge_agreement(
        comparison_df, config["judges"], model_a_name, model_b_name, eval_limit
    )

    # Create output directory
    output_config = config.get("output", {})
    output_dir = Path(output_config.get("output_dir", "win_rate_results"))
    output_dir.mkdir(exist_ok=True)

    # Save results
    if output_config.get("save_results", True):
        results_file = output_dir / format_filename(
            output_config.get("results_filename", "results_{model_a}_vs_{model_b}.csv"),
            model_a_name,
            model_b_name,
        )
        eval_df = comparison_df.head(eval_limit) if eval_limit else comparison_df
        eval_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")

        # Save summary statistics
        summary_file = output_dir / format_filename(
            output_config.get("summary_filename", "summary_{model_a}_vs_{model_b}.csv"),
            model_a_name,
            model_b_name,
        )
        summary_df = pd.DataFrame(win_rates).T
        summary_df.to_csv(summary_file)
        print(f"Summary statistics saved to: {summary_file}")

    # Create and save visualizations
    figure = None
    if output_config.get("save_visualizations", True):
        viz_file = output_dir / format_filename(
            output_config.get("visualization_filename", "comparison_{model_a}_vs_{model_b}.png"),
            model_a_name,
            model_b_name,
        )
        figure = create_visualizations(win_rates, model_a_name, model_b_name, viz_file)

    # Create and save metrics table
    print("Creating metrics table...")
    metrics_table = create_metrics_table(
        eval_config, win_rates, accuracy_rates, position_bias, judge_agreement
    )

    if output_config.get("save_metrics_table", True):
        metrics_file = output_dir / format_filename(
            output_config.get("metrics_filename", "metrics_{model_a}_vs_{model_b}.csv"),
            model_a_name,
            model_b_name,
        )
        metrics_table.to_csv(metrics_file, index=False)
        print(f"Metrics table saved to: {metrics_file}")

    # Print results to console and capture output
    results_text = print_results(
        eval_config, win_rates, accuracy_rates, position_bias, judge_agreement
    )

    # Log to wandb
    print("\nLogging to wandb...")
    log_to_wandb(
        eval_config,
        win_rates,
        accuracy_rates,
        position_bias,
        judge_agreement,
        comparison_df,
        results_text,
        metrics_table,
        figure,
        output_dir,
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Win rate evaluation using multiple LLM judges")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    print("=" * 80)
    print("WIN RATE EVALUATION")
    print("=" * 80)
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Generate all model combinations
    combinations = generate_model_combinations(config)

    # Run evaluation for each combination
    for idx, (model_a_config, model_b_config) in enumerate(combinations, start=1):
        try:
            run_single_evaluation(
                config,
                model_a_config,
                model_b_config,
                combination_idx=idx,
                total_combinations=len(combinations),
            )
        except Exception as e:
            model_a_name = model_a_config.get("name", "unknown")
            model_b_name = model_b_config.get("name", "unknown")
            print(
                f"\nERROR in evaluation {idx}/{len(combinations)} ({model_a_name} vs {model_b_name}):"
            )
            print(f"  {str(e)}")
            print("  Continuing to next combination...\n")
            continue

    print("\n" + "=" * 80)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 80)
    print(f"Successfully completed {len(combinations)} evaluation(s)")


if __name__ == "__main__":
    main()
