"""
Utilities for win rate evaluation with LLM judges.

This module provides:
- Structured output models for judge responses
- Parallel evaluation functions
- Judge query functions with retry logic
"""

import time
import random
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


class JudgeResponse(BaseModel):
    """Structured response from a judge."""

    choice: str = Field(description="The judge's choice: 'A', 'B', or 'TIE'")
    rationale: str = Field(description="Explanation for why this answer is better")


class DoubleEvaluation(BaseModel):
    """Structured response for double evaluation."""

    answer_a_correct: bool = Field(description="True if answer A is correct, False otherwise")
    answer_b_correct: bool = Field(description="True if answer B is correct, False otherwise")
    rationale: str = Field(description="Explanation for the evaluations")


@dataclass
class EvaluationResult:
    """Result from evaluating a single question with a single judge."""

    question_idx: int
    judge_name: str
    raw_choice: str  # The choice the judge gave (A, B, or TIE)
    actual_result: str  # Mapped to actual model (considering randomization)
    rationale: str
    model_a_position: str  # 'A' or 'B'
    model_a_correct: Optional[bool] = None  # Whether model A's answer is correct
    model_b_correct: Optional[bool] = None  # Whether model B's answer is correct
    score_a: Optional[int] = None  # Score for answer A (from judge's perspective) - deprecated
    score_b: Optional[int] = None  # Score for answer B (from judge's perspective) - deprecated
    model_a_score: Optional[int] = None  # Actual model A's score (after mapping) - deprecated
    model_b_score: Optional[int] = None  # Actual model B's score (after mapping) - deprecated
    error: Optional[str] = None


def create_judge_prompt(question: str, reference_answer: str, answer_a: str, answer_b: str) -> str:
    """
    Create a prompt for the judge to compare two answers.

    Args:
        question: The question being answered
        reference_answer: The ground truth answer
        answer_a: First answer to compare
        answer_b: Second answer to compare

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are evaluating answers about Satellite Communication (SatCom).

**Task**: Determine if each answer is CORRECT or INCORRECT by comparing against the reference answer.

**Focus on Correctness, NOT Form**:
- Evaluate based on factual accuracy, not writing style, length, or formatting
- A short, accurate answer is just as correct as a long, detailed one

**Evaluation Criteria**:

An answer is **CORRECT (True)** if:
- It captures the key factual information from the reference answer
- Domain terminology is used correctly (e.g., "SNR" = Signal-to-Noise Ratio, "BER" = Bit Error Rate, "EIRP" = Effective Isotropic Radiated Power, "CNR" = Carrier-to-Noise Ratio)
- No major factual errors or contradictions with the reference
- The core meaning aligns with the reference, regardless of length or style

An answer is **INCORRECT (False)** if:
- It contradicts the reference answer on key facts
- It misinterprets domain terminology
- Contains major factual errors or contradictions
- Misses critical information that fundamentally changes the answer
- States "I don't know" or equivalent when a factual answer is expected

**Important**: Judge based on correctness only. Do not penalize for being concise or reward for being verbose.

Question: "{question}"

Reference Answer: "{reference_answer}"

Answer A: "{answer_a}"
Answer B: "{answer_b}"

Provide your evaluation with:
- answer_a_correct: boolean (true if Answer A is correct, false otherwise)
- answer_b_correct: boolean (true if Answer B is correct, false otherwise)
- rationale: string explaining your reasoning
"""

    return prompt


def query_judge_structured(
    judge_config: Dict, prompt: str, max_retries: int = 3
) -> Tuple[str, str, Optional[bool], Optional[bool]]:
    """
    Query a judge with structured output and retry logic.

    Args:
        judge_config: Dictionary with judge configuration (name, model, api_key, base_url)
        prompt: The evaluation prompt
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (choice, rationale, answer_a_correct, answer_b_correct)
    """
    import random as _rng
    if not hasattr(query_judge_structured, "_printed"):
        query_judge_structured._printed = False

    if not query_judge_structured._printed or _rng.random() < 0.001:
        print(f"\n{'='*60}")
        print(f"[JUDGE PROMPT] → {judge_config['name']}")
        print(f"{'='*60}")
        print(prompt)
        print(f"{'='*60}\n")
        query_judge_structured._printed = True

    client = OpenAI(api_key=judge_config["api_key"], base_url=judge_config["base_url"])

    for attempt in range(max_retries):
        # Increase temperature on retries to help with Pydantic validation errors
        # Start at 0.0, then 0.4, then 0.7
        temperature = min(0.35 * attempt, 0.9)

        try:
            response = client.beta.chat.completions.parse(
                model=judge_config["model"],
                messages=[{"role": "user", "content": prompt}],
                response_format=DoubleEvaluation,
                temperature=temperature,
                max_tokens=10000,
            )

            parsed = response.choices[0].message.parsed

            if (
                parsed
                and hasattr(parsed, "answer_a_correct")
                and hasattr(parsed, "answer_b_correct")
            ):
                a_correct = parsed.answer_a_correct
                b_correct = parsed.answer_b_correct
                rationale = parsed.rationale.strip() if parsed.rationale else ""

                # Determine winner based on correctness
                # A wins if A is correct and B is not
                # B wins if B is correct and A is not
                # Otherwise it's a TIE
                if a_correct and not b_correct:
                    choice = "A"
                elif b_correct and not a_correct:
                    choice = "B"
                else:
                    choice = "TIE"

                return choice, rationale, a_correct, b_correct

            # If we got here, response was unclear
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return "INVALID", "Response format was invalid", None, None

        except Exception as e:
            # Check if it's a Pydantic validation error
            is_pydantic_error = "validation error" in str(e).lower() or "pydantic" in str(e).lower()

            if is_pydantic_error and attempt < max_retries - 1:
                next_temp = min(0.35 * (attempt + 1), 0.9)
                print(f"Pydantic validation error for {judge_config['name']}: {str(e)[:150]}... - Retrying with temperature {next_temp:.2f}")
                time.sleep(2**attempt)  # Exponential backoff
            elif attempt < max_retries - 1:
                error_msg = f"Error querying {judge_config['name']}: {e}"
                print(error_msg)
                time.sleep(2**attempt)  # Exponential backoff
            else:
                error_msg = f"Error querying {judge_config['name']}: {e}"
                print(f"{error_msg} - Max retries exceeded")
                return "ERROR", error_msg, None, None

    return "ERROR", "Max retries exceeded", None, None


def evaluate_single_question_judge(
    idx: int,
    question: str,
    target: str,
    answer_model_a: str,
    answer_model_b: str,
    judge_config: Dict,
    rate_limit_delay: float = 0.5,
) -> EvaluationResult:
    """
    Evaluate a single question with a single judge.

    Args:
        idx: Question index
        question: The question text
        target: Reference answer
        answer_model_a: Answer from model A
        answer_model_b: Answer from model B
        judge_config: Judge configuration
        rate_limit_delay: Delay in seconds between API calls

    Returns:
        EvaluationResult object
    """
    # Randomly decide answer order to prevent position bias
    model_a_is_first = random.choice([True, False])

    if model_a_is_first:
        answer_a = answer_model_a
        answer_b = answer_model_b
        model_a_position = "A"
    else:
        answer_a = answer_model_b
        answer_b = answer_model_a
        model_a_position = "B"

    # Create prompt and query judge
    prompt = create_judge_prompt(question, target, answer_a, answer_b)
    judge_response, rationale, a_correct, b_correct = query_judge_structured(judge_config, prompt)

    # Map the judge's response back to the actual model
    if judge_response in ["A", "B", "TIE"]:
        if model_a_is_first:
            # model_a was A, model_b was B
            actual_result = judge_response
            model_a_correct = a_correct
            model_b_correct = b_correct
        else:
            # model_a was B, model_b was A - flip the response and correctness
            if judge_response == "A":
                actual_result = "B"
            elif judge_response == "B":
                actual_result = "A"
            else:
                actual_result = "TIE"
            model_a_correct = b_correct
            model_b_correct = a_correct
    else:
        # ERROR or INVALID - keep as is
        actual_result = judge_response
        model_a_correct = a_correct if model_a_is_first else b_correct
        model_b_correct = b_correct if model_a_is_first else a_correct

    # Rate limiting
    time.sleep(rate_limit_delay)

    return EvaluationResult(
        question_idx=idx,
        judge_name=judge_config["name"],
        raw_choice=judge_response,
        actual_result=actual_result,
        rationale=rationale,
        model_a_position=model_a_position,
        model_a_correct=model_a_correct,
        model_b_correct=model_b_correct,
        score_a=None,  # Deprecated
        score_b=None,  # Deprecated
        model_a_score=None,  # Deprecated
        model_b_score=None,  # Deprecated
        error=None if judge_response not in ["ERROR", "INVALID"] else rationale,
    )


def evaluate_parallel(
    questions_df,
    judges: List[Dict],
    model_a_name: str,
    model_b_name: str,
    max_workers: int = 10,
    rate_limit_delay: float = 0.1,
    progress_bar: bool = True,
) -> List[EvaluationResult]:
    """
    Evaluate all questions with all judges in parallel.

    Args:
        questions_df: DataFrame with columns: question, target, answer_{model_a}, answer_{model_b}
        judges: List of judge configurations
        model_a_name: Name of model A
        model_b_name: Name of model B
        max_workers: Maximum number of parallel workers
        rate_limit_delay: Delay between API calls (per thread)
        progress_bar: Whether to show progress bar

    Returns:
        List of EvaluationResult objects
    """
    results = []
    total_tasks = len(questions_df) * len(judges)

    # Get column names with sanitized model names
    model_a_col = _sanitize_column_name(model_a_name)
    model_b_col = _sanitize_column_name(model_b_name)
    answer_a_col = f"answer_{model_a_col}"
    answer_b_col = f"answer_{model_b_col}"

    # Create all tasks
    tasks = []
    for idx, row in questions_df.iterrows():
        for judge in judges:
            tasks.append(
                (
                    idx,
                    row["question"],
                    row["target"],
                    row[answer_a_col],
                    row[answer_b_col],
                    judge,
                    rate_limit_delay,
                )
            )

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(evaluate_single_question_judge, *task): task for task in tasks
        }

        # Collect results with progress bar
        iterator = as_completed(future_to_task)
        if progress_bar:
            iterator = tqdm(iterator, total=total_tasks, desc="Evaluating")

        for future in iterator:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = future_to_task[future]
                print(f"Task failed for question {task[0]}, judge {task[5]['name']}: {e}")
                # Create error result
                results.append(
                    EvaluationResult(
                        question_idx=task[0],
                        judge_name=task[5]["name"],
                        raw_choice="ERROR",
                        actual_result="ERROR",
                        rationale=str(e),
                        model_a_position="UNKNOWN",
                        error=str(e),
                    )
                )

    return results


def _sanitize_column_name(name: str) -> str:
    """Sanitize model name for use in column names."""
    return name.lower().replace(" ", "_").replace(".", "").replace("-", "_").replace("/", "_")


def results_to_dataframe(results: List[EvaluationResult], base_df, model_a_name: str = "model_a", model_b_name: str = "model_b"):
    """
    Convert evaluation results to DataFrame format.

    Args:
        results: List of EvaluationResult objects
        base_df: Base DataFrame to add results to
        model_a_name: Name of model A (for column naming)
        model_b_name: Name of model B (for column naming)

    Returns:
        Updated DataFrame with judge columns
    """
    import pandas as pd

    # Create a copy of the base dataframe
    df = base_df.copy()

    # Sanitize model names for column naming
    model_a_col = _sanitize_column_name(model_a_name)
    model_b_col = _sanitize_column_name(model_b_name)

    # Get unique judges
    judges = list(set(r.judge_name for r in results))

    # Initialize columns with actual model names
    for judge in judges:
        df[f"judge_{judge}_decision"] = None
        df[f"judge_{judge}_rationale"] = None
        df[f"judge_{judge}_{model_a_col}_correct"] = None
        df[f"judge_{judge}_{model_b_col}_correct"] = None

    df[f"{model_a_col}_position"] = None

    # Fill in results - map A/B to actual model names
    for result in results:
        idx = result.question_idx

        # Map decision from A/B/TIE to actual model names
        if result.actual_result == "A":
            decision = model_a_name
        elif result.actual_result == "B":
            decision = model_b_name
        else:
            decision = result.actual_result  # TIE, ERROR, INVALID

        df.loc[idx, f"judge_{result.judge_name}_decision"] = decision
        df.loc[idx, f"judge_{result.judge_name}_rationale"] = result.rationale
        df.loc[idx, f"judge_{result.judge_name}_{model_a_col}_correct"] = result.model_a_correct
        df.loc[idx, f"judge_{result.judge_name}_{model_b_col}_correct"] = result.model_b_correct
        df.loc[idx, f"{model_a_col}_position"] = result.model_a_position

    return df


def calculate_win_rates(df, judges: List[Dict], model_a_name: str, model_b_name: str) -> Dict:
    """
    Calculate win rates for each judge.

    Args:
        df: DataFrame with judge columns
        judges: List of judge configurations
        model_a_name: Name of model A
        model_b_name: Name of model B

    Returns:
        Dictionary with win rate statistics per judge
    """
    results = {}

    for judge in judges:
        judge_col = f"judge_{judge['name']}_decision"

        if judge_col not in df.columns:
            continue

        judge_results = df[judge_col].value_counts()
        total = len(df[df[judge_col].notna()])

        if total > 0:
            results[judge["name"]] = {
                "model_a_wins": judge_results.get(model_a_name, 0),
                "model_b_wins": judge_results.get(model_b_name, 0),
                "ties": judge_results.get("TIE", 0),
                "errors": judge_results.get("ERROR", 0) + judge_results.get("INVALID", 0),
                "total": total,
                "model_a_win_rate": judge_results.get(model_a_name, 0) / total * 100,
                "model_b_win_rate": judge_results.get(model_b_name, 0) / total * 100,
                "tie_rate": judge_results.get("TIE", 0) / total * 100,
            }

    return results
