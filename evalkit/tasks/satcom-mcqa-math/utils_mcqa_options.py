"""
Utilities for MCQA tasks with auto-detected dataset format.

Supports two schemas:
  1. Option columns: Questions, Option1-Option5, Answer="Option N"
  2. Choices list:   question, choices=[], answer=int (0-based index)
"""

import re
import sys
from pathlib import Path

import datasets
from lm_eval.api.filter import Filter

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcqa_utils import extract_labels

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def filter_math_reasoning(dataset):
    """Filter dataset to only include questions requiring mathematical reasoning."""
    return dataset.filter(
        lambda doc: str(doc.get("Mathematical Reasoning Required", "")).strip().lower() == "yes"
    )


def _answer_to_letter(doc):
    """Convert answer field to a letter label, auto-detecting format."""
    answer = doc.get("answer") if "answer" in doc else doc.get("Answer", "")
    if answer is None:
        return ""
    if isinstance(answer, int):
        return LABELS[answer] if 0 <= answer < len(LABELS) else ""
    answer_str = str(answer).strip()
    if not answer_str:
        return ""
    option_match = re.match(r"(?i)option\s*(\d+)", answer_str)
    if option_match:
        idx = int(option_match.group(1))
        return LABELS[idx - 1] if 1 <= idx <= len(LABELS) else ""
    if answer_str.isdigit():
        idx = int(answer_str)
        if "choices" in doc:
            return LABELS[idx] if 0 <= idx < len(LABELS) else ""
        else:
            return LABELS[idx - 1] if 1 <= idx <= len(LABELS) else ""
    if len(answer_str) == 1 and answer_str.upper() in LABELS:
        return answer_str.upper()
    return ""


def process_answer(answer):
    """Extract letter choice from model output using shared mcqa_utils."""
    labels = extract_labels(answer)
    return labels[0] if labels else ""


class filter_answer(Filter):
    """Filter class for lm-eval harness filter_list."""

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        def filter_set(inst):
            filtered = []
            for resp in inst:
                if not isinstance(resp, str):
                    resp = ""
                labels = extract_labels(resp)
                filtered_resp = labels[0] if labels else ""
                filtered.append(filtered_resp)
            return filtered

        return [filter_set(resp) for resp in resps]


def process_results(doc: datasets.Dataset, results):
    """
    Process results for single-answer MCQA task.

    Auto-detects answer format and compares with the model's predicted letter.
    """
    pred = results[0]

    if isinstance(pred, list):
        pred_text = pred[0] if pred else ""
    else:
        pred_text = pred

    if isinstance(pred_text, str) and len(pred_text) <= 5 and pred_text.strip():
        cleaned = pred_text.strip().replace(" ", "").replace(",", "")
        if cleaned.isalpha() and len(cleaned) <= 3:
            pred_label = cleaned[0]
        else:
            pred_label = process_answer(pred_text)
    else:
        pred_label = process_answer(pred_text) if pred_text else ""

    ref_label = _answer_to_letter(doc)

    acc = 1.0 if pred_label == ref_label else 0.0

    return {"acc": acc}
