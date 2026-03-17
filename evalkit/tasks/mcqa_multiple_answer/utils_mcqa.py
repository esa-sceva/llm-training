import re
import sys
from pathlib import Path

import datasets
from lm_eval.api.filter import Filter

# Add parent directory to path to import common MCQA utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcqa_utils import extract_labels


def subset_accuracy(references, predictions):
    correct_count = 0
    for correct, pred in zip(references, predictions):
        if set(correct) == set(pred):
            correct_count += 1
    return correct_count / len(references)


def jaccard_index(references, predictions):
    jaccard_scores = []
    for correct, pred in zip(references, predictions):
        intersection = len(set(correct) & set(pred))
        union = len(set(correct) | set(pred))
        jaccard_scores.append(intersection / union)

    if len(jaccard_scores) == 0:
        return 0
    return sum(jaccard_scores) / len(jaccard_scores)


def map_to_answers(row):
    return {"output": ", ".join(row["Answers"])}


def doc_to_text(doc):
    string = f"{doc['Question']}\n"

    for label, txt in zip(doc["Choices"]["label"], doc["Choices"]["text"]):
        string += f"{label}. {txt}\n"

    return string


def process_answer(answer):
    """
    Process model answer to extract letter choices.

    This is a wrapper around the common extract_labels function from mcqa_utils.

    Handles multiple formats:
    - Simple letters: "A", "B, C", "A,B,C"
    - Letter with period: "D.", "A. B."
    - Letter with full text: "D. Sea ice forms on ocean water..."
    - Multiple letters with text: "A. text, B. more text"
    - With prefix: "Answer: C. SIRAL"
    """
    return extract_labels(answer)


class filter_answer(Filter):
    """
    Filter class for lm-eval harness filter_list.

    Extracts letter choices and returns them as a comma-separated string
    for display in filtered_resps.
    """

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        """
        Apply the filter to extract answer labels.

        Args:
            resps: List of lists of model responses
            docs: List of document dictionaries (not used here)

        Returns:
            List of lists of filtered responses (comma-separated labels)
        """
        def filter_set(inst):
            filtered = []
            for resp in inst:
                if not isinstance(resp, str):
                    resp = ""
                labels = extract_labels(resp)
                filtered_resp = ", ".join(labels) if labels else ""
                filtered.append(filtered_resp)
            return filtered

        filtered_resps = [filter_set(resp) for resp in resps]
        return filtered_resps


def process_dataset(dataset: datasets.Dataset):
    return dataset.map(map_to_answers)


def process_results(doc: datasets.Dataset, results):
    """
    Process results for multiple-answer MCQA task.

    Args:
        doc: Document containing ground truth answers
        results: Model predictions (list of responses or filtered responses)

    Returns:
        dict: Metrics (accuracy and IoU)
    """
    pred = results[0]
    references = doc["Answers"]

    # Handle different result formats
    if isinstance(pred, list):
        # If it's a list, take the first element (filtered response)
        pred_text = pred[0] if pred else ""
    else:
        # It's already a string
        pred_text = pred

    # Check if already filtered (comma-separated letters) or needs processing
    if isinstance(pred_text, str) and pred_text.strip():
        # Check if it looks like filtered answer (e.g., "A, C" or "A,C")
        cleaned = pred_text.strip().replace(" ", "").replace(",", "")
        if cleaned.isalpha() and len(cleaned) <= 10:
            # Already filtered, parse comma-separated letters
            preds = [letter.strip() for letter in pred_text.split(",") if letter.strip()]
        else:
            # Raw response, needs processing
            preds = process_answer(pred_text)
    else:
        # Raw response, needs processing
        preds = process_answer(pred_text) if pred_text else []

    subset_acc = subset_accuracy([references], [preds])
    jaccard = jaccard_index([references], [preds])
    return {"acc": subset_acc, "IoU": jaccard}
