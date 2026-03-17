import re
import sys
from pathlib import Path

import datasets
from lm_eval.api.filter import Filter

# Add parent directory to path to import common MCQA utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcqa_utils import extract_labels


def accuracy(references, predictions):
    """
    Calculate accuracy for single-answer MCQA.

    Args:
        references: List of single correct answer letters (e.g., ['A', 'B', 'C'])
        predictions: List of predicted answer letters (e.g., ['A', 'B', 'D'])

    Returns:
        float: Accuracy score
    """
    correct_count = 0
    for correct, pred in zip(references, predictions):
        # For single answer, take first prediction if multiple provided
        pred_first = pred[0] if isinstance(pred, list) and pred else pred
        if pred_first == correct:
            correct_count += 1
    return correct_count / len(references) if references else 0


def doc_to_text(doc):
    """
    Format document into MCQA prompt.

    Expected doc format:
    - 'question': The question text
    - 'choices': List of answer options (A, B, C, D)
    """
    question = doc.get("Question", "")
    choices = doc.get("Choices", [])

    # Build the question with choices
    string = f"{question}\n"

    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for i, choice in enumerate(choices[:10]):  # Limit to 5 choices (A-E)
        string += f"{labels[i]}. {choice}\n"

    return string


def process_answer(answer):
    """
    Process model answer to extract letter choice.

    This is a wrapper around the common extract_labels function from mcqa_utils.

    Handles multiple formats:
    - Simple letter: "A", "B"
    - Letter with period: "D.", "A."
    - Letter with full text: "D. Sea ice forms on ocean water..."
    - With prefix: "Answer: C. SIRAL"

    For single answer MCQA, only the first letter is used if multiple are extracted.
    """
    labels = extract_labels(answer)
    # Return first label only for single-answer MCQA
    return labels[0] if labels else ""


class filter_answer(Filter):
    """
    Filter class for lm-eval harness filter_list.

    Extracts the letter choice and returns it as a string
    for display in filtered_resps.
    """

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        """
        Apply the filter to extract answer label.

        Args:
            resps: List of lists of model responses
            docs: List of document dictionaries (not used here)

        Returns:
            List of lists of filtered responses (single letter)
        """

        def filter_set(inst):
            filtered = []
            for resp in inst:
                if not isinstance(resp, str):
                    resp = ""
                labels = extract_labels(resp)
                # For single answer, take first label only
                filtered_resp = labels[0] if labels else ""
                filtered.append(filtered_resp)
            return filtered

        filtered_resps = [filter_set(resp) for resp in resps]
        return filtered_resps


def process_results(doc: datasets.Dataset, results):
    """
    Process results for single-answer MCQA task.

    Args:
        doc: Document containing ground truth answer
        results: Model predictions (list of responses or filtered responses)

    Returns:
        dict: Metrics (accuracy)
    """
    # Get model prediction
    pred = results[0]

    # Handle different result formats
    if isinstance(pred, list):
        # If it's a list, take the first element (filtered response)
        pred_text = pred[0] if pred else ""
    else:
        # It's already a string
        pred_text = pred

    # Check if already filtered (single letter string) or needs processing (raw response)
    if isinstance(pred_text, str) and len(pred_text) <= 5 and pred_text.strip():
        # Check if it looks like a filtered answer (just letters, maybe with spaces/commas)
        cleaned = pred_text.strip().replace(" ", "").replace(",", "")
        if cleaned.isalpha() and len(cleaned) <= 3:
            # Already filtered to letter(s), take first letter for single answer
            pred_label = cleaned[0]
        else:
            # Raw response, needs processing
            pred_label = process_answer(pred_text)
    else:
        # Raw response, needs processing
        pred_label = process_answer(pred_text) if pred_text else ""

    # Get ground truth answer
    reference = doc.get("Answer", "")

    # Calculate accuracy
    acc = 1.0 if pred_label == reference else 0.0

    return {"acc": acc}
