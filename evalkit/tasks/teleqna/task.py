"""
Custom MCQA task class that auto-detects dataset format.

Supports two schemas:
  1. Option columns: Questions, Option1-Option5, Answer="Option N"
  2. Choices list:   question, choices=[], answer=int (0-based index)
"""

import re
from lm_eval.api.task import ConfigurableTask

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def _get_question(doc):
    """Extract question text from either format."""
    return doc.get("question") or doc.get("Questions") or doc.get("Question") or ""


def _get_choices(doc):
    """Extract choices list from either format."""
    # Choices as a list (handles both "choices" and "Choices")
    for key in ("choices", "Choices"):
        if key in doc and isinstance(doc[key], list):
            return [str(c).strip() for c in doc[key] if c is not None]

    # Option1-Option5 columns
    options = []
    for i in range(1, 6):
        val = doc.get(f"Option{i}")
        if val is not None and str(val).strip():
            options.append(str(val).strip())
    return options


def _answer_to_letter(doc):
    """
    Convert answer field to a letter label, auto-detecting format.

    Format 1: "Option 2" / "Option2" -> "B"
    Format 2: integer index (0-based) -> "C" for 2
    Also handles: bare letter "B", bare number "2" (1-based for Option format)
    """
    answer = doc.get("answer") if "answer" in doc else doc.get("Answer", "")

    if answer is None:
        return ""

    # Integer index (0-based) - format 2 (netop/TeleQnA style)
    if isinstance(answer, int):
        if 0 <= answer < len(LABELS):
            return LABELS[answer]
        return ""

    answer_str = str(answer).strip()
    if not answer_str:
        return ""

    # "Option N" format (1-based)
    option_match = re.match(r"(?i)option\s*(\d+)", answer_str)
    if option_match:
        idx = int(option_match.group(1))
        if 1 <= idx <= len(LABELS):
            return LABELS[idx - 1]

    # Bare number -- could be 0-based or 1-based; check context
    # If "choices" key exists, dataset uses 0-based indexing
    if answer_str.isdigit():
        idx = int(answer_str)
        if "choices" in doc:
            if 0 <= idx < len(LABELS):
                return LABELS[idx]
        else:
            if 1 <= idx <= len(LABELS):
                return LABELS[idx - 1]

    # Already a single letter
    if len(answer_str) == 1 and answer_str.upper() in LABELS:
        return answer_str.upper()

    return ""


class McqaOptionsTask(ConfigurableTask):
    """MCQA task that auto-detects dataset format."""

    INSTRUCTION = (
        "You are an Satellite Communication expert. After reading carefully the following "
        "multiple choice question about SatCom, select the correct answer. Strictly respond "
        'in the format: letter only (e.g., "A", "B", "C", "D", "E").'
    )

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None):
        if config is None:
            config = {}
        config_for_parent = {k: v for k, v in config.items() if k not in ('class',)}
        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            config=config_for_parent,
        )

    def doc_to_text(self, doc):
        question = _get_question(doc)
        choices = _get_choices(doc)

        text = f"{self.INSTRUCTION}\n\n{question}\n"
        for i, opt in enumerate(choices):
            text += f"{LABELS[i]}. {opt}\n"

        import random
        if random.random() < 0.0001:
            print(f"\n[DEBUG] === MCQA prompt sample ===\n{text}[DEBUG] === End prompt ===\n")

        return text

    def doc_to_target(self, doc):
        return _answer_to_letter(doc)
