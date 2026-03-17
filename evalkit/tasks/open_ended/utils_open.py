# Import centralized LLM judge utilities
from metrics.judge_utils import (
    LoggableFuture,
    process_qa_results,
    aggregate_llm_judge,
    create_judge_aggregator,
    calculate_judge_agreement,
)
import os


def process_results(doc: dict, results: list[str]) -> dict:
    """
    Process results for open-ended questions.
    Auto-detects field names (question/answer vs Question/Answer).
    Supports multi-judge evaluation via judges configuration in doc.
    """
    # Auto-detect field names
    if "Question" in doc:
        question_key = "Question"
        answer_key = "Answer"
    elif "question" in doc:
        question_key = "question"
        answer_key = "answer"
    else:
        # Fallback: try to find any key containing 'question'
        question_keys = [k for k in doc.keys() if "question" in k.lower()]
        answer_keys = [k for k in doc.keys() if "answer" in k.lower()]
        question_key = question_keys[0] if question_keys else "question"
        answer_key = answer_keys[0] if answer_keys else "answer"

    # Check if judges configuration is available in doc
    judges = doc.get("judges")

    return process_qa_results(
        doc=doc,
        results=results,
        question_key=question_key,
        answer_key=answer_key,
        sleep_time=0.0,
        judges=judges,
    )


# Export the aggregation functions
aggregate_llm_judge = aggregate_llm_judge
calculate_judge_agreement = calculate_judge_agreement
