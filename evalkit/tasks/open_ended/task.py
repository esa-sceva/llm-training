"""
Custom task class for open-ended evaluation with multi-judge support.
"""

import json
import os
from lm_eval.api.task import ConfigurableTask
from metrics.judge_utils import (
    process_qa_results,
    aggregate_llm_judge,
)


class OpenEndedTask(ConfigurableTask):
    """
    Custom task for open-ended evaluation that supports multi-judge configuration.
    """

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None):
        """Initialize task, injecting judges from environment variable if present."""
        # Check for judges in environment variable (set by evaluate.py)
        if config is None:
            config = {}

        # Extract judges before passing to parent (lm_eval's TaskConfig doesn't support it)
        judges = config.get('judges', None)

        # If not in config, try environment variable
        if judges is None:
            task_judges_env = os.environ.get("TASK_JUDGES")
            if task_judges_env:
                try:
                    judges = json.loads(task_judges_env)
                    print(f"[DEBUG] Loaded {len(judges)} judges from environment variable")
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Failed to parse TASK_JUDGES environment variable: {e}")

        # Remove fields that lm_eval's TaskConfig doesn't support
        config_for_parent = {k: v for k, v in config.items() if k not in ('class', 'judges')}

        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            config=config_for_parent
        )

        # Store judges in the config after parent initialization
        if judges:
            self._config.judges = judges

    def process_results(self, doc, results):
        """
        Process results with access to task configuration including judges.

        Args:
            doc: The document containing question and answer
            results: Model outputs

        Returns:
            Dictionary with judge evaluation results
        """
        # Get judges configuration from task config (stored as attribute)
        judges = getattr(self.config, 'judges', None)

        # Debug: Print judges config on first call
        if not hasattr(self, '_debug_printed'):
            config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
            print(f"\n[DEBUG] Task config keys: {list(config_dict.keys())}")
            print(f"[DEBUG] Judges config: {judges}")
            if judges:
                for j in judges:
                    print(f"[DEBUG]   Judge: {j.get('name')} - has api_key: {bool(j.get('api_key'))}")
            self._debug_printed = True

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

        # Use the centralized processing function with judges config
        return process_qa_results(
            doc=doc,
            results=results,
            question_key=question_key,
            answer_key=answer_key,
            sleep_time=0.0,
            judges=judges,
        )

    def aggregation(self):
        """
        Define aggregation functions for metrics.
        """
        judges = getattr(self.config, 'judges', None)

        if judges and len(judges) > 0:
            # Multi-judge mode: create aggregations for each judge
            result = {
                "llm_as_judge_avg": aggregate_llm_judge,
            }

            # Add aggregation for each individual judge
            for judge_config in judges:
                judge_name = judge_config.get("name", judge_config.get("model", "unknown"))
                result[f"llm_as_judge_{judge_name}"] = aggregate_llm_judge

            return result
        else:
            # Single judge mode
            return {
                "llm_as_judge": aggregate_llm_judge,
            }

    def higher_is_better(self):
        """
        Define whether higher values are better for each metric.
        """
        judges = getattr(self.config, 'judges', None)

        if judges and len(judges) > 0:
            # Multi-judge mode
            result = {
                "llm_as_judge_avg": True,
            }

            # Add for each individual judge
            for judge_config in judges:
                judge_name = judge_config.get("name", judge_config.get("model", "unknown"))
                result[f"llm_as_judge_{judge_name}"] = True

            return result
        else:
            # Single judge mode
            return {
                "llm_as_judge": True,
            }