"""
Utility functions for LitGPT checkpoint processing.

This module provides utilities for:
- Merging LoRA weights with base models
- Converting LitGPT checkpoints to HuggingFace format
"""

from .merge_lora import merge_lora, load_lora_metadata
from .convert_lit_checkpoint import convert_lit_checkpoint

__all__ = [
    'merge_lora',
    'load_lora_metadata',
    'convert_lit_checkpoint'
]

