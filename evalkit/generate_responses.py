#!/usr/bin/env python3
"""
Generate model responses for win-rate evaluation.

Queries a locally served model (vLLM / OpenAI-compatible) on a HuggingFace
dataset and writes a CSV compatible with the win_rate_evaluation.py script.

Usage:
    python generate_responses.py --config generate_config.yaml

Example config (generate_config.yaml):
    model:
      name: merged-output
      base_url: http://localhost:8010/v1
      api_key: EMPTY
      temperature: 0.2
      max_tokens: 1000

    dataset:
      path: esa-sceva/satcom-qa
      split: train
      question_key: Question
      answer_key: Answer

    prompt:
      system: |
        You are a Satellite Communication expert. Please answer the following
        question with technical accuracy. Provide a clear, detailed answer.
      user_template: "Q: {question}"
      assistant_prefix: "A: "

    output:
      file: responses/merged-output_satcom-qa.csv

    max_concurrent: 20
    limit: null          # set to int to evaluate a subset
"""

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def query_model(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: Optional[str],
    user_text: str,
    assistant_prefix: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    if assistant_prefix:
        messages.append({"role": "assistant", "content": assistant_prefix})

    extra = {}
    if assistant_prefix:
        extra["extra_body"] = {
            "add_generation_prompt": False,
            "continue_final_message": True,
        }

    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            return f"[ERROR] {e}"


async def run(cfg: dict):
    model_cfg = cfg["model"]
    ds_cfg = cfg["dataset"]
    prompt_cfg = cfg.get("prompt", {})
    output_cfg = cfg["output"]

    client = AsyncOpenAI(
        base_url=model_cfg["base_url"],
        api_key=model_cfg.get("api_key", "EMPTY"),
    )

    print(f"Loading dataset: {ds_cfg['path']} (split={ds_cfg['split']})")
    ds = load_dataset(ds_cfg["path"], split=ds_cfg["split"])

    limit = cfg.get("limit")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"  Limited to {len(ds)} samples")
    else:
        print(f"  {len(ds)} samples")

    q_key = ds_cfg.get("question_key", "Question")
    a_key = ds_cfg.get("answer_key", "Answer")
    system_prompt = prompt_cfg.get("system", "").strip() or None
    user_tpl = prompt_cfg.get("user_template", "Q: {question}")
    assistant_prefix = prompt_cfg.get("assistant_prefix", "")
    temperature = model_cfg.get("temperature", 0.2)
    max_tokens = model_cfg.get("max_tokens", 1000)
    max_concurrent = cfg.get("max_concurrent", 20)

    semaphore = asyncio.Semaphore(max_concurrent)

    print(f"\nGenerating responses with {model_cfg['name']}...")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Concurrency: {max_concurrent}")
    if system_prompt:
        print(f"  System prompt: {system_prompt[:80]}...")
    print()

    tasks = []
    for row in ds:
        question = row.get(q_key, "")
        user_text = user_tpl.format(question=question)
        tasks.append(
            query_model(
                client, model_cfg["name"], system_prompt, user_text,
                assistant_prefix, temperature, max_tokens, semaphore,
            )
        )

    responses = await tqdm_asyncio.gather(*tasks, desc="Requesting API")

    out_path = Path(output_cfg["file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Question", "target", "filtered_resps"])
        writer.writeheader()
        for row, resp in zip(ds, responses):
            writer.writerow({
                "Question": row.get(q_key, ""),
                "target": row.get(a_key, ""),
                "filtered_resps": resp.strip(),
            })

    print(f"\nWrote {len(responses)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate model responses for win-rate evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
