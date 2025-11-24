#!/usr/bin/env python3
"""
vLLM Direct Library LoRA Checkpoint Benchmarking Script

This script uses vLLM as a direct Python library for efficient inference:
1. Loads models directly with vLLM LLM class
2. Uses del to properly clean up GPU memory between checkpoints
3. Simpler and more memory-efficient than server-based approach

Usage:
    python benchmark_vllm_server_v1.py --config benchmark_vllm_config.yaml
    python benchmark_vllm_server_v1.py --config benchmark_vllm_config.yaml --dry-run
"""

import os
import sys
import yaml
import json
import argparse
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import torch
import requests  # For GPT evaluation API calls

# Import vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError as e:
    VLLM_AVAILABLE = False
    print(f"Warning: vLLM not available: {e}")

# Note: We use vLLM's built-in tokenizer via llm.get_tokenizer()
# No need to import transformers separately

# Import local LitGPT utilities
try:
    from utils.merge_lora import merge_lora
    from utils.convert_lit_checkpoint import convert_lit_checkpoint
    LITGPT_AVAILABLE = True
except ImportError as e:
    LITGPT_AVAILABLE = False
    print(f"Warning: Local LitGPT utilities not available: {e}")


class VLLMServerBenchmarkMaster:
    def __init__(self, config_path: str):
        """Initialize the vLLM benchmark master with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        
        # Results storage
        self.all_results = []
        self.checkpoint_results = {}
        
        # vLLM model instance - will be recreated for each checkpoint
        self.llm = None
        self.current_model_path = None
        self.tokenizer = None
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = [
            'experiment_name', 'base_model_dir', 'checkpoints_dir',
            'benchmarks', 'vllm_config'
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        # Validate benchmark configurations
        if not isinstance(config['benchmarks'], list) or not config['benchmarks']:
            raise ValueError("benchmarks must be a non-empty list")
        
        for i, benchmark in enumerate(config['benchmarks']):
            if 'name' not in benchmark or 'dataset_path' not in benchmark:
                raise ValueError(f"Benchmark {i} missing required fields: name, dataset_path")
        
        return config
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO'))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup logger
        self.logger = logging.getLogger('VLLMServerBenchmarkMaster')
        self.logger.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if requested
        if self.config.get('logging', {}).get('save_logs', False):
            log_file = self.config.get('logging', {}).get('log_file', 'benchmark_vllm_server.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def setup_directories(self):
        """Create necessary output directories."""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'results').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'temp_merged').mkdir(exist_ok=True)
        (self.output_dir / 'temp_hf').mkdir(exist_ok=True)
        (self.output_dir / 'detailed_scores').mkdir(exist_ok=True)
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def discover_checkpoints(self) -> List[Path]:
        """Discover LoRA checkpoint directories (step-xxxxx format) and final checkpoint."""
        checkpoints_dir = Path(self.config['checkpoints_dir']).expanduser().resolve()
        
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
        
        self.logger.info(f"Searching for checkpoints in: {checkpoints_dir}")
        
        # Find all step-xxxxx directories and final directory that contain LoRA files
        checkpoint_dirs = []
        
        for subdir in checkpoints_dir.iterdir():
            if subdir.is_dir() and (subdir.name.startswith('step-') or subdir.name == 'final'):
                # Check if it contains LoRA files
                lora_file = subdir / "lit_model.pth.lora"
                hyperparams_file = subdir / "hyperparameters.yaml"
                
                if lora_file.exists() and hyperparams_file.exists():
                    checkpoint_dirs.append(subdir)
                    self.logger.info(f"  Found LoRA checkpoint: {subdir.name}")
        
        # Apply filtering based on config
        checkpoint_dirs = self.filter_checkpoints(checkpoint_dirs)
        
        # Sort checkpoints by step number, with final at the end
        def extract_step_number(path):
            if path.name == 'final':
                return float('inf')  # Put final at the end
            try:
                return int(path.name.split('-')[1])
            except (IndexError, ValueError):
                return 0
        
        checkpoint_dirs.sort(key=extract_step_number)
        
        self.logger.info(f"Discovered {len(checkpoint_dirs)} LoRA checkpoints")
        for cp in checkpoint_dirs:
            self.logger.info(f"  - {cp.name}")
        
        return checkpoint_dirs
    
    def filter_checkpoints(self, checkpoint_dirs: List[Path]) -> List[Path]:
        """Filter checkpoints based on configuration."""
        selection = self.config.get('checkpoints', {}).get('selection', 'all')
        
        if selection == 'all':
            filtered = checkpoint_dirs
        elif selection == 'latest':
            # Get the most recently modified checkpoint
            if checkpoint_dirs:
                latest = max(checkpoint_dirs, key=lambda x: x.stat().st_mtime)
                filtered = [latest]
            else:
                filtered = []
        elif selection == 'specific':
            # Filter specific checkpoints listed in config
            specific_names = self.config.get('checkpoints', {}).get('specific_list', [])
            filtered = [cp for cp in checkpoint_dirs if cp.name in specific_names]
        else:
            self.logger.warning(f"Unknown checkpoint selection: {selection}. Using all.")
            filtered = checkpoint_dirs
        
        return filtered
    
    def merge_lora_weights(self, checkpoint_dir: Path) -> Path:
        """Merge LoRA weights with base model."""
        self.logger.info(f"Merging LoRA weights for: {checkpoint_dir.name}")
        
        # Create temporary merged directory
        merged_dir = self.output_dir / 'temp_merged' / f"merged_{checkpoint_dir.name}_{int(time.time())}"
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy LoRA files from checkpoint
            lora_file = checkpoint_dir / "lit_model.pth.lora"
            hyperparams_file = checkpoint_dir / "hyperparameters.yaml"
            
            shutil.copy2(lora_file, merged_dir / "lit_model.pth.lora")
            shutil.copy2(hyperparams_file, merged_dir / "hyperparameters.yaml")
            
            # Copy necessary files from base model
            base_model_path = Path(self.config['base_model_dir']).expanduser().resolve()
            base_model_files = [
                'model_config.yaml',
                'tokenizer.json',
                'tokenizer_config.json', 
                'generation_config.json',
                'config.json'
            ]
            
            for file_name in base_model_files:
                src_file = base_model_path / file_name
                dst_file = merged_dir / file_name
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    self.logger.debug(f"Copied {file_name} from base model")
            
            # Perform LoRA merge
            self.logger.info(f"Starting LoRA merge operation...")
            merge_start_time = time.time()
            
            merge_lora(
                checkpoint_dir=merged_dir,
                pretrained_checkpoint_dir=base_model_path,
                precision=None
            )
            
            merge_time = time.time() - merge_start_time
            self.logger.info(f"LoRA merge completed in {merge_time:.1f}s")
            
            # Verify merged model exists
            if not (merged_dir / 'lit_model.pth').exists():
                raise RuntimeError(f"Merge failed - no lit_model.pth found")
            
            return merged_dir
            
        except Exception as e:
            # Cleanup on failure
            if merged_dir.exists():
                shutil.rmtree(merged_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to merge LoRA weights for {checkpoint_dir.name}: {e}")
    
    def convert_litgpt_to_hf(self, merged_litgpt_dir: Path) -> Path:
        """Convert LitGPT checkpoint to HuggingFace format."""
        self.logger.info(f"Converting LitGPT to HuggingFace format: {merged_litgpt_dir.name}")
        
        # Create temporary HF directory
        hf_dir = self.output_dir / 'temp_hf' / f"hf_{merged_litgpt_dir.name}_{int(time.time())}"
        hf_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Starting LitGPT to HuggingFace conversion...")
            conversion_start_time = time.time()
            
            # Use local convert_lit_checkpoint function directly
            convert_lit_checkpoint(checkpoint_dir=merged_litgpt_dir, output_dir=hf_dir)
            
            conversion_time = time.time() - conversion_start_time
            self.logger.info(f"LitGPT to HuggingFace conversion completed in {conversion_time:.1f}s")
            
            # The convert_lit_checkpoint function creates model.pth, but we need to rename/copy 
            # it to pytorch_model.bin for HuggingFace compatibility
            model_pth = hf_dir / "model.pth"
            pytorch_model_bin = hf_dir / "pytorch_model.bin"
            
            if model_pth.exists():
                shutil.move(str(model_pth), str(pytorch_model_bin))
                self.logger.info("Renamed model.pth to pytorch_model.bin for HF compatibility")
            
            # Copy ALL tokenizer and config files from merged directory to HF directory
            # This ensures we have all necessary files for the tokenizer
            files_to_copy = [
                'tokenizer.json',
                'tokenizer_config.json', 
                'generation_config.json',
                'config.json',
                'model_config.yaml',  # Include this too
                'special_tokens_map.json'  # If it exists
            ]
            
            for file_name in files_to_copy:
                src_file = merged_litgpt_dir / file_name
                dst_file = hf_dir / file_name
                
                if src_file.exists() and not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    self.logger.debug(f"Copied {file_name} to HF directory")
                elif not src_file.exists():
                    self.logger.debug(f"Source file {file_name} not found in {merged_litgpt_dir}")
            
            # Verify HF model files exist
            required_hf_files = ['pytorch_model.bin', 'config.json']
            missing_files = []
            
            for file_name in required_hf_files:
                if not (hf_dir / file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                raise RuntimeError(f"Conversion failed - missing required HF files: {missing_files}")
            
            self.logger.info("HuggingFace format checkpoint created successfully")
            return hf_dir
            
        except Exception as e:
            # Cleanup on failure
            if hf_dir.exists():
                shutil.rmtree(hf_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to convert to HF format: {e}")
    
    
    def clean_vram(self):
        """Clean VRAM using del and CUDA cache clearing."""
        self.logger.info("Cleaning VRAM with del...")
        
        # Delete vLLM model instance
        if self.llm is not None:
            del self.llm
            self.llm = None
            self.logger.info("Deleted vLLM model instance")
        
        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.info("Cleared CUDA cache")
        
        # Force garbage collection
        gc.collect()
        
        # Wait for memory to clear
        time.sleep(2)
        self.logger.info("✅ VRAM cleaned")
    
    def load_vllm_model(self, hf_model_dir: Path) -> bool:
        """Load vLLM model directly with automatic tokenizer."""
        # Clean VRAM first
        self.clean_vram()
        
        self.logger.info(f"Loading vLLM model: {hf_model_dir}")
        
        # Get vLLM configuration
        vllm_config = self.config.get('vllm_config', {})
        
        try:
            # Create vLLM instance - it will automatically load the tokenizer
            self.llm = LLM(
                model=str(hf_model_dir),
                dtype=vllm_config.get('dtype', 'auto'),
                max_model_len=vllm_config.get('max_model_len', 2048),
                gpu_memory_utilization=vllm_config.get('gpu_memory_utilization', 0.8),
                tensor_parallel_size=vllm_config.get('tensor_parallel_size', 1),
                trust_remote_code=vllm_config.get('trust_remote_code', True)
            )
            
            # Get tokenizer from vLLM after model is loaded
            try:
                self.tokenizer = self.llm.get_tokenizer()
                self.logger.info(f"✅ Tokenizer loaded automatically from vLLM")
                
                # Check if tokenizer has chat template
                if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                    self.logger.info("Tokenizer has chat template available")
                else:
                    self.logger.info("Tokenizer does not have chat template, will use raw prompts")
            except Exception as tokenizer_error:
                self.logger.warning(f"Could not access tokenizer from vLLM: {tokenizer_error}")
                self.tokenizer = None
            
            self.current_model_path = hf_model_dir
            self.logger.info(f"✅ Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load vLLM model: {e}")
            return False
    
    def unload_vllm_model(self):
        """Unload vLLM model and clean VRAM."""
        self.clean_vram()
    
    def generate_response_batch(self, prompts: List[str], sampling_config: Dict) -> List[str]:
        """Generate responses for a batch of prompts using vLLM."""
        if self.llm is None:
            raise RuntimeError("No vLLM model is loaded")
        
        # Get stop tokens and add common stop patterns for MCQA
        stop_tokens = sampling_config.get('stop_tokens', [])
        if not stop_tokens:
            # Default stop tokens for better MCQA responses
            stop_tokens = ['\n\n', '\n\nQuestion:', '\n\nAnswer:', '.  .', '(Note:', '(End']
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=sampling_config.get('temperature', 0.7),
            top_p=sampling_config.get('top_p', 0.9),
            max_tokens=sampling_config.get('max_tokens', 512),
            stop=stop_tokens,
            skip_special_tokens=True,  # Skip special tokens in output
            repetition_penalty=sampling_config.get('repetition_penalty', 1.15),  # Penalize repetitions
            frequency_penalty=sampling_config.get('frequency_penalty', 0.0),  # Additional repetition control
            presence_penalty=sampling_config.get('presence_penalty', 0.0)  # Additional diversity control
        )
        
        # Generate responses
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract and clean text from outputs
        responses = []
        for output in outputs:
            if output.outputs:
                text = output.outputs[0].text.strip()
                
                # Remove thinking/reasoning blocks (for models like DeepSeek)
                # Handle both complete tags and orphaned closing tags
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
                
                # Remove text before orphaned closing tags (when opening tag is missing)
                text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'^.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'^.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'^.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
                
                # Remove orphaned opening tags to end of text (when closing tag is missing)
                text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<thinking>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<thought>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<reasoning>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
                
                # Additional cleanup for common patterns
                # Remove repetitive dots/periods
                text = re.sub(r'(\.\s*){3,}.*$', '', text)
                # Remove trailing notes/comments
                text = re.sub(r'\s*\(Note:.*$', '', text, flags=re.IGNORECASE)
                text = re.sub(r'\s*\(End.*$', '', text, flags=re.IGNORECASE)
                # Remove extra whitespace
                text = ' '.join(text.split())
                
                responses.append(text)
            else:
                responses.append("")
        
        return responses
    
    def load_benchmark_dataset(self, dataset_path: str, num_samples: Optional[int] = None) -> List[Dict]:
        """Load benchmark dataset from JSON/JSONL file."""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        data = []
        
        if dataset_path.suffix == '.jsonl':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse line in {dataset_path}: {e}")
        
        elif dataset_path.suffix == '.json':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, list):
                    data = loaded_data
                elif isinstance(loaded_data, dict):
                    # Check if this is a nested structure like {"question 0": {...}, "question 1": {...}}
                    if any(key.startswith('question ') for key in loaded_data.keys()):
                        # Extract individual questions from nested structure
                        for question_key, question_data in loaded_data.items():
                            if isinstance(question_data, dict):
                                data.append(question_data)
                    else:
                        # Single question in dict format
                        data = [loaded_data]
        
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
        
        # Limit number of samples if specified
        if num_samples and num_samples < len(data):
            data = data[:num_samples]
            self.logger.info(f"Limited dataset to {num_samples} samples")
        
        self.logger.info(f"Loaded {len(data)} samples from {dataset_path}")
        return data
    
    def evaluate_benchmark(self, benchmark_config: Dict, checkpoint_name: str) -> Dict[str, Any]:
        """Evaluate model on a specific benchmark."""
        benchmark_name = benchmark_config['name']
        dataset_path = benchmark_config['dataset_path']
        
        self.logger.info(f"Evaluating benchmark: {benchmark_name}")
        
        # Load dataset
        num_samples = benchmark_config.get('num_samples')
        dataset = self.load_benchmark_dataset(dataset_path, num_samples)
        
        if not dataset:
            raise ValueError(f"No data loaded from {dataset_path}")
        
        # Get prompt template and evaluation config
        prompt_template = benchmark_config.get('prompt_template', "{question}")
        system_prompt = benchmark_config.get('system_prompt', None)
        sampling_config = benchmark_config.get('sampling_config', {})
        evaluation_type = benchmark_config.get('evaluation_type', 'qa_similarity')
        
        # Generate responses
        responses = []
        individual_scores = []
        
        self.logger.info(f"Generating responses for {len(dataset)} samples...")
        
        # Prepare prompts
        prompts = []
        valid_items = []
        
        for i, item in enumerate(dataset):
            try:
                question = item.get('question', item.get('input', ''))
                format_args = item.copy()
                format_args['question'] = question
                
                # Handle MCQA datasets
                mcqa_keys = ['option 1', 'option 2', 'option 3', 'option 4', 'option 5']
                for key in mcqa_keys:
                    if key in format_args:
                        underscore_key = key.replace(' ', '_')
                        format_args[underscore_key] = format_args[key]
                
                # Add empty placeholders for missing options
                for option_num in range(1, 6):
                    key = f'option_{option_num}'
                    if key not in format_args:
                        format_args[key] = ''
                
                user_prompt = prompt_template.format(**format_args)
                
                # Use chat template if tokenizer is available
                if (self.tokenizer and 
                    hasattr(self.tokenizer, 'apply_chat_template') and 
                    hasattr(self.tokenizer, 'chat_template') and 
                    self.tokenizer.chat_template):
                    
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": user_prompt})
                    
                    try:
                        prompt = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
                        if system_prompt:
                            prompt = f"{system_prompt}\n\n{user_prompt}"
                        else:
                            prompt = user_prompt
                else:
                    # Fallback to simple concatenation
                    if system_prompt:
                        prompt = f"{system_prompt}\n\n{user_prompt}"
                    else:
                        prompt = user_prompt
                
                prompts.append(prompt)
                valid_items.append(item)
            except KeyError as e:
                self.logger.warning(f"Failed to format prompt for item {i}: {e}")
                continue
        
        # Generate responses in batches
        batch_size = self.config.get('vllm_config', {}).get('batch_size', 50)
        all_generated_texts = []
        
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            current_batch = prompts[batch_start:batch_end]
            
            self.logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} ({len(current_batch)} items)")
            
            batch_responses = self.generate_response_batch(current_batch, sampling_config)
            all_generated_texts.extend(batch_responses)
            
            # Periodic garbage collection
            if (batch_start // batch_size + 1) % 10 == 0:
                gc.collect()
        
        # Process results
        for i, (item, generated_text) in enumerate(zip(valid_items, all_generated_texts)):
            # Ensure all fields are never None
            question = item.get('question', item.get('input', '')) or ''
            expected_answer = item.get('expected_answer', item.get('output', '')) or ''
            generated_answer = generated_text or ''
            
            responses.append({
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': generated_answer,
                'item_index': i
            })
        
        self.logger.info(f"Completed {len(responses)} responses")
        
        # Evaluate responses
        if evaluation_type == 'qa_similarity':
            individual_scores = self.evaluate_qa_similarity(responses)
        elif evaluation_type == 'exact_match':
            individual_scores = self.evaluate_exact_match(responses)
        elif evaluation_type == 'mcqa_accuracy':
            individual_scores = self.evaluate_mcqa_accuracy(responses)
        elif evaluation_type == 'mcqa_advanced':
            individual_scores = self.evaluate_mcqa_advanced(responses)
        elif evaluation_type == 'gpt_rating':
            individual_scores = self.evaluate_with_gpt_rating(responses, benchmark_config)
        elif evaluation_type == 'custom':
            # Use custom evaluation function if provided
            eval_function = benchmark_config.get('eval_function')
            if eval_function:
                individual_scores = eval_function(responses)
            else:
                self.logger.warning("Custom evaluation requested but no eval_function provided")
                individual_scores = [0.0] * len(responses)
        else:
            self.logger.warning(f"Unknown evaluation type: {evaluation_type}")
            individual_scores = [0.0] * len(responses)
        
        # Calculate aggregate metrics
        if individual_scores:
            metrics = {
                'average_score': float(np.mean(individual_scores)),
                'std_score': float(np.std(individual_scores)),
                'min_score': float(np.min(individual_scores)),
                'max_score': float(np.max(individual_scores)),
                'num_samples': int(len(individual_scores))
            }
        else:
            metrics = {
                'average_score': 0.0,
                'std_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'num_samples': 0
            }
        
        # Save detailed results
        detailed_results = {
            'checkpoint_name': checkpoint_name,
            'benchmark_name': benchmark_name,
            'evaluation_time': datetime.now().isoformat(),
            'individual_scores': individual_scores,
            'responses': responses,  # Save all responses for debugging
            'metrics': metrics
        }
        
        self.save_detailed_benchmark_results(benchmark_name, checkpoint_name, detailed_results)
        
        return {
            'benchmark_name': benchmark_name,
            'metrics': metrics,
            'detailed_results': detailed_results
        }
    
    def evaluate_with_gpt_rating(self, responses: List[Dict], benchmark_config: Dict) -> List[float]:
        """Evaluate responses using ChatGPT API to rate from 1-10."""
        # Get OpenAI API configuration
        openai_config = self.config.get('openai_api', {})
        api_key = openai_config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            self.logger.error("OpenAI API key not found. Set 'openai_api.api_key' in config or OPENAI_API_KEY environment variable")
            return [0.0] * len(responses)
        
        api_url = openai_config.get('api_url', 'https://api.openai.com/v1/chat/completions')
        model = openai_config.get('model', 'gpt-4o-mini')
        max_workers = openai_config.get('max_workers', 3)  # Lower to respect rate limits
        
        self.logger.info(f"Using GPT evaluation with model: {model}")
        
        def evaluate_single_response(response_data):
            # Safely handle None values
            question = response_data.get('question') or ''
            expected = response_data.get('expected_answer') or ''
            generated = response_data.get('generated_answer') or ''
            
            prompt = f"""You are an expert evaluator in satellite communications. Please rate the quality of the generated answer. Use the expected answer as a reference, but remember it is not the only factor, evaluate the generated answer based on its own technical correctness, completeness, clarity, and relevance.
            
            Question: {question}
            
            Expected Answer: {expected}
            
            Generated Answer: {generated}
            
            Evaluation Criteria:
            1. Technical Accuracy – Are the equations, definitions, and concepts (e.g., link budget, C/N, EIRP, antenna gain, propagation losses) correct and consistent with standard SatCom theory?
            2. Completeness – Does the answer include all key components and reasoning steps expected in a full SatCom explanation (e.g., key parameters, assumptions, or relevant formulas)?
            3. Clarity and Structure – Is the answer well-organized, logically presented, and easy to follow? Are steps and results clearly explained?
            4. Relevance – Does the answer stay focused on the specific question and avoid unrelated information or unnecessary background theory?
            
            Instructions:
            - Rate the answer from 1 to 10 based on overall quality, considering all four criteria equally.
            - Use this scale:
              - 10: Excellent – technically flawless, comprehensive, and clearly presented.
              - 8–9: Very good – mostly accurate with only minor omissions or clarity issues.
              - 6–7: Fair – generally correct but missing key points or minor technical mistakes.
              - 4–5: Weak – noticeable conceptual or computational errors; incomplete explanation.
              - 2–3: Poor – largely incorrect, confusing, or off-topic.
              - 1: Useless – completely wrong or irrelevant to satellite communications.
            
            Respond with just the numeric score (1–10):"""

            xxx = f"""You are an expert evaluator in satellite communications. Please rate the quality of the generated answer. Evaluate the generated answer based on its own technical correctness, completeness, clarity, and relevance.
            
            Question: {question}
                        
            Generated Answer: {generated}
            
            Evaluation Criteria:
            1. Technical Accuracy – Are the equations, definitions, and concepts (e.g., link budget, C/N, EIRP, antenna gain, propagation losses) correct and consistent with standard SatCom theory?
            2. Completeness – Does the answer include all key components and reasoning steps expected in a full SatCom explanation (e.g., key parameters, assumptions, or relevant formulas)?
            3. Clarity and Structure – Is the answer well-organized, logically presented, and easy to follow? Are steps and results clearly explained?
            4. Relevance – Does the answer stay focused on the specific question and avoid unrelated information or unnecessary background theory?
            
            Instructions:
            - Rate the answer from 1 to 10 based on overall quality, considering all four criteria equally.
            - Use this scale:
              - 10: Excellent – technically flawless, comprehensive, and clearly presented.
              - 8–9: Very good – mostly accurate with only minor omissions or clarity issues.
              - 6–7: Fair – generally correct but missing key points or minor technical mistakes.
              - 4–5: Weak – noticeable conceptual or computational errors; incomplete explanation.
              - 2–3: Poor – largely incorrect, confusing, or off-topic.
              - 1: Useless – completely wrong or irrelevant to satellite communications.
            
            After reasoning about the grade, only respond with just the numeric score (1–10):"""

            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.1,
                'max_tokens': 10
            }
            
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content'].strip()
                        # Extract numeric score
                        score_match = re.search(r'\b(\d+(?:\.\d+)?)\b', content)
                        if score_match:
                            score = float(score_match.group(1))
                            # Normalize to 0-1 scale
                            return min(max(score / 10.0, 0.0), 1.0)
                        else:
                            self.logger.warning(f"Could not extract score from GPT response: {content}")
                            return 0.5  # Default middle score
                    else:
                        self.logger.error("No choices in GPT response")
                        return 0.5
                else:
                    self.logger.error(f"GPT API error {response.status_code}: {response.text}")
                    return 0.5
            except Exception as e:
                self.logger.error(f"GPT evaluation error: {e}")
                return 0.5
        
        # Process responses concurrently with rate limiting
        scores = []
        batch_size = 10  # Process in small batches to avoid rate limits
        
        for i in range(0, len(responses), batch_size):
            batch = responses[i:i + batch_size]
            self.logger.info(f"GPT evaluating batch {i//batch_size + 1}/{(len(responses)-1)//batch_size + 1}")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                batch_scores = list(executor.map(evaluate_single_response, batch))
                scores.extend(batch_scores)
            
            # Rate limiting - wait between batches
            if i + batch_size < len(responses):
                time.sleep(1)  # 1 second between batches
        
        return scores
    
    def evaluate_mcqa_advanced(self, responses: List[Dict]) -> List[float]:
        """Advanced MCQA evaluation with better option extraction."""
        scores = []
        
        for response in responses:
            # Safely handle None values
            expected = response.get('expected_answer') or ''
            generated = response.get('generated_answer') or ''
            
            expected = expected.lower().strip()
            generated = generated.lower().strip()
            
            if not expected or not generated:
                scores.append(0.0)
                continue
            
            # Extract expected option (e.g., "option 2: ..." -> "2")
            expected_option_num = None
            expected_match = re.search(r'option\s*(\d+)', expected)
            if expected_match:
                expected_option_num = expected_match.group(1)
            
            # Try multiple patterns to extract option from generated response
            score = 0.0
            
            if expected_option_num:
                patterns = [
                    rf'\b{expected_option_num}\b',  # Just the number
                    rf'\boption\s*{expected_option_num}\b',  # "option 2"
                    rf'\({expected_option_num}\)',  # "(2)"
                    rf'\[{expected_option_num}\]',  # "[2]"
                ]
                
                # Convert number to letter (1->A, 2->B, 3->C, 4->D, 5->E)
                if int(expected_option_num) <= 5:  # Ensure we don't go beyond E
                    option_letter = chr(ord('A') + int(expected_option_num) - 1).lower()
                    patterns.extend([
                        rf'\b{option_letter}\b',  # "c"
                        rf'\b{option_letter}\)',  # "c)"
                        rf'\({option_letter}\)',  # "(c)"
                    ])
                
                for pattern in patterns:
                    if re.search(pattern, generated):
                        score = 1.0
                        break
                
                # If no pattern matches, check if the generated text starts with the option
                if score == 0.0:
                    if generated.strip().startswith(expected_option_num):
                        score = 1.0
                    elif int(expected_option_num) <= 5:  # Only check letter if within valid range
                        option_letter = chr(ord('A') + int(expected_option_num) - 1).lower()
                        if generated.strip().startswith(option_letter):
                            score = 1.0
            
            scores.append(score)
        
        return scores
    
    def evaluate_mcqa_accuracy(self, responses: List[Dict]) -> List[float]:
        """Evaluate multiple choice question accuracy."""
        scores = []
        
        for response in responses:
            # Safely handle None values
            expected = response.get('expected_answer') or ''
            generated = response.get('generated_answer') or ''
            
            expected = expected.lower().strip()
            generated = generated.lower().strip()
            
            if not expected or not generated:
                scores.append(0.0)
                continue
            
            # Extract option from expected answer (e.g., "option 1: ..." -> "option 1")
            expected_option = ""
            if expected.startswith("option "):
                expected_option = expected.split(":")[0].strip()
            
            # Check if generated response contains the correct option
            score = 0.0
            if expected_option:
                # Check for exact option match
                if expected_option in generated:
                    score = 1.0
                # Check for option number only (e.g., "1", "A", etc.)
                elif expected_option.split()[-1] in generated:
                    score = 1.0
                # Check for option letter (convert "option 1" to "A", etc.)
                elif expected_option == "option 1" and ("a)" in generated or " a " in generated or generated.startswith("a")):
                    score = 1.0
                elif expected_option == "option 2" and ("b)" in generated or " b " in generated or generated.startswith("b")):
                    score = 1.0
                elif expected_option == "option 3" and ("c)" in generated or " c " in generated or generated.startswith("c")):
                    score = 1.0
                elif expected_option == "option 4" and ("d)" in generated or " d " in generated or generated.startswith("d")):
                    score = 1.0
                elif expected_option == "option 5" and ("e)" in generated or " e " in generated or generated.startswith("e")):
                    score = 1.0
            
            scores.append(score)
        
        return scores
    
    def evaluate_qa_similarity(self, responses: List[Dict]) -> List[float]:
        """Evaluate Q&A similarity using simple text matching."""
        scores = []
        
        for response in responses:
            # Safely handle None values
            expected = response.get('expected_answer') or ''
            generated = response.get('generated_answer') or ''
            
            expected = expected.lower().strip()
            generated = generated.lower().strip()
            
            if not expected or not generated:
                scores.append(0.0)
                continue
            
            # Simple similarity scoring
            if expected == generated:
                score = 1.0
            elif expected in generated or generated in expected:
                score = 0.7
            else:
                # Simple word overlap scoring
                expected_words = set(expected.split())
                generated_words = set(generated.split())
                
                if expected_words and generated_words:
                    overlap = len(expected_words.intersection(generated_words))
                    union = len(expected_words.union(generated_words))
                    score = overlap / union if union > 0 else 0.0
                else:
                    score = 0.0
            
            scores.append(score)
        
        return scores
    
    def evaluate_exact_match(self, responses: List[Dict]) -> List[float]:
        """Evaluate exact match scoring."""
        scores = []
        
        for response in responses:
            # Safely handle None values
            expected = response.get('expected_answer') or ''
            generated = response.get('generated_answer') or ''
            
            expected = expected.lower().strip()
            generated = generated.lower().strip()
            
            score = 1.0 if expected == generated else 0.0
            scores.append(score)
        
        return scores
    
    def save_detailed_benchmark_results(self, benchmark_name: str, checkpoint_name: str, detailed_results: Dict):
        """Save detailed benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config['experiment_name']}_{checkpoint_name}_{benchmark_name}_detailed_{timestamp}.json"
        filepath = self.output_dir / 'detailed_scores' / filename
        
        with open(filepath, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        self.logger.info(f"Detailed benchmark results saved: {filepath}")
    
    def prepare_base_model_for_evaluation(self) -> Path:
        """Prepare the base model for evaluation (no LoRA merge needed)."""
        self.logger.info("Preparing base model for evaluation")
        
        base_model_path = Path(self.config['base_model_dir']).expanduser().resolve()
        
        if not base_model_path.exists():
            raise FileNotFoundError(f"Base model directory not found: {base_model_path}")
        
        # For base model, we need to convert from LitGPT to HuggingFace format
        hf_dir = self.output_dir / 'temp_hf' / f"hf_base_model_{int(time.time())}"
        hf_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Converting base model to HuggingFace format...")
            conversion_start_time = time.time()
            
            # Use local convert_lit_checkpoint function directly
            convert_lit_checkpoint(checkpoint_dir=base_model_path, output_dir=hf_dir)
            
            conversion_time = time.time() - conversion_start_time
            self.logger.info(f"Base model conversion completed in {conversion_time:.1f}s")
            
            # The convert_lit_checkpoint function creates model.pth, but we need to rename/copy 
            # it to pytorch_model.bin for HuggingFace compatibility
            model_pth = hf_dir / "model.pth"
            pytorch_model_bin = hf_dir / "pytorch_model.bin"
            
            if model_pth.exists():
                shutil.move(str(model_pth), str(pytorch_model_bin))
                self.logger.info("Renamed model.pth to pytorch_model.bin for HF compatibility")
            
            # Copy essential config files from base model to HF directory
            config_files_to_copy = [
                'tokenizer.json',
                'tokenizer_config.json',
                'generation_config.json',
                'config.json'
            ]
            
            for file_name in config_files_to_copy:
                src_file = base_model_path / file_name
                dst_file = hf_dir / file_name
                
                if src_file.exists() and not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    self.logger.debug(f"Copied {file_name} to HF directory")
            
            # Verify HF model files exist
            required_hf_files = ['pytorch_model.bin']
            missing_files = []
            
            for file_name in required_hf_files:
                if not (hf_dir / file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                raise RuntimeError(f"Base model conversion failed - missing required HF files: {missing_files}")
            
            self.logger.info("Base model HuggingFace format created successfully")
            return hf_dir
            
        except Exception as e:
            # Cleanup on failure
            if hf_dir.exists():
                shutil.rmtree(hf_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to prepare base model: {e}")
    
    def evaluate_base_model(self) -> Dict[str, Any]:
        """Evaluate the base model on all benchmarks."""
        self.logger.info("Starting evaluation for base model")
        
        hf_dir = None
        
        try:
            # Step 1: Convert base model to HuggingFace format
            hf_dir = self.prepare_base_model_for_evaluation()
            
            # Step 2: Load vLLM model
            if not self.load_vllm_model(hf_dir):
                raise RuntimeError("Failed to load vLLM model for base model")
            
            # Step 3: Run benchmarks
            benchmark_results = []
            for benchmark_config in self.config['benchmarks']:
                try:
                    result = self.evaluate_benchmark(benchmark_config, "base_model")
                    benchmark_results.append(result)
                    self.logger.info(f"Completed benchmark {benchmark_config['name']} for base_model")
                except Exception as e:
                    self.logger.error(f"Failed benchmark {benchmark_config['name']} for base_model: {e}")
                    benchmark_results.append({
                        'benchmark_name': benchmark_config['name'],
                        'error': str(e),
                        'metrics': {}
                    })
            
            # Compile results BEFORE cleanup
            result = {
                'checkpoint_name': 'base_model',
                'checkpoint_path': str(self.config['base_model_dir']),
                'evaluation_time': datetime.now().isoformat(),
                'benchmark_results': benchmark_results,
                'num_benchmarks': len(benchmark_results)
            }
            
            self.logger.info(f"✅ Completed all benchmarks for base model")
            return result
            
        except Exception as e:
            error_result = {
                'checkpoint_name': 'base_model',
                'checkpoint_path': str(self.config['base_model_dir']),
                'evaluation_time': datetime.now().isoformat(),
                'error': str(e),
                'benchmark_results': []
            }
            
            self.logger.error(f"❌ Failed evaluation for base model: {e}")
            return error_result
        
        finally:
            # Cleanup happens ONCE in finally block
            self.unload_vllm_model()
            
            # Add a small delay to ensure all GPU resources are released
            time.sleep(1)
            
            # Delete HuggingFace checkpoint using safe method
            if self.config.get('cleanup', {}).get('delete_hf_checkpoints', True) and hf_dir:
                self.logger.info(f"Deleting base model HuggingFace checkpoint: {hf_dir}")
                self.safe_rmtree(hf_dir)
                
            self.logger.info(f"✅ Cleanup completed for base model")
    
    def safe_rmtree(self, path: Path, max_retries: int = 3):
        """Safely remove directory tree with retries for Windows file locking issues."""
        if not path or not path.exists():
            return
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Attempting to delete {path} (attempt {attempt + 1}/{max_retries})")
                shutil.rmtree(path, ignore_errors=False)
                self.logger.info(f"✅ Successfully deleted: {path}")
                return
            except PermissionError as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Permission error deleting {path}, retrying in 2s... ({e})")
                    time.sleep(2)
                else:
                    self.logger.warning(f"Failed to delete {path} after {max_retries} attempts, ignoring")
                    # Try with ignore_errors as last resort
                    try:
                        shutil.rmtree(path, ignore_errors=True)
                    except:
                        pass
            except Exception as e:
                self.logger.warning(f"Error deleting {path}: {e}")
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except:
                    pass
                return
    
    def cleanup_temp_files(self, merged_dir: Path = None, hf_dir: Path = None):
        """Clean up temporary files."""
        cleanup_config = self.config.get('cleanup', {})
        
        if cleanup_config.get('delete_merged_litgpt', True) and merged_dir:
            self.logger.info(f"Cleaning up merged LitGPT directory: {merged_dir}")
            self.safe_rmtree(merged_dir)
        
        if cleanup_config.get('delete_hf_checkpoints', True) and hf_dir:
            self.logger.info(f"Cleaning up HuggingFace directory: {hf_dir}")
            self.safe_rmtree(hf_dir)
    
    def evaluate_checkpoint(self, checkpoint_dir: Path) -> Dict[str, Any]:
        """Evaluate a single checkpoint on all benchmarks."""
        self.logger.info(f"Starting evaluation for checkpoint: {checkpoint_dir.name}")
        
        merged_dir = None
        hf_dir = None
        
        try:
            # Step 1: Merge LoRA weights
            merged_dir = self.merge_lora_weights(checkpoint_dir)
            
            # Step 2: Convert to HuggingFace format
            hf_dir = self.convert_litgpt_to_hf(merged_dir)
            
            # Step 3: Delete merged LitGPT checkpoint
            if self.config.get('cleanup', {}).get('delete_merged_litgpt', True):
                self.logger.info(f"Deleting merged LitGPT checkpoint: {merged_dir}")
                self.safe_rmtree(merged_dir)
                merged_dir = None
            
            # Step 4: Load vLLM model
            if not self.load_vllm_model(hf_dir):
                raise RuntimeError("Failed to load vLLM model")
            
            # Step 5: Run benchmarks
            benchmark_results = []
            for benchmark_config in self.config['benchmarks']:
                try:
                    result = self.evaluate_benchmark(benchmark_config, checkpoint_dir.name)
                    benchmark_results.append(result)
                    self.logger.info(f"Completed benchmark {benchmark_config['name']} for {checkpoint_dir.name}")
                except Exception as e:
                    self.logger.error(f"Failed benchmark {benchmark_config['name']} for {checkpoint_dir.name}: {e}")
                    benchmark_results.append({
                        'benchmark_name': benchmark_config['name'],
                        'error': str(e),
                        'metrics': {}
                    })
            
            # Compile results BEFORE cleanup
            result = {
                'checkpoint_name': checkpoint_dir.name,
                'checkpoint_path': str(checkpoint_dir),
                'evaluation_time': datetime.now().isoformat(),
                'benchmark_results': benchmark_results,
                'num_benchmarks': len(benchmark_results)
            }
            
            self.logger.info(f"✅ Completed all benchmarks for checkpoint: {checkpoint_dir.name}")
            return result
            
        except Exception as e:
            error_result = {
                'checkpoint_name': checkpoint_dir.name,
                'checkpoint_path': str(checkpoint_dir),
                'evaluation_time': datetime.now().isoformat(),
                'error': str(e),
                'benchmark_results': []
            }
            
            self.logger.error(f"❌ Failed evaluation for checkpoint {checkpoint_dir.name}: {e}")
            return error_result
        
        finally:
            # Cleanup happens ONCE in finally block (runs on both success and error)
            self.unload_vllm_model()
            
            # Add a small delay to ensure all GPU resources are released
            time.sleep(1)
            
            # Use safe cleanup method (handles both merged_dir and hf_dir)
            self.cleanup_temp_files(merged_dir, hf_dir)
            
            self.logger.info(f"✅ Cleanup completed for checkpoint: {checkpoint_dir.name}")

    
    def final_cleanup(self):
        """Final cleanup to be called at the end of benchmarking."""
        self.logger.info("Performing final cleanup...")
        self.clean_vram()
        self.logger.info("✅ Final cleanup complete")

    
    def save_results(self):
        """Save evaluation results to files."""
        if not self.all_results:
            self.logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.config['experiment_name']
        
        # Save summary results
        summary_file = self.output_dir / 'results' / f"{experiment_name}_vllm_server_summary_{timestamp}.json"
        summary = {
            'experiment_name': experiment_name,
            'config': self.config,
            'evaluation_time': datetime.now().isoformat(),
            'num_checkpoints': len(self.all_results),
            'results': self.all_results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save as CSV
        self.save_results_csv(summary_file.with_suffix('.csv'))
        
        self.logger.info(f"Results saved to {self.output_dir / 'results'}")
    
    def save_results_csv(self, csv_path: Path):
        """Save results as CSV for easy analysis."""
        if not self.all_results:
            return
        
        csv_data = []
        for result in self.all_results:
            base_row = {
                'checkpoint_name': result['checkpoint_name'],
                'evaluation_time': result['evaluation_time'],
                'has_error': 'error' in result
            }
            
            if 'error' in result:
                base_row['error'] = result['error']
                csv_data.append(base_row)
            else:
                # Add benchmark results
                for benchmark_result in result.get('benchmark_results', []):
                    row = base_row.copy()
                    row['benchmark_name'] = benchmark_result.get('benchmark_name', '')
                    
                    if 'metrics' in benchmark_result:
                        for metric_name, metric_value in benchmark_result['metrics'].items():
                            row[f'metric_{metric_name}'] = metric_value
                    
                    if 'error' in benchmark_result:
                        row['benchmark_error'] = benchmark_result['error']
                    
                    csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            self.logger.info(f"CSV results saved to {csv_path}")
    
    def print_summary(self):
        """Print a summary of evaluation results."""
        if not self.all_results:
            self.logger.info("No results to summarize")
            return
        
        print("\n" + "="*80)
        print(f"vLLM SERVER BENCHMARK RESULTS SUMMARY - {self.config['experiment_name']}")
        print("="*80)
        
        successful_results = [r for r in self.all_results if 'error' not in r]
        failed_results = [r for r in self.all_results if 'error' in r]
        
        print(f"Total checkpoints evaluated: {len(self.all_results)}")
        print(f"Successful evaluations: {len(successful_results)}")
        print(f"Failed evaluations: {len(failed_results)}")
        
        if failed_results:
            print(f"\nFailed checkpoints:")
            for result in failed_results:
                print(f"  - {result['checkpoint_name']}: {result.get('error', 'Unknown error')}")
        
        if successful_results:
            print(f"\nBenchmark Results Summary:")
            
            # Collect all benchmark results
            all_benchmark_results = {}
            for result in successful_results:
                checkpoint_name = result['checkpoint_name']
                for benchmark_result in result.get('benchmark_results', []):
                    benchmark_name = benchmark_result.get('benchmark_name', '')
                    if benchmark_name not in all_benchmark_results:
                        all_benchmark_results[benchmark_name] = []
                    
                    if 'metrics' in benchmark_result:
                        all_benchmark_results[benchmark_name].append({
                            'checkpoint': checkpoint_name,
                            'metrics': benchmark_result['metrics']
                        })
            
            # Print results for each benchmark
            for benchmark_name, benchmark_data in all_benchmark_results.items():
                print(f"\n📊 {benchmark_name}:")
                
                if not benchmark_data:
                    print("  No successful results")
                    continue
                
                # Get common metrics
                all_metrics = set()
                for data in benchmark_data:
                    all_metrics.update(data['metrics'].keys())
                
                for metric in sorted(all_metrics):
                    print(f"  {metric.replace('_', ' ').title()}:")
                    
                    metric_results = []
                    for data in benchmark_data:
                        if metric in data['metrics']:
                            metric_results.append((data['checkpoint'], data['metrics'][metric]))
                    
                    # Sort by metric value
                    metric_results.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (checkpoint, value) in enumerate(metric_results[:3]):  # Top 3
                        marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                        print(f"    {marker} {checkpoint}: {value:.4f}")
        
        print("\n" + "="*80)
    
    def run_benchmark(self, dry_run: bool = False):
        """Run the complete vLLM benchmark process."""
        self.logger.info(f"Starting vLLM benchmark: {self.config['experiment_name']}")
        
        if dry_run:
            self.logger.info("DRY RUN MODE - No actual evaluation will be performed")
        
        # Check dependencies
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not available. Install with: pip install vllm")
        
        if not LITGPT_AVAILABLE:
            raise RuntimeError("Local LitGPT utilities are not available. Ensure merge_lora.py and convert_lit_checkpoint.py are in the same directory.")
        
        # Discover checkpoints
        checkpoint_dirs = self.discover_checkpoints()
        
        if not checkpoint_dirs:
            self.logger.error("No valid LoRA checkpoints found!")
            return
        
        if dry_run:
            self.logger.info(f"Would evaluate {len(checkpoint_dirs)} checkpoints:")
            for cp in checkpoint_dirs:
                self.logger.info(f"  - {cp.name}")
            self.logger.info(f"Would run {len(self.config['benchmarks'])} benchmarks:")
            for benchmark in self.config['benchmarks']:
                self.logger.info(f"  - {benchmark['name']}")
            return
        
        # Evaluate base model first (for comparison baseline)
        # Check if base_model is in the specific_list (if selection is 'specific')
        should_evaluate_base = self.config.get('evaluate_base_model', True)
        
        checkpoints_config = self.config.get('checkpoints', {})
        if checkpoints_config.get('selection') == 'specific':
            specific_list = checkpoints_config.get('specific_list', [])
            # Only evaluate base model if it's explicitly in the list OR if evaluate_base_model is True and list is empty
            if specific_list:
                should_evaluate_base = 'base_model' in specific_list
            
        if should_evaluate_base:
            self.logger.info("Evaluating base model for baseline comparison")
            try:
                base_result = self.evaluate_base_model()
                self.all_results.append(base_result)
                self.checkpoint_results['base_model'] = base_result
            except Exception as e:
                self.logger.error(f"Failed to evaluate base model: {e}")
                
                error_result = {
                    'checkpoint_name': 'base_model',
                    'checkpoint_path': str(self.config['base_model_dir']),
                    'evaluation_time': datetime.now().isoformat(),
                    'error': str(e),
                    'benchmark_results': []
                }
                self.all_results.append(error_result)
        else:
            self.logger.info("Skipping base model evaluation (not in checkpoint selection)")
        
        # Evaluate each LoRA checkpoint
        for i, checkpoint_dir in enumerate(checkpoint_dirs, 1):
            self.logger.info(f"Processing checkpoint {i}/{len(checkpoint_dirs)}: {checkpoint_dir.name}")
            
            try:
                result = self.evaluate_checkpoint(checkpoint_dir)
                self.all_results.append(result)
                self.checkpoint_results[checkpoint_dir.name] = result
                
            except Exception as e:
                self.logger.error(f"Failed to process checkpoint {checkpoint_dir.name}: {e}")
                
                error_result = {
                    'checkpoint_name': checkpoint_dir.name,
                    'checkpoint_path': str(checkpoint_dir),
                    'evaluation_time': datetime.now().isoformat(),
                    'error': str(e),
                    'benchmark_results': []
                }
                self.all_results.append(error_result)
        
        # Save results and print summary
        self.save_results()
        self.print_summary()
        self.final_cleanup()
        self.logger.info("vLLM server benchmark completed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="vLLM Server-based LoRA Checkpoint Benchmarking Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_vllm_server_v1.py --config benchmark_vllm_config.yaml
  python benchmark_vllm_server_v1.py --config benchmark_vllm_config.yaml --dry-run
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without actual evaluation'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize and run benchmark
        benchmark = VLLMServerBenchmarkMaster(args.config)
        benchmark.run_benchmark(dry_run=args.dry_run)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()