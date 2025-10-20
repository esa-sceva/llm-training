#!/usr/bin/env python3
"""
Compare Two Checkpoint Results Using GPT Evaluation

This script compares answers from two different checkpoints by asking GPT
which answer is better for each question. It generates a comprehensive
report with visualizations.

Usage:
    python compare_checkpoints_gpt.py --file1 result1.json --file2 result2.json --api-key YOUR_KEY
    python compare_checkpoints_gpt.py --file1 result1.json --file2 result2.json --config test_vllm_config.yaml
"""

import os
import sys
import json
import argparse
import logging
import time
import yaml
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import re

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. CSV export will be skipped.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Statistics will be limited.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)


class CheckpointComparator:
    """Compare two checkpoint results using GPT evaluation."""
    
    def __init__(self, api_key: str, model: str = 'gpt-4o-mini', max_workers: int = 5):
        """Initialize the comparator."""
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.api_url = 'https://api.openai.com/v1/chat/completions'
        
        # Setup logging
        self.setup_logging()
        
        # Results storage
        self.comparison_results = []
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CheckpointComparator')
    
    def load_result_file(self, filepath: str) -> Dict:
        """Load a benchmark result JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded {filepath}")
        self.logger.info(f"  Checkpoint: {data.get('checkpoint_name', 'Unknown')}")
        self.logger.info(f"  Benchmark: {data.get('benchmark_name', 'Unknown')}")
        self.logger.info(f"  Responses: {len(data.get('responses', []))}")
        
        return data
    
    def match_responses(self, data1: Dict, data2: Dict) -> List[Tuple[Dict, Dict]]:
        """Match responses between two result files by question."""
        responses1 = data1.get('responses', [])
        responses2 = data2.get('responses', [])
        
        # Create a mapping of questions to responses for file2
        question_map2 = {r['question']: r for r in responses2}
        
        # Match responses by question
        matched_pairs = []
        unmatched_count = 0
        
        for r1 in responses1:
            question = r1['question']
            if question in question_map2:
                matched_pairs.append((r1, question_map2[question]))
            else:
                unmatched_count += 1
        
        self.logger.info(f"Matched {len(matched_pairs)} question pairs")
        if unmatched_count > 0:
            self.logger.warning(f"Could not match {unmatched_count} questions")
        
        return matched_pairs
    
    def compare_single_pair(self, pair_data: Tuple[Dict, Dict, int, str, str]) -> Dict:
        """Use GPT to compare two answers and determine which is better."""
        response1, response2, index, checkpoint1_name, checkpoint2_name = pair_data
        
        question = response1['question']
        expected = response1.get('expected_answer', '')
        answer1 = response1.get('generated_answer', '')
        answer2 = response2.get('generated_answer', '')
        
        # Randomly swap the order to reduce position bias
        swapped = random.choice([True, False])
        
        if swapped:
            # Show answer2 first (as A), answer1 second (as B)
            display_answer_A = answer2
            display_answer_B = answer1
            display_name_A = checkpoint2_name
            display_name_B = checkpoint1_name
        else:
            # Show answer1 first (as A), answer2 second (as B)
            display_answer_A = answer1
            display_answer_B = answer2
            display_name_A = checkpoint1_name
            display_name_B = checkpoint2_name
        
        prompt = f"""You are an expert evaluator comparing two AI-generated answers to the same question.

Question: {question}

Expected/Reference Answer: {expected}

Answer A:
{display_answer_A}

Answer B:
{display_answer_B}

Instructions:
- Compare both answers based on: accuracy, completeness, relevance, and clarity
- Consider how well each answer aligns with the expected answer
- Determine which answer is better overall
- Do not make assumptions about which answer should be better based on any external factors

Respond in JSON format with:
{{
  "winner": "A" or "B" or "tie",
  "confidence": "high", "medium", or "low",
  "reasoning": "<brief explanation>"
}}

Your response:"""

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.1,
            'max_tokens': 300,
            'response_format': {'type': 'json_object'}
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content'].strip()
                    
                    # Parse JSON response
                    try:
                        comparison = json.loads(content)
                        
                        # Get GPT's winner and scores
                        gpt_winner = comparison.get('winner', 'tie')
                        score_A = comparison.get('score_A', 5)
                        score_B = comparison.get('score_B', 5)
                        
                        # Unswap the winner to match original file order (file1 vs file2)
                        if swapped:
                            # We showed answer2 as A and answer1 as B, so swap back
                            if gpt_winner == 'A':
                                actual_winner = 'B'  # A was checkpoint2, so winner is B (file2)
                            elif gpt_winner == 'B':
                                actual_winner = 'A'  # B was checkpoint1, so winner is A (file1)
                            else:
                                actual_winner = 'tie'
                            # Swap scores back
                            actual_score_A = score_B  # File1's score was shown as B
                            actual_score_B = score_A  # File2's score was shown as A
                        else:
                            # No swap, winner stays as-is
                            actual_winner = gpt_winner
                            actual_score_A = score_A
                            actual_score_B = score_B
                        
                        return {
                            'question_index': index,
                            'question': question,
                            'winner': actual_winner,
                            'confidence': comparison.get('confidence', 'medium'),
                            'score_A': actual_score_A,  # Score for checkpoint1/file1
                            'score_B': actual_score_B,  # Score for checkpoint2/file2
                            'reasoning': comparison.get('reasoning', ''),
                            'answer_A': answer1[:200] + '...' if len(answer1) > 200 else answer1,
                            'answer_B': answer2[:200] + '...' if len(answer2) > 200 else answer2,
                            'expected_answer': expected[:200] + '...' if len(expected) > 200 else expected,
                            'status': 'success',
                            'position_swapped': swapped  # Track if we swapped for transparency
                        }
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse GPT JSON response for question {index}")
                        return self.create_error_result(index, question, 'json_parse_error')
                else:
                    self.logger.error(f"No choices in GPT response for question {index}")
                    return self.create_error_result(index, question, 'no_choices')
            else:
                self.logger.error(f"GPT API error {response.status_code} for question {index}")
                return self.create_error_result(index, question, f'api_error_{response.status_code}')
        except Exception as e:
            self.logger.error(f"Error comparing question {index}: {e}")
            return self.create_error_result(index, question, str(e))
    
    def create_error_result(self, index: int, question: str, error: str) -> Dict:
        """Create an error result for failed comparisons."""
        return {
            'question_index': index,
            'question': question,
            'winner': 'error',
            'confidence': 'low',
            'score_A': 5,
            'score_B': 5,
            'reasoning': f'Error: {error}',
            'answer_A': '',
            'answer_B': '',
            'expected_answer': '',
            'status': 'error'
        }
    
    def compare_responses_batch(
        self, 
        matched_pairs: List[Tuple[Dict, Dict]], 
        checkpoint1_name: str,
        checkpoint2_name: str
    ) -> List[Dict]:
        """Compare all matched response pairs using batch processing."""
        self.logger.info(f"Comparing {len(matched_pairs)} question pairs...")
        print(f"\n🔍 Starting comparison: {checkpoint1_name} vs {checkpoint2_name}")
        print(f"📝 Total questions to evaluate: {len(matched_pairs)}")
        print(f"⚙️  Processing with {self.max_workers} concurrent workers\n")
        
        # Prepare data with indices for batch processing
        comparison_data = [
            (pair[0], pair[1], i, checkpoint1_name, checkpoint2_name)
            for i, pair in enumerate(matched_pairs)
        ]
        
        results = []
        batch_size = 10  # Process in batches to respect rate limits
        
        for batch_start in range(0, len(comparison_data), batch_size):
            batch_end = min(batch_start + batch_size, len(comparison_data))
            current_batch = comparison_data[batch_start:batch_end]
            
            self.logger.info(
                f"Processing batch {batch_start//batch_size + 1}/"
                f"{(len(comparison_data)-1)//batch_size + 1} "
                f"({len(current_batch)} comparisons)"
            )
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batch_results = list(executor.map(self.compare_single_pair, current_batch))
                results.extend(batch_results)
            
            # Print progress summary after each batch
            current_wins_A = sum(1 for r in results if r['winner'] == 'A')
            current_wins_B = sum(1 for r in results if r['winner'] == 'B')
            current_ties = sum(1 for r in results if r['winner'] == 'tie')
            print(f"  Progress: {checkpoint1_name}={current_wins_A} | {checkpoint2_name}={current_wins_B} | Ties={current_ties} | Total={len(results)}/{len(comparison_data)}")
            
            # Rate limiting
            if batch_end < len(comparison_data):
                time.sleep(1)
        
        # Print final summary
        print(f"\n✅ Comparison complete!")
        print(f"Final Results: {checkpoint1_name}={current_wins_A} wins | {checkpoint2_name}={current_wins_B} wins | Ties={current_ties}\n")
        
        return results
    
    def analyze_results(self, results: List[Dict], checkpoint1_name: str, checkpoint2_name: str) -> Dict:
        """Analyze comparison results and compute statistics."""
        # Count wins, ties, errors
        wins_A = sum(1 for r in results if r['winner'] == 'A')
        wins_B = sum(1 for r in results if r['winner'] == 'B')
        ties = sum(1 for r in results if r['winner'] == 'tie')
        errors = sum(1 for r in results if r['winner'] == 'error')
        
        # Count position swaps (for transparency)
        swapped_count = sum(1 for r in results if r.get('position_swapped', False))
        self.logger.info(f"Position randomization: {swapped_count}/{len(results)} comparisons had swapped positions")
        
        # Calculate average scores
        valid_results = [r for r in results if r['status'] == 'success']
        
        if valid_results:
            avg_score_A = sum(r['score_A'] for r in valid_results) / len(valid_results)
            avg_score_B = sum(r['score_B'] for r in valid_results) / len(valid_results)
        else:
            avg_score_A = 0.0
            avg_score_B = 0.0
        
        # Confidence breakdown
        high_conf = sum(1 for r in results if r['confidence'] == 'high')
        med_conf = sum(1 for r in results if r['confidence'] == 'medium')
        low_conf = sum(1 for r in results if r['confidence'] == 'low')
        
        # Win percentages (excluding errors)
        valid_comparisons = len(results) - errors
        if valid_comparisons > 0:
            win_pct_A = (wins_A / valid_comparisons) * 100
            win_pct_B = (wins_B / valid_comparisons) * 100
            tie_pct = (ties / valid_comparisons) * 100
        else:
            win_pct_A = win_pct_B = tie_pct = 0.0
        
        analysis = {
            'total_comparisons': len(results),
            'valid_comparisons': valid_comparisons,
            'errors': errors,
            'position_swaps': swapped_count,
            'swap_percentage': (swapped_count / len(results) * 100) if len(results) > 0 else 0.0,
            f'{checkpoint1_name}_wins': wins_A,
            f'{checkpoint2_name}_wins': wins_B,
            'ties': ties,
            f'{checkpoint1_name}_win_percentage': win_pct_A,
            f'{checkpoint2_name}_win_percentage': win_pct_B,
            'tie_percentage': tie_pct,
            f'{checkpoint1_name}_avg_score': avg_score_A,
            f'{checkpoint2_name}_avg_score': avg_score_B,
            'high_confidence': high_conf,
            'medium_confidence': med_conf,
            'low_confidence': low_conf
        }
        
        return analysis
    
    def generate_report(
        self, 
        results: List[Dict], 
        analysis: Dict,
        checkpoint1_name: str,
        checkpoint2_name: str,
        output_dir: Path
    ):
        """Generate a comprehensive text report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"comparison_report_{checkpoint1_name}_vs_{checkpoint2_name}_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CHECKPOINT COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Comparison: {checkpoint1_name} vs {checkpoint2_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Used: {self.model}\n\n")
            
            f.write("="*80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Comparisons: {analysis['total_comparisons']}\n")
            f.write(f"Valid Comparisons: {analysis['valid_comparisons']}\n")
            f.write(f"Errors: {analysis['errors']}\n")
            f.write(f"Position Randomization: {analysis['position_swaps']} swaps ({analysis['swap_percentage']:.1f}%)\n")
            f.write(f"  Note: Answer positions were randomized to reduce position bias\n\n")
            
            f.write(f"🏆 WINNER BREAKDOWN:\n")
            f.write(f"  {checkpoint1_name} wins: {analysis[f'{checkpoint1_name}_wins']} ({analysis[f'{checkpoint1_name}_win_percentage']:.1f}%)\n")
            f.write(f"  {checkpoint2_name} wins: {analysis[f'{checkpoint2_name}_wins']} ({analysis[f'{checkpoint2_name}_win_percentage']:.1f}%)\n")
            f.write(f"  Ties: {analysis['ties']} ({analysis['tie_percentage']:.1f}%)\n\n")
            
            f.write(f"📊 AVERAGE SCORES (out of 10):\n")
            f.write(f"  {checkpoint1_name}: {analysis[f'{checkpoint1_name}_avg_score']:.2f}\n")
            f.write(f"  {checkpoint2_name}: {analysis[f'{checkpoint2_name}_avg_score']:.2f}\n")
            f.write(f"  Difference: {abs(analysis[f'{checkpoint1_name}_avg_score'] - analysis[f'{checkpoint2_name}_avg_score']):.2f}\n\n")
            
            f.write(f"🎯 CONFIDENCE LEVELS:\n")
            f.write(f"  High confidence: {analysis['high_confidence']}\n")
            f.write(f"  Medium confidence: {analysis['medium_confidence']}\n")
            f.write(f"  Low confidence: {analysis['low_confidence']}\n\n")
            
            # Determine overall winner
            if analysis[f'{checkpoint1_name}_wins'] > analysis[f'{checkpoint2_name}_wins']:
                winner = checkpoint1_name
                margin = analysis[f'{checkpoint1_name}_win_percentage'] - analysis[f'{checkpoint2_name}_win_percentage']
            elif analysis[f'{checkpoint2_name}_wins'] > analysis[f'{checkpoint1_name}_wins']:
                winner = checkpoint2_name
                margin = analysis[f'{checkpoint2_name}_win_percentage'] - analysis[f'{checkpoint1_name}_win_percentage']
            else:
                winner = "TIE"
                margin = 0
            
            f.write("="*80 + "\n")
            f.write("🏅 OVERALL WINNER\n")
            f.write("="*80 + "\n\n")
            
            if winner == "TIE":
                f.write(f"Result: TIE - Both checkpoints performed equally well\n\n")
            else:
                f.write(f"Winner: {winner}\n")
                f.write(f"Margin: {margin:.1f}% more wins\n\n")
            
            # Sample of wins for each checkpoint
            f.write("="*80 + "\n")
            f.write(f"SAMPLE WINS FOR {checkpoint1_name}\n")
            f.write("="*80 + "\n\n")
            
            wins_A = [r for r in results if r['winner'] == 'A' and r['status'] == 'success'][:5]
            for i, result in enumerate(wins_A, 1):
                f.write(f"Example {i}:\n")
                f.write(f"Question: {result['question']}\n")
                f.write(f"Reasoning: {result['reasoning']}\n")
                f.write(f"Scores: A={result['score_A']}, B={result['score_B']}\n\n")
            
            f.write("="*80 + "\n")
            f.write(f"SAMPLE WINS FOR {checkpoint2_name}\n")
            f.write("="*80 + "\n\n")
            
            wins_B = [r for r in results if r['winner'] == 'B' and r['status'] == 'success'][:5]
            for i, result in enumerate(wins_B, 1):
                f.write(f"Example {i}:\n")
                f.write(f"Question: {result['question']}\n")
                f.write(f"Reasoning: {result['reasoning']}\n")
                f.write(f"Scores: A={result['score_A']}, B={result['score_B']}\n\n")
        
        self.logger.info(f"Report saved to: {report_file}")
        return report_file
    
    def generate_plots(
        self, 
        results: List[Dict], 
        analysis: Dict,
        checkpoint1_name: str,
        checkpoint2_name: str,
        output_dir: Path
    ):
        """Generate visualization plots."""
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Plotting libraries not available, skipping plots")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Checkpoint Comparison: {checkpoint1_name} vs {checkpoint2_name}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Win Distribution (Pie Chart)
        ax1 = axes[0, 0]
        wins_A = analysis[f'{checkpoint1_name}_wins']
        wins_B = analysis[f'{checkpoint2_name}_wins']
        ties = analysis['ties']
        
        sizes = [wins_A, wins_B, ties]
        labels = [f'{checkpoint1_name}\n({wins_A})', f'{checkpoint2_name}\n({wins_B})', f'Ties\n({ties})']
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        explode = (0.05, 0.05, 0.05)
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=explode, shadow=True)
        ax1.set_title('Win Distribution', fontweight='bold')
        
        # Plot 2: Average Scores Comparison (Bar Chart)
        ax2 = axes[0, 1]
        checkpoints = [checkpoint1_name, checkpoint2_name]
        avg_scores = [
            analysis[f'{checkpoint1_name}_avg_score'],
            analysis[f'{checkpoint2_name}_avg_score']
        ]
        bars = ax2.bar(checkpoints, avg_scores, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax2.set_ylabel('Average Score (out of 10)', fontweight='bold')
        ax2.set_title('Average GPT Scores', fontweight='bold')
        ax2.set_ylim([0, 10])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Confidence Distribution
        ax3 = axes[0, 2]
        confidence_levels = ['High', 'Medium', 'Low']
        confidence_counts = [
            analysis['high_confidence'],
            analysis['medium_confidence'],
            analysis['low_confidence']
        ]
        bars = ax3.bar(confidence_levels, confidence_counts, 
                      color=['#27ae60', '#f39c12', '#e67e22'], alpha=0.7)
        ax3.set_ylabel('Count', fontweight='bold')
        ax3.set_title('Confidence Distribution', fontweight='bold')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Score Distribution for Checkpoint 1
        ax4 = axes[1, 0]
        valid_results = [r for r in results if r['status'] == 'success']
        scores_A = [r['score_A'] for r in valid_results]
        
        ax4.hist(scores_A, bins=10, range=(0, 10), color='#2ecc71', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Score', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title(f'{checkpoint1_name} Score Distribution', fontweight='bold')
        ax4.axvline(analysis[f'{checkpoint1_name}_avg_score'], color='red', 
                   linestyle='--', linewidth=2, label=f'Avg: {analysis[f"{checkpoint1_name}_avg_score"]:.2f}')
        ax4.legend()
        
        # Plot 5: Score Distribution for Checkpoint 2
        ax5 = axes[1, 1]
        scores_B = [r['score_B'] for r in valid_results]
        
        ax5.hist(scores_B, bins=10, range=(0, 10), color='#e74c3c', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Score', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title(f'{checkpoint2_name} Score Distribution', fontweight='bold')
        ax5.axvline(analysis[f'{checkpoint2_name}_avg_score'], color='red', 
                   linestyle='--', linewidth=2, label=f'Avg: {analysis[f"{checkpoint2_name}_avg_score"]:.2f}')
        ax5.legend()
        
        # Plot 6: Score Comparison Scatter Plot
        ax6 = axes[1, 2]
        
        # Color points based on winner
        colors_scatter = []
        for r in valid_results:
            if r['winner'] == 'A':
                colors_scatter.append('#2ecc71')
            elif r['winner'] == 'B':
                colors_scatter.append('#e74c3c')
            else:
                colors_scatter.append('#95a5a6')
        
        ax6.scatter(scores_A, scores_B, c=colors_scatter, alpha=0.6, s=50)
        ax6.plot([0, 10], [0, 10], 'k--', alpha=0.3, linewidth=1)
        ax6.set_xlabel(f'{checkpoint1_name} Score', fontweight='bold')
        ax6.set_ylabel(f'{checkpoint2_name} Score', fontweight='bold')
        ax6.set_title('Score Comparison', fontweight='bold')
        ax6.set_xlim([0, 10])
        ax6.set_ylim([0, 10])
        ax6.grid(True, alpha=0.3)
        
        # Add legend
        green_patch = mpatches.Patch(color='#2ecc71', label=f'{checkpoint1_name} wins')
        red_patch = mpatches.Patch(color='#e74c3c', label=f'{checkpoint2_name} wins')
        gray_patch = mpatches.Patch(color='#95a5a6', label='Ties')
        ax6.legend(handles=[green_patch, red_patch, gray_patch], loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / f"comparison_plots_{checkpoint1_name}_vs_{checkpoint2_name}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plots saved to: {plot_file}")
        
        plt.close()
    
    def save_detailed_results(
        self, 
        results: List[Dict], 
        checkpoint1_name: str,
        checkpoint2_name: str,
        output_dir: Path
    ):
        """Save detailed comparison results to JSON and CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = output_dir / f"comparison_detailed_{checkpoint1_name}_vs_{checkpoint2_name}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Detailed results saved to: {json_file}")
        
        # Save CSV if pandas is available
        if PANDAS_AVAILABLE:
            csv_file = output_dir / f"comparison_detailed_{checkpoint1_name}_vs_{checkpoint2_name}_{timestamp}.csv"
            df = pd.DataFrame(results)
            df.to_csv(csv_file, index=False)
            self.logger.info(f"CSV results saved to: {csv_file}")
    
    def run_comparison(
        self, 
        file1: str, 
        file2: str, 
        output_dir: str = './comparison_results'
    ):
        """Run the complete comparison pipeline."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("="*80)
        self.logger.info("CHECKPOINT COMPARISON STARTED")
        self.logger.info("="*80)
        
        # Load result files
        data1 = self.load_result_file(file1)
        data2 = self.load_result_file(file2)
        
        checkpoint1_name = data1.get('checkpoint_name', Path(file1).stem)
        checkpoint2_name = data2.get('checkpoint_name', Path(file2).stem)
        
        # Match responses
        matched_pairs = self.match_responses(data1, data2)
        
        if not matched_pairs:
            self.logger.error("No matching questions found between files!")
            return
        
        # Compare responses
        comparison_results = self.compare_responses_batch(
            matched_pairs, 
            checkpoint1_name,
            checkpoint2_name
        )
        
        # Analyze results
        print("📊 Analyzing results...")
        analysis = self.analyze_results(comparison_results, checkpoint1_name, checkpoint2_name)
        
        # Generate outputs
        self.logger.info("Generating outputs...")
        print("\n📄 Generating outputs...")
        
        # Text report
        print("  ├─ Creating text report...")
        self.generate_report(comparison_results, analysis, checkpoint1_name, checkpoint2_name, output_path)
        
        # Plots
        print("  ├─ Generating visualization plots...")
        self.generate_plots(comparison_results, analysis, checkpoint1_name, checkpoint2_name, output_path)
        
        # Detailed results
        print("  └─ Saving detailed results (JSON/CSV)...")
        self.save_detailed_results(comparison_results, checkpoint1_name, checkpoint2_name, output_path)
        
        self.logger.info("="*80)
        self.logger.info("COMPARISON COMPLETED SUCCESSFULLY!")
        self.logger.info("="*80)
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 COMPARISON SUMMARY")
        print("="*80)
        print(f"{checkpoint1_name} wins: {analysis[f'{checkpoint1_name}_wins']} ({analysis[f'{checkpoint1_name}_win_percentage']:.1f}%)")
        print(f"{checkpoint2_name} wins: {analysis[f'{checkpoint2_name}_wins']} ({analysis[f'{checkpoint2_name}_win_percentage']:.1f}%)")
        print(f"Ties: {analysis['ties']} ({analysis['tie_percentage']:.1f}%)")
        print(f"\nAverage Scores:")
        print(f"  {checkpoint1_name}: {analysis[f'{checkpoint1_name}_avg_score']:.2f}/10")
        print(f"  {checkpoint2_name}: {analysis[f'{checkpoint2_name}_avg_score']:.2f}/10")
        print(f"\n📁 Results saved to: {output_path}")
        print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two checkpoint results using GPT evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_checkpoints_gpt.py --file1 result1.json --file2 result2.json --api-key YOUR_KEY
  python compare_checkpoints_gpt.py --file1 result1.json --file2 result2.json --config config.yaml
  python compare_checkpoints_gpt.py --file1 result1.json --file2 result2.json --api-key YOUR_KEY --output ./my_results
        """
    )
    
    parser.add_argument(
        '--file1',
        type=str,
        required=True,
        help='Path to first benchmark result JSON file'
    )
    
    parser.add_argument(
        '--file2',
        type=str,
        required=True,
        help='Path to second benchmark result JSON file'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (or use --config)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config YAML file containing OpenAI API settings'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model to use (default: gpt-4o-mini)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum number of concurrent API requests (default: 5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./comparison_results_2',
        help='Output directory for results (default: ./comparison_results)'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key
    model = args.model
    max_workers = args.max_workers
    
    if args.config:
        # Load from config file
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        openai_config = config.get('openai_api', {})
        api_key = api_key or openai_config.get('api_key')
        model = openai_config.get('model', model)
        max_workers = openai_config.get('max_workers', max_workers)
    
    # Check for API key in environment if not provided
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("Error: OpenAI API key not provided!")
        print("Please provide it via:")
        print("  --api-key YOUR_KEY")
        print("  --config config.yaml (with openai_api.api_key field)")
        print("  OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Validate files exist
    if not Path(args.file1).exists():
        print(f"Error: File not found: {args.file1}")
        sys.exit(1)
    
    if not Path(args.file2).exists():
        print(f"Error: File not found: {args.file2}")
        sys.exit(1)
    
    # Run comparison
    comparator = CheckpointComparator(api_key=api_key, model=model, max_workers=max_workers)
    comparator.run_comparison(args.file1, args.file2, args.output)


if __name__ == "__main__":
    main()

