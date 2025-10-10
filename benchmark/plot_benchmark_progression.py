#!/usr/bin/env python3
"""
Plot benchmark progression through training checkpoints.

Usage:
    python plot_benchmark_progression.py vllm_test_run_vllm_server_summary_20251010_100014.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path

def parse_checkpoint_name(checkpoint_name):
    """Extract step number from checkpoint name for ordering."""
    if checkpoint_name == 'base_model':
        return -1  # Base model comes first
    elif checkpoint_name == 'final':
        return float('inf')  # Final comes last
    elif checkpoint_name.startswith('step-'):
        return int(checkpoint_name.split('-')[1])
    else:
        return 0

def create_benchmark_plots(csv_file):
    """Create progression plots for each benchmark type."""
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Add step number for proper ordering
    df['step_number'] = df['checkpoint_name'].apply(parse_checkpoint_name)
    df = df.sort_values('step_number')
    
    # Get unique benchmarks
    benchmarks = df['benchmark_name'].unique()
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create subplots - bar plots and scatter plots for each benchmark
    n_benchmarks = len(benchmarks)
    fig, axes = plt.subplots(n_benchmarks, 2, figsize=(20, 6 * n_benchmarks))
    
    # If only one benchmark, make axes a 2D array for consistency
    if n_benchmarks == 1:
        axes = axes.reshape(1, -1)
    
    # Color mapping for different checkpoint types
    colors = {
        'base_model': '#e74c3c',  # Red
        'step': '#3498db',        # Blue
        'final': '#2ecc71'        # Green
    }
    
    for i, benchmark in enumerate(benchmarks):
        # Filter data for this benchmark
        benchmark_data = df[df['benchmark_name'] == benchmark].copy()
        
        # === BAR PLOT (Left column) ===
        ax_bar = axes[i, 0]
        
        # Prepare data for bar plotting
        checkpoints = []
        scores = []
        colors_list = []
        
        for _, row in benchmark_data.iterrows():
            checkpoint = row['checkpoint_name']
            score = row['metric_average_score']
            
            checkpoints.append(checkpoint)
            scores.append(score)
            
            # Assign colors based on checkpoint type
            if checkpoint == 'base_model':
                colors_list.append(colors['base_model'])
            elif checkpoint == 'final':
                colors_list.append(colors['final'])
            else:
                colors_list.append(colors['step'])
        
        # Create the bar plot
        bars = ax_bar.bar(range(len(checkpoints)), scores, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize the bar plot
        ax_bar.set_title(f'{benchmark.replace("_", " ").title()} - Score Comparison', 
                        fontsize=14, fontweight='bold', pad=20)
        ax_bar.set_xlabel('Checkpoint', fontsize=12)
        ax_bar.set_ylabel('Average Score', fontsize=12)
        ax_bar.set_ylim(0, 1.05)
        
        # Set x-axis labels
        ax_bar.set_xticks(range(len(checkpoints)))
        ax_bar.set_xticklabels(checkpoints, rotation=45, ha='right')
        
        # Add value labels on bars
        for j, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add grid for better readability
        ax_bar.grid(True, alpha=0.3, axis='y')
        ax_bar.set_axisbelow(True)
        
        # === SCATTER PLOT (Right column) ===
        ax_scatter = axes[i, 1]
        
        # Prepare data for scatter plot (only training steps, not base/final)
        training_data = benchmark_data[
            (benchmark_data['checkpoint_name'] != 'base_model') & 
            (benchmark_data['checkpoint_name'] != 'final')
        ].copy()
        
        base_score = benchmark_data[benchmark_data['checkpoint_name'] == 'base_model']['metric_average_score'].iloc[0]
        final_score = benchmark_data[benchmark_data['checkpoint_name'] == 'final']['metric_average_score'].iloc[0]
        
        if len(training_data) > 0:
            # Extract step numbers and scores for training steps
            step_numbers = []
            step_scores = []
            
            for _, row in training_data.iterrows():
                step_num = int(row['checkpoint_name'].split('-')[1])
                step_numbers.append(step_num)
                step_scores.append(row['metric_average_score'])
            
            # Create scatter plot
            ax_scatter.scatter(step_numbers, step_scores, color=colors['step'], s=100, alpha=0.8, 
                             edgecolor='black', linewidth=1, label='Training Steps', zorder=3)
            
            # Add trend line if we have multiple points (without slope in label)
            if len(step_numbers) > 1:
                z = np.polyfit(step_numbers, step_scores, 1)
                p = np.poly1d(z)
                ax_scatter.plot(step_numbers, p(step_numbers), '--', color=colors['step'], 
                              alpha=0.7, linewidth=2, label='Trend Line')
            
            # Add base model and final model as reference points
            ax_scatter.axhline(y=base_score, color=colors['base_model'], linestyle='-', alpha=0.8, 
                             linewidth=2, label=f'Base Model ({base_score:.3f})')
            ax_scatter.axhline(y=final_score, color=colors['final'], linestyle='-', alpha=0.8, 
                             linewidth=2, label=f'Final Model ({final_score:.3f})')
            
            # Customize scatter plot
            ax_scatter.set_title(f'{benchmark.replace("_", " ").title()} - Training Progression', 
                               fontsize=14, fontweight='bold', pad=20)
            ax_scatter.set_xlabel('Training Step', fontsize=12)
            ax_scatter.set_ylabel('Average Score', fontsize=12)
            ax_scatter.grid(True, alpha=0.3)
            ax_scatter.set_axisbelow(True)
            
            # Set y-axis limits with some padding
            y_min = min(min(step_scores), base_score, final_score) - 0.05
            y_max = max(max(step_scores), base_score, final_score) + 0.05
            ax_scatter.set_ylim(y_min, y_max)
            
            # Add legend
            ax_scatter.legend(loc='best')
            
            # Calculate improvement metrics
            if len(step_scores) > 1:
                initial_step_score = step_scores[0]
                final_step_score = step_scores[-1]
                step_improvement = ((final_step_score - initial_step_score) / initial_step_score) * 100
                
                # Add improvement text
                ax_scatter.text(0.02, 0.98, 
                              f'Step {step_numbers[0]}→{step_numbers[-1]}: {step_improvement:+.1f}%\n'
                              f'Base→Final: {((final_score - base_score) / base_score) * 100:+.1f}%', 
                              transform=ax_scatter.transAxes, ha='left', va='top',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                              fontsize=10, fontweight='bold')
        else:
            ax_scatter.text(0.5, 0.5, 'No training step data available', 
                          transform=ax_scatter.transAxes, ha='center', va='center',
                          fontsize=12, style='italic')
            ax_scatter.set_title(f'{benchmark.replace("_", " ").title()} - No Training Data', 
                               fontsize=14, fontweight='bold', pad=20)
        
        # Add base model reference line to bar plot
        ax_bar.axhline(y=base_score, color=colors['base_model'], linestyle='--', alpha=0.7, 
                      label=f'Base Model ({base_score:.3f})')
        
        # Add legend to bar plot
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=colors['base_model'], alpha=0.8, label='Base Model'),
            plt.Rectangle((0,0),1,1, facecolor=colors['step'], alpha=0.8, label='Training Steps'),
            plt.Rectangle((0,0),1,1, facecolor=colors['final'], alpha=0.8, label='Final Model')
        ]
        ax_bar.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # Calculate and display improvement on bar plot
        final_score = benchmark_data[benchmark_data['checkpoint_name'] == 'final']['metric_average_score'].iloc[0]
        improvement = ((final_score - base_score) / base_score) * 100
        
        # Add improvement text to bar plot
        ax_bar.text(0.98, 0.02, f'Final vs Base: {improvement:+.1f}%', 
                   transform=ax_bar.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7),
                   fontsize=10, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = csv_file.replace('.csv', '_progression_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Plots saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    # Create a summary table
    create_summary_table(df, csv_file)

def create_summary_table(df, csv_file):
    """Create a summary table showing key metrics."""
    print("\n" + "="*80)
    print("📈 BENCHMARK PROGRESSION SUMMARY")
    print("="*80)
    
    benchmarks = df['benchmark_name'].unique()
    
    for benchmark in benchmarks:
        benchmark_data = df[df['benchmark_name'] == benchmark].copy()
        benchmark_data = benchmark_data.sort_values('step_number')
        
        print(f"\n🎯 {benchmark.replace('_', ' ').title()}:")
        print("-" * 50)
        
        base_score = benchmark_data[benchmark_data['checkpoint_name'] == 'base_model']['metric_average_score'].iloc[0]
        final_score = benchmark_data[benchmark_data['checkpoint_name'] == 'final']['metric_average_score'].iloc[0]
        best_step = benchmark_data.loc[benchmark_data['metric_average_score'].idxmax()]
        
        print(f"Base Model Score:    {base_score:.4f}")
        print(f"Final Model Score:   {final_score:.4f}")
        print(f"Best Score:          {best_step['metric_average_score']:.4f} ({best_step['checkpoint_name']})")
        print(f"Improvement:         {((final_score - base_score) / base_score) * 100:+.2f}%")
        print(f"Samples Evaluated:   {benchmark_data['metric_num_samples'].iloc[0]}")

def main():
    parser = argparse.ArgumentParser(description='Plot benchmark progression through training checkpoints')
    parser.add_argument('csv_file', help='Path to the benchmark summary CSV file')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"❌ Error: File {args.csv_file} not found!")
        return
    
    create_benchmark_plots(args.csv_file)

if __name__ == "__main__":
    main()
