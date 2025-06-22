# -*- coding: utf-8 -*-
"""
LBA Results Analyzer and Visualizer for 18 Participants
Analyzes outcomes and creates comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import ast
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_and_process_results(results_file: str, original_data_file: str) -> Dict:
    """
    Load and process LBA results with original data
    """
    
    print("📊 Loading and processing results...")
    
    # Load results
    results_df = pd.read_csv(results_file)
    original_df = pd.read_csv(original_data_file)
    
    # Process choice distributions
    def parse_choice_dist(dist_str):
        try:
            return ast.literal_eval(dist_str)
        except:
            return {}
    
    results_df['choice_dist_parsed'] = results_df['choice_distribution'].apply(parse_choice_dist)
    
    # Add original data metrics
    enhanced_results = []
    
    for _, row in results_df.iterrows():
        subject_id = row['subject_id']
        subject_data = original_df[original_df['participant'] == subject_id]
        
        if len(subject_data) > 0:
            # Calculate additional metrics
            accuracy = subject_data['Correct'].mean()
            choice_data = subject_data['Response'].values
            rt_data = subject_data['RT'].values
            
            # Choice patterns
            choice_counts = {i: np.sum(choice_data == i) for i in range(4)}
            total_trials = len(choice_data)
            
            # Symmetry analysis
            left_diagonal = choice_counts[0] + choice_counts[1]  # \ choices
            left_vertical = choice_counts[2] + choice_counts[3]   # | choices (left)
            right_vertical = choice_counts[0] + choice_counts[2]  # | choices (right)
            right_diagonal = choice_counts[1] + choice_counts[3]  # / choices
            
            left_bias = (left_diagonal - left_vertical) / total_trials
            right_bias = (right_diagonal - right_vertical) / total_trials
            
            # Enhanced row
            enhanced_row = row.to_dict()
            enhanced_row.update({
                'accuracy': accuracy,
                'choice_0': choice_counts[0],
                'choice_1': choice_counts[1], 
                'choice_2': choice_counts[2],
                'choice_3': choice_counts[3],
                'left_bias': left_bias,
                'right_bias': right_bias,
                'absolute_left_bias': abs(left_bias),
                'absolute_right_bias': abs(right_bias),
                'bilateral_bias_diff': abs(left_bias - right_bias),
                'rt_std': np.std(rt_data),
                'choice_entropy': -sum([p * np.log(p) for p in [choice_counts[i]/total_trials for i in range(4)] if p > 0])
            })
            
            enhanced_results.append(enhanced_row)
    
    enhanced_df = pd.DataFrame(enhanced_results)
    
    print(f"✅ Processed {len(enhanced_df)} subjects")
    print(f"   Success rate: {enhanced_df['success'].mean():.1%}")
    print(f"   Convergence rate: {enhanced_df['converged'].mean():.1%}")
    
    return {
        'results_df': enhanced_df,
        'original_df': original_df,
        'summary_stats': calculate_summary_statistics(enhanced_df)
    }

def calculate_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive summary statistics
    """
    
    successful_df = df[df['success'] == True]
    
    if len(successful_df) == 0:
        return {'error': 'No successful analyses'}
    
    stats = {
        'n_subjects': len(df),
        'n_successful': len(successful_df),
        'success_rate': len(successful_df) / len(df),
        
        # Performance metrics
        'accuracy_mean': successful_df['accuracy'].mean(),
        'accuracy_std': successful_df['accuracy'].std(),
        'accuracy_range': [successful_df['accuracy'].min(), successful_df['accuracy'].max()],
        
        # Reaction time metrics
        'rt_mean': successful_df['mean_rt'].mean(),
        'rt_std': successful_df['mean_rt'].std(),
        'rt_range': [successful_df['mean_rt'].min(), successful_df['mean_rt'].max()],
        
        # Bias metrics
        'left_bias_mean': successful_df['left_bias'].mean(),
        'right_bias_mean': successful_df['right_bias'].mean(),
        'bilateral_bias_diff_mean': successful_df['bilateral_bias_diff'].mean(),
        
        # Convergence metrics
        'rhat_mean': successful_df['rhat_max'].mean(),
        'ess_mean': successful_df['ess_min'].mean(),
        'sampling_time_mean': successful_df['sampling_time_minutes'].mean(),
        
        # Individual differences
        'accuracy_below_chance': (successful_df['accuracy'] < 0.25).sum(),
        'strong_left_bias': (successful_df['absolute_left_bias'] > 0.2).sum(),
        'strong_right_bias': (successful_df['absolute_right_bias'] > 0.2).sum(),
        'asymmetric_processing': (successful_df['bilateral_bias_diff'] > 0.15).sum()
    }
    
    return stats

def create_overview_visualizations(data: Dict):
    """
    Create comprehensive overview visualizations
    """
    
    df = data['results_df']
    successful_df = df[df['success'] == True]
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Success and Convergence Overview
    ax1 = plt.subplot(4, 4, 1)
    success_data = [df['success'].sum(), (~df['success']).sum()]
    plt.pie(success_data, labels=['Success', 'Failed'], autopct='%1.1f%%', startangle=90)
    plt.title('Analysis Success Rate')
    
    ax2 = plt.subplot(4, 4, 2)
    if len(successful_df) > 0:
        conv_data = [successful_df['converged'].sum(), (~successful_df['converged']).sum()]
        plt.pie(conv_data, labels=['Converged', 'Not Converged'], autopct='%1.1f%%', startangle=90)
    plt.title('Convergence Rate')
    
    # 2. Performance Distribution
    ax3 = plt.subplot(4, 4, 3)
    if len(successful_df) > 0:
        plt.hist(successful_df['accuracy'], bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(0.25, color='red', linestyle='--', label='Chance Level')
        plt.xlabel('Accuracy')
        plt.ylabel('Count')
        plt.title('Accuracy Distribution')
        plt.legend()
    
    # 3. Reaction Time Distribution
    ax4 = plt.subplot(4, 4, 4)
    if len(successful_df) > 0:
        plt.hist(successful_df['mean_rt'], bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Mean RT (seconds)')
        plt.ylabel('Count')
        plt.title('Reaction Time Distribution')
    
    # 4. Sampling Time Analysis
    ax5 = plt.subplot(4, 4, 5)
    if len(successful_df) > 0:
        plt.hist(successful_df['sampling_time_minutes'], bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Sampling Time (minutes)')
        plt.ylabel('Count')
        plt.title('MCMC Sampling Time')
    
    # 5. R-hat Distribution
    ax6 = plt.subplot(4, 4, 6)
    if len(successful_df) > 0:
        plt.hist(successful_df['rhat_max'], bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(1.05, color='red', linestyle='--', label='Good Convergence')
        plt.xlabel('Max R-hat')
        plt.ylabel('Count')
        plt.title('Convergence Quality (R-hat)')
        plt.legend()
    
    # 6. ESS Distribution
    ax7 = plt.subplot(4, 4, 7)
    if len(successful_df) > 0:
        plt.hist(successful_df['ess_min'], bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(100, color='red', linestyle='--', label='Adequate ESS')
        plt.xlabel('Min ESS')
        plt.ylabel('Count')
        plt.title('Effective Sample Size')
        plt.legend()
    
    # 7. Choice Bias Patterns
    ax8 = plt.subplot(4, 4, 8)
    if len(successful_df) > 0:
        plt.scatter(successful_df['left_bias'], successful_df['right_bias'], alpha=0.7)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Left Channel Bias')
        plt.ylabel('Right Channel Bias')
        plt.title('Bilateral Bias Patterns')
    
    # 8-11. Individual Choice Distributions
    for i in range(4):
        ax = plt.subplot(4, 4, 9+i)
        if len(successful_df) > 0:
            choice_col = f'choice_{i}'
            plt.hist(successful_df[choice_col], bins=10, alpha=0.7, edgecolor='black')
            plt.xlabel(f'Choice {i} Count')
            plt.ylabel('Subjects')
            plt.title(f'Choice {i} Distribution')
    
    # 12. Accuracy vs RT Relationship
    ax13 = plt.subplot(4, 4, 13)
    if len(successful_df) > 0:
        plt.scatter(successful_df['mean_rt'], successful_df['accuracy'], alpha=0.7)
        plt.xlabel('Mean RT (seconds)')
        plt.ylabel('Accuracy')
        plt.title('Speed-Accuracy Relationship')
    
    # 13. Bilateral Bias Difference
    ax14 = plt.subplot(4, 4, 14)
    if len(successful_df) > 0:
        plt.hist(successful_df['bilateral_bias_diff'], bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Bilateral Bias Difference')
        plt.ylabel('Count')
        plt.title('Asymmetry in Processing')
    
    # 14. Choice Entropy (Randomness)
    ax15 = plt.subplot(4, 4, 15)
    if len(successful_df) > 0:
        plt.hist(successful_df['choice_entropy'], bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(np.log(4), color='red', linestyle='--', label='Max Entropy')
        plt.xlabel('Choice Entropy')
        plt.ylabel('Count')
        plt.title('Response Randomness')
        plt.legend()
    
    # 15. Subject Performance Ranking
    ax16 = plt.subplot(4, 4, 16)
    if len(successful_df) > 0:
        sorted_df = successful_df.sort_values('accuracy')
        colors = ['red' if acc < 0.25 else 'orange' if acc < 0.5 else 'green' for acc in sorted_df['accuracy']]
        plt.bar(range(len(sorted_df)), sorted_df['accuracy'], color=colors, alpha=0.7)
        plt.axhline(0.25, color='red', linestyle='--', label='Chance')
        plt.xlabel('Subject (ranked)')
        plt.ylabel('Accuracy')
        plt.title('Subject Performance Ranking')
        plt.xticks(range(len(sorted_df)), sorted_df['subject_id'], rotation=45)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('lba_overview_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_individual_differences_analysis(data: Dict):
    """
    Create detailed individual differences analysis
    """
    
    df = data['results_df']
    successful_df = df[df['success'] == True]
    
    if len(successful_df) == 0:
        print("No successful analyses to plot")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. Performance Categories
    ax = axes[0, 0]
    perf_categories = []
    for acc in successful_df['accuracy']:
        if acc < 0.25:
            perf_categories.append('Below Chance')
        elif acc < 0.5:
            perf_categories.append('Poor')
        elif acc < 0.75:
            perf_categories.append('Moderate')
        else:
            perf_categories.append('Good')
    
    cat_counts = pd.Series(perf_categories).value_counts()
    ax.pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%')
    ax.set_title('Performance Categories')
    
    # 2. Bias Strength Categories
    ax = axes[0, 1]
    bias_categories = []
    for _, row in successful_df.iterrows():
        max_bias = max(row['absolute_left_bias'], row['absolute_right_bias'])
        if max_bias < 0.05:
            bias_categories.append('No Bias')
        elif max_bias < 0.15:
            bias_categories.append('Moderate Bias')
        else:
            bias_categories.append('Strong Bias')
    
    bias_counts = pd.Series(bias_categories).value_counts()
    ax.pie(bias_counts.values, labels=bias_counts.index, autopct='%1.1f%%')
    ax.set_title('Choice Bias Strength')
    
    # 3. Processing Symmetry
    ax = axes[0, 2]
    symmetry_categories = []
    for diff in successful_df['bilateral_bias_diff']:
        if diff < 0.05:
            symmetry_categories.append('Symmetric')
        elif diff < 0.15:
            symmetry_categories.append('Mildly Asymmetric')
        else:
            symmetry_categories.append('Strongly Asymmetric')
    
    sym_counts = pd.Series(symmetry_categories).value_counts()
    ax.pie(sym_counts.values, labels=sym_counts.index, autopct='%1.1f%%')
    ax.set_title('Processing Symmetry')
    
    # 4. Heatmap of Choice Patterns
    ax = axes[1, 0]
    choice_matrix = successful_df[['choice_0', 'choice_1', 'choice_2', 'choice_3']].T
    sns.heatmap(choice_matrix, annot=False, cmap='viridis', ax=ax)
    ax.set_title('Choice Patterns Across Subjects')
    ax.set_ylabel('Choice Type')
    ax.set_xlabel('Subjects')
    
    # 5. Correlation Matrix
    ax = axes[1, 1]
    corr_vars = ['accuracy', 'mean_rt', 'left_bias', 'right_bias', 'bilateral_bias_diff', 'choice_entropy']
    corr_matrix = successful_df[corr_vars].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Variable Correlations')
    
    # 6. Subject Clustering (by performance)
    ax = axes[1, 2]
    scatter = ax.scatter(successful_df['mean_rt'], successful_df['accuracy'], 
                        c=successful_df['bilateral_bias_diff'], cmap='plasma', alpha=0.7)
    ax.set_xlabel('Mean RT (seconds)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Subject Clustering')
    plt.colorbar(scatter, ax=ax, label='Bilateral Bias Diff')
    
    # 7. Choice Pattern Comparison
    ax = axes[2, 0]
    choice_means = [successful_df[f'choice_{i}'].mean() for i in range(4)]
    choice_stds = [successful_df[f'choice_{i}'].std() for i in range(4)]
    x_pos = range(4)
    ax.bar(x_pos, choice_means, yerr=choice_stds, capsize=5, alpha=0.7)
    ax.axhline(successful_df['n_trials'].mean()/4, color='red', linestyle='--', label='Expected')
    ax.set_xlabel('Choice Type')
    ax.set_ylabel('Mean Count')
    ax.set_title('Average Choice Usage')
    ax.legend()
    
    # 8. RT vs Bias Relationship
    ax = axes[2, 1]
    total_bias = successful_df['absolute_left_bias'] + successful_df['absolute_right_bias']
    ax.scatter(total_bias, successful_df['mean_rt'], alpha=0.7)
    ax.set_xlabel('Total Absolute Bias')
    ax.set_ylabel('Mean RT (seconds)')
    ax.set_title('Bias vs Processing Speed')
    
    # 9. Model Quality vs Performance
    ax = axes[2, 2]
    ax.scatter(successful_df['rhat_max'], successful_df['accuracy'], alpha=0.7)
    ax.axvline(1.05, color='red', linestyle='--', label='Good R-hat')
    ax.set_xlabel('Max R-hat')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Quality vs Performance')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('individual_differences_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_summary(data: Dict):
    """
    Print comprehensive summary of results
    """
    
    stats = data['summary_stats']
    df = data['results_df']
    
    print("="*80)
    print("COMPREHENSIVE LBA ANALYSIS SUMMARY - 18 PARTICIPANTS")
    print("="*80)
    
    print(f"\n📊 OVERALL SUCCESS METRICS:")
    print(f"   Total subjects analyzed: {stats['n_subjects']}")
    print(f"   Successful analyses: {stats['n_successful']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    
    if stats['n_successful'] > 0:
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   Mean accuracy: {stats['accuracy_mean']:.1%} (SD: {stats['accuracy_std']:.1%})")
        print(f"   Accuracy range: {stats['accuracy_range'][0]:.1%} - {stats['accuracy_range'][1]:.1%}")
        print(f"   Subjects below chance: {stats['accuracy_below_chance']}/{stats['n_successful']}")
        
        print(f"\n⏱️  REACTION TIME SUMMARY:")
        print(f"   Mean RT: {stats['rt_mean']:.3f}s (SD: {stats['rt_std']:.3f}s)")
        print(f"   RT range: {stats['rt_range'][0]:.3f}s - {stats['rt_range'][1]:.3f}s")
        
        print(f"\n🧠 COGNITIVE BIAS SUMMARY:")
        print(f"   Mean left bias: {stats['left_bias_mean']:+.3f}")
        print(f"   Mean right bias: {stats['right_bias_mean']:+.3f}")
        print(f"   Mean bilateral difference: {stats['bilateral_bias_diff_mean']:.3f}")
        print(f"   Strong left bias: {stats['strong_left_bias']}/{stats['n_successful']} subjects")
        print(f"   Strong right bias: {stats['strong_right_bias']}/{stats['n_successful']} subjects")
        print(f"   Asymmetric processing: {stats['asymmetric_processing']}/{stats['n_successful']} subjects")
        
        print(f"\n🔧 MODEL QUALITY SUMMARY:")
        print(f"   Mean R-hat: {stats['rhat_mean']:.3f}")
        print(f"   Mean ESS: {stats['ess_mean']:.0f}")
        print(f"   Mean sampling time: {stats['sampling_time_mean']:.1f} minutes")
        
        # Individual subject details
        successful_df = df[df['success'] == True]
        
        print(f"\n👥 INDIVIDUAL SUBJECT PATTERNS:")
        print("Subject | Accuracy | Mean RT | Left Bias | Right Bias | Asymmetry")
        print("-" * 70)
        
        for _, row in successful_df.iterrows():
            print(f"{row['subject_id']:7d} | {row['accuracy']:8.1%} | {row['mean_rt']:7.3f} | "
                  f"{row['left_bias']:9.3f} | {row['right_bias']:10.3f} | {row['bilateral_bias_diff']:9.3f}")

def main_analysis(results_file: str = 'dual_lba_results_20250615_122314.csv', 
                 original_file: str = 'GRT_LBA.csv'):
    """
    Main analysis function
    """
    
    print("🚀 Starting comprehensive LBA results analysis...")
    
    # Load and process data
    data = load_and_process_results(results_file, original_file)
    
    # Print summary
    print_detailed_summary(data)
    
    # Create visualizations
    print("\n📈 Creating overview visualizations...")
    create_overview_visualizations(data)
    
    print("\n🔍 Creating individual differences analysis...")
    create_individual_differences_analysis(data)
    
    print("\n✅ Analysis complete! Check the generated PNG files.")
    
    return data

if __name__ == "__main__":
    # Run the analysis
    results = main_analysis()# -*- coding: utf-8 -*-
"""
LBA Results Analyzer and Visualizer for 18 Participants
Analyzes outcomes and creates comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import ast
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_and_process_results(results_file: str, original_data_file: str) -> Dict:
    """
    Load and process LBA results with original data
    """
    
    print("📊 Loading and processing results...")
    
    # Load results
    results_df = pd.read_csv(results_file)
    original_df = pd.read_csv(original_data_file)
    
    # Process choice distributions
    def parse_choice_dist(dist_str):
        try:
            return ast.literal_eval(dist_str)
        except:
            return {}
    
    results_df['choice_dist_parsed'] = results_df['choice_distribution'].apply(parse_choice_dist)
    
    # Add original data metrics
    enhanced_results = []
    
    for _, row in results_df.iterrows():
        subject_id = row['subject_id']
        subject_data = original_df[original_df['participant'] == subject_id]
        
        if len(subject_data) > 0:
            # Calculate additional metrics
            accuracy = subject_data['Correct'].mean()
            choice_data = subject_data['Response'].values
            rt_data = subject_data['RT'].values
            
            # Choice patterns
            choice_counts = {i: np.sum(choice_data == i) for i in range(4)}
            total_trials = len(choice_data)
            
            # Symmetry analysis
            left_diagonal = choice_counts[0] + choice_counts[1]  # \ choices
            left_vertical = choice_counts[2] + choice_counts[3]   # | choices (left)
            right_vertical = choice_counts[0] + choice_counts[2]  # | choices (right)
            right_diagonal = choice_counts[1] + choice_counts[3]  # / choices
            
            left_bias = (left_diagonal - left_vertical) / total_trials
            right_bias = (right_diagonal - right_vertical) / total_trials
            
            # Enhanced row
            enhanced_row = row.to_dict()
            enhanced_row.update({
                'accuracy': accuracy,
                'choice_0': choice_counts[0],
                'choice_1': choice_counts[1], 
                'choice_2': choice_counts[2],
                'choice_3': choice_counts[3],
                'left_bias': left_bias,
                'right_bias': right_bias,
                'absolute_left_bias': abs(left_bias),
                'absolute_right_bias': abs(right_bias),
                'bilateral_bias_diff': abs(left_bias - right_bias),
                'rt_std': np.std(rt_data),
                'choice_entropy': -sum([p * np.log(p) for p in [choice_counts[i]/total_trials for i in range(4)] if p > 0])
            })
            
            enhanced_results.append(enhanced_row)
    
    enhanced_df = pd.DataFrame(enhanced_results)
    
    print(f"✅ Processed {len(enhanced_df)} subjects")
    print(f"   Success rate: {enhanced_df['success'].mean():.1%}")
    print(f"   Convergence rate: {enhanced_df['converged'].mean():.1%}")
    
    return {
        'results_df': enhanced_df,
        'original_df': original_df,
        'summary_stats': calculate_summary_statistics(enhanced_df)
    }

def calculate_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive summary statistics
    """
    
    successful_df = df[df['success'] == True]
    
    if len(successful_df) == 0:
        return {'error': 'No successful analyses'}
    
    stats = {
        'n_subjects': len(df),
        'n_successful': len(successful_df),
        'success_rate': len(successful_df) / len(df),
        
        # Performance metrics
        'accuracy_mean': successful_df['accuracy'].mean(),
        'accuracy_std': successful_df['accuracy'].std(),
        'accuracy_range': [successful_df['accuracy'].min(), successful_df['accuracy'].max()],
        
        # Reaction time metrics
        'rt_mean': successful_df['mean_rt'].mean(),
        'rt_std': successful_df['mean_rt'].std(),
        'rt_range': [successful_df['mean_rt'].min(), successful_df['mean_rt'].max()],
        
        # Bias metrics
        'left_bias_mean': successful_df['left_bias'].mean(),
        'right_bias_mean': successful_df['right_bias'].mean(),
        'bilateral_bias_diff_mean': successful_df['bilateral_bias_diff'].mean(),
        
        # Convergence metrics
        'rhat_mean': successful_df['rhat_max'].mean(),
        'ess_mean': successful_df['ess_min'].mean(),
        'sampling_time_mean': successful_df['sampling_time_minutes'].mean(),
        
        # Individual differences
        'accuracy_below_chance': (successful_df['accuracy'] < 0.25).sum(),
        'strong_left_bias': (successful_df['absolute_left_bias'] > 0.2).sum(),
        'strong_right_bias': (successful_df['absolute_right_bias'] > 0.2).sum(),
        'asymmetric_processing': (successful_df['bilateral_bias_diff'] > 0.15).sum()
    }
    
    return stats

def create_overview_visualizations(data: Dict):
    """
    Create comprehensive overview visualizations
    """
    
    df = data['results_df']
    successful_df = df[df['success'] == True]
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Success and Convergence Overview
    ax1 = plt.subplot(4, 4, 1)
    success_data = [df['success'].sum(), (~df['success']).sum()]
    plt.pie(success_data, labels=['Success', 'Failed'], autopct='%1.1f%%', startangle=90)
    plt.title('Analysis Success Rate')
    
    ax2 = plt.subplot(4, 4, 2)
    if len(successful_df) > 0:
        conv_data = [successful_df['converged'].sum(), (~successful_df['converged']).sum()]
        plt.pie(conv_data, labels=['Converged', 'Not Converged'], autopct='%1.1f%%', startangle=90)
    plt.title('Convergence Rate')
    
    # 2. Performance Distribution
    ax3 = plt.subplot(4, 4, 3)
    if len(successful_df) > 0:
        plt.hist(successful_df['accuracy'], bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(0.25, color='red', linestyle='--', label='Chance Level')
        plt.xlabel('Accuracy')
        plt.ylabel('Count')
        plt.title('Accuracy Distribution')
        plt.legend()
    
    # 3. Reaction Time Distribution
    ax4 = plt.subplot(4, 4, 4)
    if len(successful_df) > 0:
        plt.hist(successful_df['mean_rt'], bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Mean RT (seconds)')
        plt.ylabel('Count')
        plt.title('Reaction Time Distribution')
    
    # 4. Sampling Time Analysis
    ax5 = plt.subplot(4, 4, 5)
    if len(successful_df) > 0:
        plt.hist(successful_df['sampling_time_minutes'], bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Sampling Time (minutes)')
        plt.ylabel('Count')
        plt.title('MCMC Sampling Time')
    
    # 5. R-hat Distribution
    ax6 = plt.subplot(4, 4, 6)
    if len(successful_df) > 0:
        plt.hist(successful_df['rhat_max'], bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(1.05, color='red', linestyle='--', label='Good Convergence')
        plt.xlabel('Max R-hat')
        plt.ylabel('Count')
        plt.title('Convergence Quality (R-hat)')
        plt.legend()
    
    # 6. ESS Distribution
    ax7 = plt.subplot(4, 4, 7)
    if len(successful_df) > 0:
        plt.hist(successful_df['ess_min'], bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(100, color='red', linestyle='--', label='Adequate ESS')
        plt.xlabel('Min ESS')
        plt.ylabel('Count')
        plt.title('Effective Sample Size')
        plt.legend()
    
    # 7. Choice Bias Patterns
    ax8 = plt.subplot(4, 4, 8)
    if len(successful_df) > 0:
        plt.scatter(successful_df['left_bias'], successful_df['right_bias'], alpha=0.7)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Left Channel Bias')
        plt.ylabel('Right Channel Bias')
        plt.title('Bilateral Bias Patterns')
    
    # 8-11. Individual Choice Distributions
    for i in range(4):
        ax = plt.subplot(4, 4, 9+i)
        if len(successful_df) > 0:
            choice_col = f'choice_{i}'
            plt.hist(successful_df[choice_col], bins=10, alpha=0.7, edgecolor='black')
            plt.xlabel(f'Choice {i} Count')
            plt.ylabel('Subjects')
            plt.title(f'Choice {i} Distribution')
    
    # 12. Accuracy vs RT Relationship
    ax13 = plt.subplot(4, 4, 13)
    if len(successful_df) > 0:
        plt.scatter(successful_df['mean_rt'], successful_df['accuracy'], alpha=0.7)
        plt.xlabel('Mean RT (seconds)')
        plt.ylabel('Accuracy')
        plt.title('Speed-Accuracy Relationship')
    
    # 13. Bilateral Bias Difference
    ax14 = plt.subplot(4, 4, 14)
    if len(successful_df) > 0:
        plt.hist(successful_df['bilateral_bias_diff'], bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Bilateral Bias Difference')
        plt.ylabel('Count')
        plt.title('Asymmetry in Processing')
    
    # 14. Choice Entropy (Randomness)
    ax15 = plt.subplot(4, 4, 15)
    if len(successful_df) > 0:
        plt.hist(successful_df['choice_entropy'], bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(np.log(4), color='red', linestyle='--', label='Max Entropy')
        plt.xlabel('Choice Entropy')
        plt.ylabel('Count')
        plt.title('Response Randomness')
        plt.legend()
    
    # 15. Subject Performance Ranking
    ax16 = plt.subplot(4, 4, 16)
    if len(successful_df) > 0:
        sorted_df = successful_df.sort_values('accuracy')
        colors = ['red' if acc < 0.25 else 'orange' if acc < 0.5 else 'green' for acc in sorted_df['accuracy']]
        plt.bar(range(len(sorted_df)), sorted_df['accuracy'], color=colors, alpha=0.7)
        plt.axhline(0.25, color='red', linestyle='--', label='Chance')
        plt.xlabel('Subject (ranked)')
        plt.ylabel('Accuracy')
        plt.title('Subject Performance Ranking')
        plt.xticks(range(len(sorted_df)), sorted_df['subject_id'], rotation=45)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('lba_overview_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_individual_differences_analysis(data: Dict):
    """
    Create detailed individual differences analysis
    """
    
    df = data['results_df']
    successful_df = df[df['success'] == True]
    
    if len(successful_df) == 0:
        print("No successful analyses to plot")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. Performance Categories
    ax = axes[0, 0]
    perf_categories = []
    for acc in successful_df['accuracy']:
        if acc < 0.25:
            perf_categories.append('Below Chance')
        elif acc < 0.5:
            perf_categories.append('Poor')
        elif acc < 0.75:
            perf_categories.append('Moderate')
        else:
            perf_categories.append('Good')
    
    cat_counts = pd.Series(perf_categories).value_counts()
    ax.pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%')
    ax.set_title('Performance Categories')
    
    # 2. Bias Strength Categories
    ax = axes[0, 1]
    bias_categories = []
    for _, row in successful_df.iterrows():
        max_bias = max(row['absolute_left_bias'], row['absolute_right_bias'])
        if max_bias < 0.05:
            bias_categories.append('No Bias')
        elif max_bias < 0.15:
            bias_categories.append('Moderate Bias')
        else:
            bias_categories.append('Strong Bias')
    
    bias_counts = pd.Series(bias_categories).value_counts()
    ax.pie(bias_counts.values, labels=bias_counts.index, autopct='%1.1f%%')
    ax.set_title('Choice Bias Strength')
    
    # 3. Processing Symmetry
    ax = axes[0, 2]
    symmetry_categories = []
    for diff in successful_df['bilateral_bias_diff']:
        if diff < 0.05:
            symmetry_categories.append('Symmetric')
        elif diff < 0.15:
            symmetry_categories.append('Mildly Asymmetric')
        else:
            symmetry_categories.append('Strongly Asymmetric')
    
    sym_counts = pd.Series(symmetry_categories).value_counts()
    ax.pie(sym_counts.values, labels=sym_counts.index, autopct='%1.1f%%')
    ax.set_title('Processing Symmetry')
    
    # 4. Heatmap of Choice Patterns
    ax = axes[1, 0]
    choice_matrix = successful_df[['choice_0', 'choice_1', 'choice_2', 'choice_3']].T
    sns.heatmap(choice_matrix, annot=False, cmap='viridis', ax=ax)
    ax.set_title('Choice Patterns Across Subjects')
    ax.set_ylabel('Choice Type')
    ax.set_xlabel('Subjects')
    
    # 5. Correlation Matrix
    ax = axes[1, 1]
    corr_vars = ['accuracy', 'mean_rt', 'left_bias', 'right_bias', 'bilateral_bias_diff', 'choice_entropy']
    corr_matrix = successful_df[corr_vars].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Variable Correlations')
    
    # 6. Subject Clustering (by performance)
    ax = axes[1, 2]
    scatter = ax.scatter(successful_df['mean_rt'], successful_df['accuracy'], 
                        c=successful_df['bilateral_bias_diff'], cmap='plasma', alpha=0.7)
    ax.set_xlabel('Mean RT (seconds)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Subject Clustering')
    plt.colorbar(scatter, ax=ax, label='Bilateral Bias Diff')
    
    # 7. Choice Pattern Comparison
    ax = axes[2, 0]
    choice_means = [successful_df[f'choice_{i}'].mean() for i in range(4)]
    choice_stds = [successful_df[f'choice_{i}'].std() for i in range(4)]
    x_pos = range(4)
    ax.bar(x_pos, choice_means, yerr=choice_stds, capsize=5, alpha=0.7)
    ax.axhline(successful_df['n_trials'].mean()/4, color='red', linestyle='--', label='Expected')
    ax.set_xlabel('Choice Type')
    ax.set_ylabel('Mean Count')
    ax.set_title('Average Choice Usage')
    ax.legend()
    
    # 8. RT vs Bias Relationship
    ax = axes[2, 1]
    total_bias = successful_df['absolute_left_bias'] + successful_df['absolute_right_bias']
    ax.scatter(total_bias, successful_df['mean_rt'], alpha=0.7)
    ax.set_xlabel('Total Absolute Bias')
    ax.set_ylabel('Mean RT (seconds)')
    ax.set_title('Bias vs Processing Speed')
    
    # 9. Model Quality vs Performance
    ax = axes[2, 2]
    ax.scatter(successful_df['rhat_max'], successful_df['accuracy'], alpha=0.7)
    ax.axvline(1.05, color='red', linestyle='--', label='Good R-hat')
    ax.set_xlabel('Max R-hat')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Quality vs Performance')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('individual_differences_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_summary(data: Dict):
    """
    Print comprehensive summary of results
    """
    
    stats = data['summary_stats']
    df = data['results_df']
    
    print("="*80)
    print("COMPREHENSIVE LBA ANALYSIS SUMMARY - 18 PARTICIPANTS")
    print("="*80)
    
    print(f"\n📊 OVERALL SUCCESS METRICS:")
    print(f"   Total subjects analyzed: {stats['n_subjects']}")
    print(f"   Successful analyses: {stats['n_successful']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    
    if stats['n_successful'] > 0:
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   Mean accuracy: {stats['accuracy_mean']:.1%} (SD: {stats['accuracy_std']:.1%})")
        print(f"   Accuracy range: {stats['accuracy_range'][0]:.1%} - {stats['accuracy_range'][1]:.1%}")
        print(f"   Subjects below chance: {stats['accuracy_below_chance']}/{stats['n_successful']}")
        
        print(f"\n⏱️  REACTION TIME SUMMARY:")
        print(f"   Mean RT: {stats['rt_mean']:.3f}s (SD: {stats['rt_std']:.3f}s)")
        print(f"   RT range: {stats['rt_range'][0]:.3f}s - {stats['rt_range'][1]:.3f}s")
        
        print(f"\n🧠 COGNITIVE BIAS SUMMARY:")
        print(f"   Mean left bias: {stats['left_bias_mean']:+.3f}")
        print(f"   Mean right bias: {stats['right_bias_mean']:+.3f}")
        print(f"   Mean bilateral difference: {stats['bilateral_bias_diff_mean']:.3f}")
        print(f"   Strong left bias: {stats['strong_left_bias']}/{stats['n_successful']} subjects")
        print(f"   Strong right bias: {stats['strong_right_bias']}/{stats['n_successful']} subjects")
        print(f"   Asymmetric processing: {stats['asymmetric_processing']}/{stats['n_successful']} subjects")
        
        print(f"\n🔧 MODEL QUALITY SUMMARY:")
        print(f"   Mean R-hat: {stats['rhat_mean']:.3f}")
        print(f"   Mean ESS: {stats['ess_mean']:.0f}")
        print(f"   Mean sampling time: {stats['sampling_time_mean']:.1f} minutes")
        
        # Individual subject details
        successful_df = df[df['success'] == True]
        
        print(f"\n👥 INDIVIDUAL SUBJECT PATTERNS:")
        print("Subject | Accuracy | Mean RT | Left Bias | Right Bias | Asymmetry")
        print("-" * 70)
        
        for _, row in successful_df.iterrows():
            print(f"{row['subject_id']:7d} | {row['accuracy']:8.1%} | {row['mean_rt']:7.3f} | "
                  f"{row['left_bias']:9.3f} | {row['right_bias']:10.3f} | {row['bilateral_bias_diff']:9.3f}")

def main_analysis(results_file: str = 'dual_lba_results_20250615_122314.csv', 
                 original_file: str = 'GRT_LBA.csv'):
    """
    Main analysis function
    """
    
    print("🚀 Starting comprehensive LBA results analysis...")
    
    # Load and process data
    data = load_and_process_results(results_file, original_file)
    
    # Print summary
    print_detailed_summary(data)
    
    # Create visualizations
    print("\n📈 Creating overview visualizations...")
    create_overview_visualizations(data)
    
    print("\n🔍 Creating individual differences analysis...")
    create_individual_differences_analysis(data)
    
    print("\n✅ Analysis complete! Check the generated PNG files.")
    
    return data

if __name__ == "__main__":
    # Run the analysis
    results = main_analysis()
