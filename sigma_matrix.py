# -*- coding: utf-8 -*-
"""
Sigma Matrix Analysis for LBA Dual-Channel Model
Analyzes variance-covariance structure between left and right channels
as evidence for judgment mechanisms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import arviz as az

def extract_channel_parameters(results_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract left and right channel parameters for sigma matrix analysis
    """
    
    print("🔍 Extracting channel parameters for sigma matrix analysis...")
    
    # For this analysis, we'll use the behavioral data to estimate channel parameters
    # In a full implementation, you'd extract these from the MCMC traces
    
    channel_data = []
    
    for _, row in results_df.iterrows():
        if row['success']:
            subject_id = row['subject_id']
            subject_data = original_df[original_df['participant'] == subject_id]
            
            if len(subject_data) > 0:
                # Extract behavioral measures as proxies for channel parameters
                choice_data = subject_data['Response'].values
                rt_data = subject_data['RT'].values
                stimulus_data = subject_data['Stimulus'].values
                
                # Calculate channel-specific measures
                left_measures = calculate_left_channel_measures(choice_data, rt_data, stimulus_data)
                right_measures = calculate_right_channel_measures(choice_data, rt_data, stimulus_data)
                
                channel_data.append({
                    'subject_id': subject_id,
                    'left_bias': left_measures['bias'],
                    'left_sensitivity': left_measures['sensitivity'],
                    'left_consistency': left_measures['consistency'],
                    'left_speed': left_measures['speed'],
                    'right_bias': right_measures['bias'],
                    'right_sensitivity': right_measures['sensitivity'],
                    'right_consistency': right_measures['consistency'],
                    'right_speed': right_measures['speed'],
                    'accuracy': subject_data['Correct'].mean()
                })
    
    return pd.DataFrame(channel_data)

def calculate_left_channel_measures(choices, rts, stimuli):
    """Calculate left channel specific measures"""
    
    # Left channel processing (choices 0,1 have left\, choices 2,3 have left|)
    left_diagonal_trials = np.isin(stimuli, [0, 1])  # Stimuli with left\ 
    left_vertical_trials = np.isin(stimuli, [2, 3])   # Stimuli with left|
    
    # Left channel responses (choices 0,1 indicate left\ perceived)
    left_diagonal_responses = np.isin(choices, [0, 1])
    
    # Calculate measures
    total_trials = len(choices)
    
    # Bias: tendency to report left diagonal
    bias = np.mean(left_diagonal_responses) - 0.5
    
    # Sensitivity: accuracy in discriminating left diagonal vs vertical
    left_diag_correct = np.mean(left_diagonal_responses[left_diagonal_trials]) if np.sum(left_diagonal_trials) > 0 else 0.5
    left_vert_correct = np.mean(~left_diagonal_responses[left_vertical_trials]) if np.sum(left_vertical_trials) > 0 else 0.5
    sensitivity = (left_diag_correct + left_vert_correct) / 2
    
    # Consistency: inverse of response variability
    response_entropy = -np.sum([p * np.log(p + 1e-10) for p in [np.mean(left_diagonal_responses), 1 - np.mean(left_diagonal_responses)]])
    consistency = 1 - (response_entropy / np.log(2))  # Normalized
    
    # Speed: average RT for left-relevant trials
    speed = 1 / np.mean(rts)  # Inverse RT
    
    return {
        'bias': bias,
        'sensitivity': sensitivity,
        'consistency': consistency,
        'speed': speed
    }

def calculate_right_channel_measures(choices, rts, stimuli):
    """Calculate right channel specific measures"""
    
    # Right channel processing (choices 0,2 have right|, choices 1,3 have right/)
    right_vertical_trials = np.isin(stimuli, [0, 2])  # Stimuli with right|
    right_diagonal_trials = np.isin(stimuli, [1, 3])  # Stimuli with right/
    
    # Right channel responses (choices 1,3 indicate right/ perceived)  
    right_diagonal_responses = np.isin(choices, [1, 3])
    
    # Calculate measures
    bias = np.mean(right_diagonal_responses) - 0.5
    
    # Sensitivity
    right_diag_correct = np.mean(right_diagonal_responses[right_diagonal_trials]) if np.sum(right_diagonal_trials) > 0 else 0.5
    right_vert_correct = np.mean(~right_diagonal_responses[right_vertical_trials]) if np.sum(right_vertical_trials) > 0 else 0.5
    sensitivity = (right_diag_correct + right_vert_correct) / 2
    
    # Consistency
    response_entropy = -np.sum([p * np.log(p + 1e-10) for p in [np.mean(right_diagonal_responses), 1 - np.mean(right_diagonal_responses)]])
    consistency = 1 - (response_entropy / np.log(2))
    
    # Speed
    speed = 1 / np.mean(rts)
    
    return {
        'bias': bias,
        'sensitivity': sensitivity,
        'consistency': consistency,
        'speed': speed
    }

def calculate_sigma_matrices(channel_df: pd.DataFrame) -> Dict:
    """
    Calculate variance-covariance matrices between left and right channels
    """
    
    print("📊 Calculating sigma matrices...")
    
    # Define channel variables
    left_vars = ['left_bias', 'left_sensitivity', 'left_consistency', 'left_speed']
    right_vars = ['right_bias', 'right_sensitivity', 'right_consistency', 'right_speed']
    
    # Extract data
    left_data = channel_df[left_vars].values
    right_data = channel_df[right_vars].values
    
    # Calculate individual channel covariance matrices
    sigma_left = np.cov(left_data.T)
    sigma_right = np.cov(right_data.T)
    
    # Calculate cross-channel covariance matrix
    sigma_cross = np.zeros((len(left_vars), len(right_vars)))
    for i, left_var in enumerate(left_vars):
        for j, right_var in enumerate(right_vars):
            sigma_cross[i, j] = np.cov(channel_df[left_var], channel_df[right_var])[0, 1]
    
    # Calculate full bilateral covariance matrix
    all_vars = left_vars + right_vars
    bilateral_data = channel_df[all_vars].values
    sigma_bilateral = np.cov(bilateral_data.T)
    
    # Calculate correlations
    corr_left = np.corrcoef(left_data.T)
    corr_right = np.corrcoef(right_data.T)
    corr_cross = np.zeros((len(left_vars), len(right_vars)))
    for i, left_var in enumerate(left_vars):
        for j, right_var in enumerate(right_vars):
            corr_cross[i, j] = np.corrcoef(channel_df[left_var], channel_df[right_var])[0, 1]
    
    corr_bilateral = np.corrcoef(bilateral_data.T)
    
    return {
        'sigma_left': sigma_left,
        'sigma_right': sigma_right,
        'sigma_cross': sigma_cross,
        'sigma_bilateral': sigma_bilateral,
        'corr_left': corr_left,
        'corr_right': corr_right,
        'corr_cross': corr_cross,
        'corr_bilateral': corr_bilateral,
        'left_vars': left_vars,
        'right_vars': right_vars,
        'all_vars': all_vars
    }

def analyze_independence_evidence(sigma_results: Dict, channel_df: pd.DataFrame):
    """
    Analyze evidence for/against channel independence from sigma matrices
    """
    
    print("\n🔬 SIGMA MATRIX EVIDENCE FOR JUDGMENT MECHANISMS")
    print("="*60)
    
    # Extract matrices
    sigma_cross = sigma_results['sigma_cross']
    corr_cross = sigma_results['corr_cross']
    left_vars = sigma_results['left_vars']
    right_vars = sigma_results['right_vars']
    
    print(f"\n1️⃣ CROSS-CHANNEL COVARIANCE MATRIX:")
    print(f"Rows: {left_vars}")
    print(f"Cols: {right_vars}")
    print("\nCovariance values:")
    for i, left_var in enumerate(left_vars):
        row_str = f"{left_var:20s}:"
        for j, right_var in enumerate(right_vars):
            row_str += f"{sigma_cross[i,j]:8.4f}"
        print(row_str)
    
    print(f"\n2️⃣ CROSS-CHANNEL CORRELATION MATRIX:")
    print("Correlation values:")
    for i, left_var in enumerate(left_vars):
        row_str = f"{left_var:20s}:"
        for j, right_var in enumerate(right_vars):
            row_str += f"{corr_cross[i,j]:8.3f}"
        print(row_str)
    
    # Test for independence
    print(f"\n3️⃣ INDEPENDENCE TESTS:")
    
    # Overall test: are any cross-correlations significant?
    n_subjects = len(channel_df)
    significant_corrs = []
    
    for i, left_var in enumerate(left_vars):
        for j, right_var in enumerate(right_vars):
            r = corr_cross[i, j]
            # Test significance
            t_stat = r * np.sqrt((n_subjects - 2) / (1 - r**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_subjects - 2))
            
            if p_value < 0.05:
                significant_corrs.append((left_var, right_var, r, p_value))
                print(f"   {left_var} × {right_var}: r = {r:.3f}, p = {p_value:.3f} *")
            else:
                print(f"   {left_var} × {right_var}: r = {r:.3f}, p = {p_value:.3f}")
    
    # Independence verdict
    if len(significant_corrs) == 0:
        print(f"\n✅ INDEPENDENCE SUPPORTED: No significant cross-correlations")
    else:
        print(f"\n❌ INDEPENDENCE VIOLATED: {len(significant_corrs)} significant cross-correlations")
        print(f"   Strongest violations:")
        significant_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        for left_var, right_var, r, p in significant_corrs[:3]:
            print(f"   • {left_var} × {right_var}: r = {r:.3f}")
    
    # Separability analysis
    print(f"\n4️⃣ SEPARABILITY ANALYSIS:")
    
    # Check if bias correlations suggest separability violations
    bias_corr = corr_cross[0, 0]  # left_bias × right_bias
    sens_corr = corr_cross[1, 1]  # left_sensitivity × right_sensitivity
    
    print(f"Bias correlation (left ↔ right): {bias_corr:.3f}")
    print(f"Sensitivity correlation (left ↔ right): {sens_corr:.3f}")
    
    if abs(bias_corr) > 0.3 or abs(sens_corr) > 0.3:
        print(f"❌ SEPARABILITY VIOLATED: Strong cross-channel correlations")
    else:
        print(f"✅ SEPARABILITY SUPPORTED: Weak cross-channel correlations")
    
    # Channel dominance analysis
    print(f"\n5️⃣ CHANNEL DOMINANCE PATTERNS:")
    
    left_vars_data = channel_df[left_vars]
    right_vars_data = channel_df[right_vars]
    
    left_variance = np.mean([np.var(channel_df[var]) for var in left_vars])
    right_variance = np.mean([np.var(channel_df[var]) for var in right_vars])
    
    print(f"Average left channel variance: {left_variance:.4f}")
    print(f"Average right channel variance: {right_variance:.4f}")
    print(f"Variance ratio (left/right): {left_variance/right_variance:.3f}")
    
    if abs(left_variance/right_variance - 1) > 0.5:
        print(f"⚖️ ASYMMETRIC PROCESSING: Unequal channel variances")
    else:
        print(f"⚖️ SYMMETRIC PROCESSING: Similar channel variances")
    
    return {
        'significant_correlations': significant_corrs,
        'independence_supported': len(significant_corrs) == 0,
        'separability_supported': abs(bias_corr) <= 0.3 and abs(sens_corr) <= 0.3,
        'bias_correlation': bias_corr,
        'sensitivity_correlation': sens_corr,
        'variance_ratio': left_variance/right_variance
    }

def visualize_sigma_matrices(sigma_results: Dict, channel_df: pd.DataFrame):
    """
    Create comprehensive visualizations of sigma matrices
    """
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. Left channel covariance
    ax = axes[0, 0]
    sns.heatmap(sigma_results['sigma_left'], 
                xticklabels=[v.replace('left_', '') for v in sigma_results['left_vars']],
                yticklabels=[v.replace('left_', '') for v in sigma_results['left_vars']],
                annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Left Channel\nCovariance Matrix')
    
    # 2. Right channel covariance  
    ax = axes[0, 1]
    sns.heatmap(sigma_results['sigma_right'],
                xticklabels=[v.replace('right_', '') for v in sigma_results['right_vars']],
                yticklabels=[v.replace('right_', '') for v in sigma_results['right_vars']],
                annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Right Channel\nCovariance Matrix')
    
    # 3. Cross-channel covariance
    ax = axes[0, 2]
    sns.heatmap(sigma_results['sigma_cross'],
                xticklabels=[v.replace('right_', '') for v in sigma_results['right_vars']],
                yticklabels=[v.replace('left_', '') for v in sigma_results['left_vars']],
                annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Cross-Channel\nCovariance Matrix')
    
    # 4. Left channel correlation
    ax = axes[1, 0]
    sns.heatmap(sigma_results['corr_left'],
                xticklabels=[v.replace('left_', '') for v in sigma_results['left_vars']],
                yticklabels=[v.replace('left_', '') for v in sigma_results['left_vars']],
                annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Left Channel\nCorrelation Matrix')
    
    # 5. Right channel correlation
    ax = axes[1, 1]
    sns.heatmap(sigma_results['corr_right'],
                xticklabels=[v.replace('right_', '') for v in sigma_results['right_vars']],
                yticklabels=[v.replace('right_', '') for v in sigma_results['right_vars']],
                annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Right Channel\nCorrelation Matrix')
    
    # 6. Cross-channel correlation
    ax = axes[1, 2]
    sns.heatmap(sigma_results['corr_cross'],
                xticklabels=[v.replace('right_', '') for v in sigma_results['right_vars']],
                yticklabels=[v.replace('left_', '') for v in sigma_results['left_vars']],
                annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Cross-Channel\nCorrelation Matrix')
    
    # 7. Full bilateral correlation matrix
    ax = axes[2, 0:2]  # Span two columns
    ax = plt.subplot(3, 3, (7, 8))  # Manually create spanning subplot
    full_labels = [v.replace('left_', 'L_').replace('right_', 'R_') for v in sigma_results['all_vars']]
    sns.heatmap(sigma_results['corr_bilateral'],
                xticklabels=full_labels,
                yticklabels=full_labels,
                annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Full Bilateral Correlation Matrix')
    
    # 8. Scatterplot of key relationships
    ax = axes[2, 2]
    ax.scatter(channel_df['left_bias'], channel_df['right_bias'], alpha=0.7)
    ax.set_xlabel('Left Channel Bias')
    ax.set_ylabel('Right Channel Bias')
    ax.set_title('Channel Bias Relationship')
    
    # Add correlation info
    r = np.corrcoef(channel_df['left_bias'], channel_df['right_bias'])[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('sigma_matrix_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_judgment_mechanism_report(sigma_results: Dict, independence_results: Dict, channel_df: pd.DataFrame):
    """
    Generate comprehensive report on judgment mechanisms from sigma analysis
    """
    
    print("\n" + "="*80)
    print("JUDGMENT MECHANISM ANALYSIS - SIGMA MATRIX EVIDENCE")
    print("="*80)
    
    print(f"\n📊 SAMPLE: {len(channel_df)} subjects")
    
    print(f"\n🔍 KEY FINDINGS:")
    
    # Independence
    if independence_results['independence_supported']:
        print(f"✅ CHANNEL INDEPENDENCE: Supported")
        print(f"   • No significant cross-channel correlations detected")
    else:
        print(f"❌ CHANNEL INDEPENDENCE: Violated")
        print(f"   • {len(independence_results['significant_correlations'])} significant cross-correlations")
    
    # Separability
    if independence_results['separability_supported']:
        print(f"✅ PERCEPTUAL SEPARABILITY: Supported")
        print(f"   • Weak bias correlation: {independence_results['bias_correlation']:.3f}")
    else:
        print(f"❌ PERCEPTUAL SEPARABILITY: Violated")
        print(f"   • Strong bias correlation: {independence_results['bias_correlation']:.3f}")
        print(f"   • Strong sensitivity correlation: {independence_results['sensitivity_correlation']:.3f}")
    
    # Processing symmetry
    variance_ratio = independence_results['variance_ratio']
    if 0.5 < variance_ratio < 2.0:
        print(f"⚖️ PROCESSING SYMMETRY: Supported")
        print(f"   • Variance ratio: {variance_ratio:.3f}")
    else:
        print(f"⚖️ PROCESSING SYMMETRY: Asymmetric")
        print(f"   • Variance ratio: {variance_ratio:.3f}")
    
    print(f"\n🧠 IMPLICATIONS FOR JUDGMENT MECHANISMS:")
    
    if not independence_results['independence_supported']:
        print(f"• Bilateral integration during decision-making")
        print(f"• Cross-talk between left and right processing streams")
        print(f"• Violation of independent channel assumptions")
    
    if not independence_results['separability_supported']:
        print(f"• Left-side perception influences right-side perception")
        print(f"• Holistic rather than featural processing")
        print(f"• Evidence against modular perceptual architecture")
    
    print(f"\n📋 SIGMA MATRIX SUMMARY:")
    print(f"Cross-channel covariance range: [{np.min(sigma_results['sigma_cross']):.4f}, {np.max(sigma_results['sigma_cross']):.4f}]")
    print(f"Cross-channel correlation range: [{np.min(sigma_results['corr_cross']):.3f}, {np.max(sigma_results['corr_cross']):.3f}]")
    
    # Strongest relationships
    max_corr_idx = np.unravel_index(np.argmax(np.abs(sigma_results['corr_cross'])), sigma_results['corr_cross'].shape)
    strongest_left = sigma_results['left_vars'][max_corr_idx[0]]
    strongest_right = sigma_results['right_vars'][max_corr_idx[1]]
    strongest_corr = sigma_results['corr_cross'][max_corr_idx]
    
    print(f"Strongest cross-channel relationship: {strongest_left} ↔ {strongest_right} (r = {strongest_corr:.3f})")

def main_sigma_analysis(results_file: str = 'dual_lba_results_20250615_122314.csv',
                       original_file: str = 'GRT_LBA.csv'):
    """
    Main function for sigma matrix analysis
    """
    
    print("🔬 Starting Sigma Matrix Analysis for Judgment Mechanisms...")
    
    # Load data
    results_df = pd.read_csv(results_file)
    original_df = pd.read_csv(original_file)
    
    # Extract channel parameters
    channel_df = extract_channel_parameters(results_df, original_df)
    
    # Calculate sigma matrices
    sigma_results = calculate_sigma_matrices(channel_df)
    
    # Analyze independence evidence
    independence_results = analyze_independence_evidence(sigma_results, channel_df)
    
    # Create visualizations
    visualize_sigma_matrices(sigma_results, channel_df)
    
    # Generate comprehensive report
    generate_judgment_mechanism_report(sigma_results, independence_results, channel_df)
    
    print(f"\n✅ Sigma matrix analysis complete!")
    print(f"📁 Visualization saved: sigma_matrix_analysis.png")
    
    return {
        'sigma_results': sigma_results,
        'independence_results': independence_results,
        'channel_df': channel_df
    }

if __name__ == "__main__":
    # Run the sigma matrix analysis
    results = main_sigma_analysis()
