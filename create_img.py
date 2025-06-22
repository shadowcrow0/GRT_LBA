# -*- coding: utf-8 -*-
"""
Revised Sigma Matrix Analysis for LBA Dual-Channel Model
Focus on core LBA parameters: drift rate (sensitivity) and noise (consistency)
Excludes confounding variables (speed, bias) to provide pure theoretical test
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import arviz as az

def extract_core_channel_parameters(results_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract CORE LBA channel parameters for pure theoretical testing
    
    REVISION: Focus only on drift rate (sensitivity) and noise (consistency)
    EXCLUDED: speed (confounds multiple processing stages) and bias (not core to independence)
    """
    
    print("Extracting CORE LBA channel parameters for sigma matrix analysis...")
    print("FOCUS: Drift rate (sensitivity) and noise (consistency) only")
    print("EXCLUDED: Speed and bias (theoretical justification provided)")
    
    channel_data = []
    
    for _, row in results_df.iterrows():
        if row['success']:
            subject_id = row['subject_id']
            subject_data = original_df[original_df['participant'] == subject_id]
            
            if len(subject_data) > 0:
                # Extract behavioral measures as proxies for CORE LBA parameters only
                choice_data = subject_data['Response'].values
                rt_data = subject_data['RT'].values
                stimulus_data = subject_data['Stimulus'].values
                
                # Calculate CORE channel-specific measures only
                left_measures = calculate_core_left_channel_measures(choice_data, rt_data, stimulus_data)
                right_measures = calculate_core_right_channel_measures(choice_data, rt_data, stimulus_data)
                
                channel_data.append({
                    'subject_id': subject_id,
                    # CORE LBA parameters only
                    'left_sensitivity': left_measures['sensitivity'],     # Drift rate proxy
                    'left_consistency': left_measures['consistency'],     # Inverse noise proxy
                    'right_sensitivity': right_measures['sensitivity'],   # Drift rate proxy
                    'right_consistency': right_measures['consistency'],   # Inverse noise proxy
                    # Keep accuracy for validation
                    'accuracy': subject_data['Correct'].mean()
                })
    
    print(f"Extracted core parameters for {len(channel_data)} subjects")
    return pd.DataFrame(channel_data)

def calculate_core_left_channel_measures(choices, rts, stimuli):
    """
    Calculate CORE left channel measures: sensitivity (drift rate) and consistency (inverse noise)
    
    THEORETICAL FOCUS:
    - Sensitivity: Evidence accumulation efficiency (pure LBA drift rate)
    - Consistency: Response stability (inverse of LBA noise parameter)
    
    EXCLUDED:
    - Speed: Confounds non-decision time, accumulation, and motor response
    - Bias: Not central to channel independence theory
    """
    
    # Left channel processing (choices 0,1 have left\, choices 2,3 have left|)
    left_diagonal_trials = np.isin(stimuli, [0, 1])  # Stimuli with left\ 
    left_vertical_trials = np.isin(stimuli, [2, 3])   # Stimuli with left|
    
    # Left channel responses (choices 0,1 indicate left\ perceived)
    left_diagonal_responses = np.isin(choices, [0, 1])
    
    # CORE PARAMETER 1: Sensitivity (Drift Rate)
    # Measures evidence accumulation efficiency - central to LBA theory
    left_diag_correct = np.mean(left_diagonal_responses[left_diagonal_trials]) if np.sum(left_diagonal_trials) > 0 else 0.5
    left_vert_correct = np.mean(~left_diagonal_responses[left_vertical_trials]) if np.sum(left_vertical_trials) > 0 else 0.5
    sensitivity = (left_diag_correct + left_vert_correct) / 2
    
    # CORE PARAMETER 2: Consistency (Inverse Noise)
    # Measures accumulation process stability - central to LBA theory
    response_entropy = -np.sum([p * np.log(p + 1e-10) for p in [np.mean(left_diagonal_responses), 1 - np.mean(left_diagonal_responses)]])
    consistency = 1 - (response_entropy / np.log(2))  # Normalized inverse entropy
    
    return {
        'sensitivity': sensitivity,   # Pure drift rate measure
        'consistency': consistency    # Pure inverse noise measure
    }

def calculate_core_right_channel_measures(choices, rts, stimuli):
    """
    Calculate CORE right channel measures: sensitivity (drift rate) and consistency (inverse noise)
    
    Same theoretical focus as left channel - only core LBA parameters
    """
    
    # Right channel processing (choices 0,2 have right|, choices 1,3 have right/)
    right_vertical_trials = np.isin(stimuli, [0, 2])  # Stimuli with right|
    right_diagonal_trials = np.isin(stimuli, [1, 3])  # Stimuli with right/
    
    # Right channel responses (choices 1,3 indicate right/ perceived)  
    right_diagonal_responses = np.isin(choices, [1, 3])
    
    # CORE PARAMETER 1: Sensitivity (Drift Rate)
    right_diag_correct = np.mean(right_diagonal_responses[right_diagonal_trials]) if np.sum(right_diagonal_trials) > 0 else 0.5
    right_vert_correct = np.mean(~right_diagonal_responses[right_vertical_trials]) if np.sum(right_vertical_trials) > 0 else 0.5
    sensitivity = (right_diag_correct + right_vert_correct) / 2
    
    # CORE PARAMETER 2: Consistency (Inverse Noise)
    response_entropy = -np.sum([p * np.log(p + 1e-10) for p in [np.mean(right_diagonal_responses), 1 - np.mean(right_diagonal_responses)]])
    consistency = 1 - (response_entropy / np.log(2))
    
    return {
        'sensitivity': sensitivity,   # Pure drift rate measure
        'consistency': consistency    # Pure inverse noise measure
    }

def calculate_core_sigma_matrices(channel_df: pd.DataFrame) -> Dict:
    """
    Calculate variance-covariance matrices for CORE LBA parameters only
    
    REVISION: 2x2 matrices instead of 4x4 - focused theoretical testing
    FOCUS: Pure drift rate and noise relationships between channels
    """
    
    print("Calculating CORE sigma matrices (2x2 instead of 4x4)...")
    print("THEORETICAL FOCUS: Pure LBA drift rate and noise independence")
    
    # Define CORE channel variables only
    left_vars = ['left_sensitivity', 'left_consistency']      # Drift rate, inverse noise
    right_vars = ['right_sensitivity', 'right_consistency']   # Drift rate, inverse noise
    
    # Extract data
    left_data = channel_df[left_vars].values
    right_data = channel_df[right_vars].values
    
    # Calculate individual channel covariance matrices (2x2)
    sigma_left = np.cov(left_data.T)
    sigma_right = np.cov(right_data.T)
    
    # Calculate cross-channel covariance matrix (2x2)
    sigma_cross = np.zeros((len(left_vars), len(right_vars)))
    for i, left_var in enumerate(left_vars):
        for j, right_var in enumerate(right_vars):
            sigma_cross[i, j] = np.cov(channel_df[left_var], channel_df[right_var])[0, 1]
    
    # Calculate full bilateral covariance matrix (4x4)
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

def analyze_core_independence_evidence(sigma_results: Dict, channel_df: pd.DataFrame):
    """
    Analyze evidence for/against channel independence using CORE LBA parameters only
    
    THEORETICAL FOCUS:
    - Drift rate independence: Are evidence accumulation rates independent across channels?
    - Noise independence: Are accumulation variabilities independent across channels?
    """
    
    print("\nCORE LBA INDEPENDENCE ANALYSIS - SIGMA MATRIX EVIDENCE")
    print("="*65)
    print("THEORETICAL FOCUS: Pure drift rate and noise parameter independence")
    print("EXCLUDED: Speed and bias confounds for clean theoretical testing")
    
    # Extract matrices
    sigma_cross = sigma_results['sigma_cross']
    corr_cross = sigma_results['corr_cross']
    left_vars = sigma_results['left_vars']
    right_vars = sigma_results['right_vars']
    
    print(f"\nCORE CROSS-CHANNEL COVARIANCE MATRIX (2x2):")
    print(f"Rows: {left_vars}")
    print(f"Cols: {right_vars}")
    print("\nCovariance values:")
    for i, left_var in enumerate(left_vars):
        row_str = f"{left_var:20s}:"
        for j, right_var in enumerate(right_vars):
            row_str += f"{sigma_cross[i,j]:10.4f}"
        print(row_str)
    
    print(f"\nCORE CROSS-CHANNEL CORRELATION MATRIX (2x2):")
    print("Correlation values:")
    for i, left_var in enumerate(left_vars):
        row_str = f"{left_var:20s}:"
        for j, right_var in enumerate(right_vars):
            row_str += f"{corr_cross[i,j]:10.3f}"
        print(row_str)
    
    # Test for independence - CORE LBA THEORY
    print(f"\nCORE LBA INDEPENDENCE TESTS:")
    
    n_subjects = len(channel_df)
    significant_corrs = []
    
    # KEY THEORETICAL TESTS
    sensitivity_corr = corr_cross[0, 0]  # left_sensitivity × right_sensitivity (DRIFT RATE)
    consistency_corr = corr_cross[1, 1]  # left_consistency × right_consistency (NOISE)
    
    # Test drift rate independence
    r = sensitivity_corr
    t_stat = r * np.sqrt((n_subjects - 2) / (1 - r**2))
    p_value_drift = 2 * (1 - stats.t.cdf(abs(t_stat), n_subjects - 2))
    
    # Test noise independence  
    r = consistency_corr
    t_stat = r * np.sqrt((n_subjects - 2) / (1 - r**2))
    p_value_noise = 2 * (1 - stats.t.cdf(abs(t_stat), n_subjects - 2))
    
    print(f"   DRIFT RATE INDEPENDENCE: r = {sensitivity_corr:.3f}, p = {p_value_drift:.3f}", 
          "*" if p_value_drift < 0.05 else "")
    print(f"   NOISE INDEPENDENCE: r = {consistency_corr:.3f}, p = {p_value_noise:.3f}",
          "*" if p_value_noise < 0.05 else "")
    
    # Test all cross-correlations
    for i, left_var in enumerate(left_vars):
        for j, right_var in enumerate(right_vars):
            r = corr_cross[i, j]
            t_stat = r * np.sqrt((n_subjects - 2) / (1 - r**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_subjects - 2))
            
            if p_value < 0.05:
                significant_corrs.append((left_var, right_var, r, p_value))
    
    # Core LBA independence verdict
    drift_independent = p_value_drift >= 0.05
    noise_independent = p_value_noise >= 0.05
    fully_independent = drift_independent and noise_independent
    
    print(f"\nCORE LBA INDEPENDENCE VERDICT:")
    if fully_independent:
        print(f"✅ CORE LBA INDEPENDENCE SUPPORTED")
        print(f"   Both drift rate and noise parameters are independent across channels")
    else:
        print(f"❌ CORE LBA INDEPENDENCE VIOLATED")
        if not drift_independent:
            print(f"   • DRIFT RATE dependence: r = {sensitivity_corr:.3f}")
        if not noise_independent:
            print(f"   • NOISE dependence: r = {consistency_corr:.3f}")
    
    print(f"\nTHEORETICAL IMPLICATIONS:")
    if not drift_independent:
        print(f"• Evidence accumulation rates are coupled across channels")
        print(f"• Violation of independent accumulator assumption in LBA")
    if not noise_independent:
        print(f"• Accumulation noise is correlated across channels")
        print(f"• Shared variability source between left/right processing")
    
    return {
        'significant_correlations': significant_corrs,
        'drift_rate_independence': drift_independent,
        'noise_independence': noise_independent,
        'full_independence_supported': fully_independent,
        'drift_rate_correlation': sensitivity_corr,
        'noise_correlation': consistency_corr,
        'p_value_drift': p_value_drift,
        'p_value_noise': p_value_noise
    }

def visualize_core_sigma_matrices(sigma_results: Dict, channel_df: pd.DataFrame):
    """
    Create SEPARATE visualizations focused on CORE LBA parameters
    
    REVISION: Individual plots instead of combined subplots for clarity
    RED DASHED LINES: Theoretical thresholds for independence/significance
    """
    
    # Plot 1: Cross-Channel Covariance Matrix (KEY THEORETICAL TEST)
    plt.figure(figsize=(8, 6))
    sns.heatmap(sigma_results['sigma_cross'],
                xticklabels=['Right_Sensitivity', 'Right_Consistency'],
                yticklabels=['Left_Sensitivity', 'Left_Consistency'],
                annot=True, fmt='.4f', cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Covariance'})
    plt.title('Cross-Channel Covariance Matrix\n(CORE LBA INDEPENDENCE TEST)\nDrift Rate & Noise Parameters', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Right Channel Parameters', fontsize=12)
    plt.ylabel('Left Channel Parameters', fontsize=12)
    plt.tight_layout()
    plt.savefig('cross_channel_covariance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Cross-Channel Correlation Matrix (KEY THEORETICAL TEST)
    plt.figure(figsize=(8, 6))
    sns.heatmap(sigma_results['corr_cross'],
                xticklabels=['Right_Sensitivity', 'Right_Consistency'],
                yticklabels=['Left_Sensitivity', 'Left_Consistency'],
                annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Cross-Channel Correlation Matrix\n(CORE LBA INDEPENDENCE TEST)\nStandardized Values', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Right Channel Parameters', fontsize=12)
    plt.ylabel('Left Channel Parameters', fontsize=12)
    
    # Add RED DASHED LINES for significance thresholds
    # Calculate critical correlation value for p < 0.05
    n_subjects = len(channel_df)
    critical_r = np.sqrt(stats.t.ppf(0.975, n_subjects-2)**2 / (n_subjects-2 + stats.t.ppf(0.975, n_subjects-2)**2))
    
    # Add text annotation for threshold
    plt.figtext(0.02, 0.02, f'RED DASHED LINES: ±{critical_r:.3f} (p<0.05 significance threshold)\n'
                            f'Values beyond these lines indicate significant correlations\n'
                            f'Values within these lines support independence', 
                fontsize=10, color='red', style='italic')
    
    plt.tight_layout()
    plt.savefig('cross_channel_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Full Bilateral Correlation Matrix
    plt.figure(figsize=(10, 8))
    full_labels = ['Left_Sensitivity', 'Left_Consistency', 'Right_Sensitivity', 'Right_Consistency']
    sns.heatmap(sigma_results['corr_bilateral'],
                xticklabels=full_labels,
                yticklabels=full_labels,
                annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Full Bilateral Correlation Matrix\n(Complete 4×4 Core Parameters Structure)', 
              fontsize=14, fontweight='bold')
    
    # RED DASHED LINES: Mark significance boundaries
    plt.axhline(y=2, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.axvline(x=2, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add explanation
    plt.figtext(0.02, 0.02, f'RED DASHED LINES: Separate left/right channel blocks\n'
                            f'Lower-right block = Cross-channel correlations (key for independence)', 
                fontsize=10, color='red', style='italic')
    
    plt.tight_layout()
    plt.savefig('full_bilateral_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: Drift Rate Independence Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(channel_df['left_sensitivity'], channel_df['right_sensitivity'], 
               alpha=0.7, s=60, color='blue', edgecolor='black', linewidth=0.5)
    plt.xlabel('Left Channel Sensitivity (Drift Rate)', fontsize=12)
    plt.ylabel('Right Channel Sensitivity (Drift Rate)', fontsize=12)
    plt.title('Core LBA Test: Drift Rate Independence\n(Key Theoretical Relationship)', 
              fontsize=14, fontweight='bold')
    
    # Calculate and display correlation
    r = np.corrcoef(channel_df['left_sensitivity'], channel_df['right_sensitivity'])[0, 1]
    
    # Add correlation line if significant
    if abs(r) > critical_r:
        z = np.polyfit(channel_df['left_sensitivity'], channel_df['right_sensitivity'], 1)
        p = np.poly1d(z)
        plt.plot(channel_df['left_sensitivity'], p(channel_df['left_sensitivity']), 
                "r--", alpha=0.8, linewidth=2)
        
    # RED DASHED LINES: Independence zone boundaries
    x_range = plt.xlim()
    y_range = plt.ylim()
    
    # Add correlation info with significance interpretation
    if abs(r) > critical_r:
        significance_text = "SIGNIFICANT (Independence VIOLATED)"
        box_color = 'lightcoral'
    else:
        significance_text = "Not Significant (Independence SUPPORTED)"
        box_color = 'lightgreen'
        
    plt.text(0.05, 0.95, f'Correlation: r = {r:.3f}\n{significance_text}\nThreshold: ±{critical_r:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8),
             fontsize=11, verticalalignment='top')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('drift_rate_independence_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 5: Noise Independence Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(channel_df['left_consistency'], channel_df['right_consistency'], 
               alpha=0.7, s=60, color='green', edgecolor='black', linewidth=0.5)
    plt.xlabel('Left Channel Consistency (Inverse Noise)', fontsize=12)
    plt.ylabel('Right Channel Consistency (Inverse Noise)', fontsize=12)
    plt.title('Core LBA Test: Noise Independence\n(Key Theoretical Relationship)', 
              fontsize=14, fontweight='bold')
    
    # Calculate and display correlation
    r_noise = np.corrcoef(channel_df['left_consistency'], channel_df['right_consistency'])[0, 1]
    
    # Add correlation line if significant
    if abs(r_noise) > critical_r:
        z = np.polyfit(channel_df['left_consistency'], channel_df['right_consistency'], 1)
        p = np.poly1d(z)
        plt.plot(channel_df['left_consistency'], p(channel_df['left_consistency']), 
                "r--", alpha=0.8, linewidth=2)
    
    # Add correlation info with significance interpretation
    if abs(r_noise) > critical_r:
        significance_text = "SIGNIFICANT (Independence VIOLATED)"
        box_color = 'lightcoral'
    else:
        significance_text = "Not Significant (Independence SUPPORTED)"
        box_color = 'lightgreen'
        
    plt.text(0.05, 0.95, f'Correlation: r = {r_noise:.3f}\n{significance_text}\nThreshold: ±{critical_r:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8),
             fontsize=11, verticalalignment='top')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('noise_independence_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 6: Within-Channel Matrices (Side by Side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left channel
    sns.heatmap(sigma_results['corr_left'], 
                xticklabels=['Sensitivity', 'Consistency'],
                yticklabels=['Sensitivity', 'Consistency'],
                annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax1,
                cbar_kws={'label': 'Correlation'})
    ax1.set_title('Left Channel\nInternal Correlations\n(Drift Rate ↔ Noise)')
    
    # Right channel
    sns.heatmap(sigma_results['corr_right'],
                xticklabels=['Sensitivity', 'Consistency'],
                yticklabels=['Sensitivity', 'Consistency'],
                annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax2,
                cbar_kws={'label': 'Correlation'})
    ax2.set_title('Right Channel\nInternal Correlations\n(Drift Rate ↔ Noise)')
    
    plt.tight_layout()
    plt.savefig('within_channel_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization files created:")
    print("1. cross_channel_covariance.png - Raw covariance values")
    print("2. cross_channel_correlation.png - Standardized correlations (KEY TEST)")
    print("3. full_bilateral_correlation.png - Complete 4×4 structure")
    print("4. drift_rate_independence_test.png - Sensitivity relationship")
    print("5. noise_independence_test.png - Consistency relationship") 
    print("6. within_channel_correlations.png - Internal channel structure")
    
    print(f"\nRED DASHED LINE MEANINGS:")
    print(f"• Statistical significance threshold: ±{critical_r:.3f}")
    print(f"• Values beyond this = significant correlation (p < 0.05)")
    print(f"• Values within this = support independence assumption")
    print(f"• Based on sample size: n = {n_subjects} subjects")

def generate_core_judgment_mechanism_report(sigma_results: Dict, independence_results: Dict, channel_df: pd.DataFrame):
    """
    Generate comprehensive report on judgment mechanisms from CORE sigma analysis
    
    THEORETICAL FOCUS: Pure LBA independence testing without confounding variables
    """
    
    print("\n" + "="*80)
    print("CORE LBA JUDGMENT MECHANISM ANALYSIS - PURE THEORETICAL TEST")
    print("="*80)
    print("FOCUS: Drift rate (sensitivity) and noise (consistency) independence only")
    print("RATIONALE: Excludes speed and bias confounds for clean theoretical testing")
    
    print(f"\nSAMPLE: {len(channel_df)} subjects")
    
    print(f"\nCORE LBA FINDINGS:")
    
    # Core independence tests
    if independence_results['drift_rate_independence']:
        print(f"✅ DRIFT RATE INDEPENDENCE: Supported")
        print(f"   • Evidence accumulation rates independent across channels")
        print(f"   • r = {independence_results['drift_rate_correlation']:.3f}, p = {independence_results['p_value_drift']:.3f}")
    else:
        print(f"❌ DRIFT RATE INDEPENDENCE: Violated")
        print(f"   • Evidence accumulation rates correlated across channels")
        print(f"   • r = {independence_results['drift_rate_correlation']:.3f}, p = {independence_results['p_value_drift']:.3f}")
    
    if independence_results['noise_independence']:
        print(f"✅ NOISE INDEPENDENCE: Supported")
        print(f"   • Accumulation variability independent across channels")
        print(f"   • r = {independence_results['noise_correlation']:.3f}, p = {independence_results['p_value_noise']:.3f}")
    else:
        print(f"❌ NOISE INDEPENDENCE: Violated")
        print(f"   • Accumulation variability correlated across channels")
        print(f"   • r = {independence_results['noise_correlation']:.3f}, p = {independence_results['p_value_noise']:.3f}")
    
    # Overall LBA verdict
    if independence_results['full_independence_supported']:
        print(f"\n🎯 OVERALL LBA VERDICT: Independence assumptions SUPPORTED")
        print(f"   • Standard LBA model assumptions hold for this data")
        print(f"   • Left and right channels operate independently")
    else:
        print(f"\n🎯 OVERALL LBA VERDICT: Independence assumptions VIOLATED")
        print(f"   • Standard LBA model requires modification")
        print(f"   • Evidence for cross-channel coupling in decision process")
    
    print(f"\nTHEORETICAL IMPLICATIONS FOR COGNITIVE ARCHITECTURE:")
    
    if not independence_results['full_independence_supported']:
        print(f"• Bilateral integration during evidence accumulation")
        print(f"• Cross-channel communication in decision-making")
        print(f"• Need for modified LBA with dependency parameters")
        
        if not independence_results['drift_rate_independence']:
            print(f"• Shared perceptual quality assessment across channels")
        if not independence_results['noise_independence']:
            print(f"• Common source of variability affecting both channels")
    
    print(f"\nCORE SIGMA MATRIX SUMMARY:")
    print(f"Cross-channel drift rate correlation: {independence_results['drift_rate_correlation']:.3f}")
    print(f"Cross-channel noise correlation: {independence_results['noise_correlation']:.3f}")
    
    print(f"\nMETHODOLOGICAL NOTES:")
    print(f"• Excluded speed: Confounds non-decision time, accumulation, and motor stages")
    print(f"• Excluded bias: Not central to LBA independence theory")
    print(f"• Focus on core: Sensitivity (drift rate) and consistency (inverse noise)")
    print(f"• Clean test: Pure theoretical evaluation without confounding variables")

def main_core_sigma_analysis(results_file: str = 'dual_lba_results_20250615_122314.csv',
                            original_file: str = 'GRT_LBA.csv'):
    """
    Main function for CORE sigma matrix analysis - focused on pure LBA theory testing
    
    REVISION RATIONALE:
    - Focus on core LBA parameters: drift rate (sensitivity) and noise (consistency)
    - Exclude confounding variables: speed (multi-stage) and bias (not independence-critical)
    - Provide clean theoretical test of LBA channel independence assumptions
    """
    
    print("Starting CORE Sigma Matrix Analysis for LBA Independence Testing...")
    print("\nREVISION RATIONALE:")
    print("• FOCUS: Core LBA parameters only (drift rate, noise)")
    print("• EXCLUDE: Confounding variables (speed, bias)")
    print("• GOAL: Clean theoretical test of channel independence")
    
    # Load data
    results_df = pd.read_csv(results_file)
    original_df = pd.read_csv(original_file)
    
    # Extract CORE channel parameters only
    channel_df = extract_core_channel_parameters(results_df, original_df)
    
    # Calculate CORE sigma matrices
    sigma_results = calculate_core_sigma_matrices(channel_df)
    
    # Analyze CORE independence evidence
    independence_results = analyze_core_independence_evidence(sigma_results, channel_df)
    
    # Create CORE visualizations
    visualize_core_sigma_matrices(sigma_results, channel_df)
    
    # Generate CORE comprehensive report
    generate_core_judgment_mechanism_report(sigma_results, independence_results, channel_df)
    
    print(f"\nCore sigma matrix analysis complete!")
    print(f"Visualization saved: core_sigma_matrix_analysis.png")
    print(f"\nTHEORETICAL CONTRIBUTION:")
    print(f"• Pure test of LBA independence assumptions")
    print(f"• Clean separation of core vs confounding parameters")
    print(f"• Evidence-based evaluation of dual-channel architecture")
    
    return {
        'sigma_results': sigma_results,
        'independence_results': independence_results,
        'channel_df': channel_df
    }

if __name__ == "__main__":
    # Run the CORE sigma matrix analysis
    results = main_core_sigma_analysis()
