# -*- coding: utf-8 -*-
"""
PYMC PRIOR ANALYSIS AND SENSITIVITY TESTING
Comprehensive prior predictive checking and sensitivity analysis for LBA models
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# Current Prior Specifications
# ============================================================================

def get_current_prior_specifications() -> Dict:
    """
    Document all current prior specifications used in the LBA models
    """
    
    print("📋 CURRENT PRIOR SPECIFICATIONS IN LBA MODELS")
    print("="*60)
    
    priors = {
        'individual_lba': {
            'drift_correct': {
                'distribution': 'Gamma',
                'parameters': {'alpha': 2.0, 'beta': 1.5},
                'adaptive_init': 'varies by accuracy (1.0-2.2)',
                'rationale': 'Ensures positive drift, higher for correct responses'
            },
            'drift_incorrect': {
                'distribution': 'Gamma', 
                'parameters': {'alpha': 2.0, 'beta': 3.5},
                'adaptive_init': 'varies by accuracy (0.4-0.8)',
                'rationale': 'Lower drift for incorrect responses, constrained positive'
            },
            'threshold': {
                'distribution': 'Gamma',
                'parameters': {'alpha': 2.5, 'beta': 3.0},
                'init': 0.8,
                'rationale': 'Conservative decision boundaries, positive constraint'
            },
            'start_var': {
                'distribution': 'Uniform',
                'parameters': {'lower': 0.1, 'upper': 0.6},
                'init': 0.3,
                'rationale': 'Starting point variability, bounded for identifiability'
            },
            'ndt': {
                'distribution': 'Uniform',
                'parameters': {'lower': 0.05, 'upper': 'min(0.5, rt_mean*0.6)'},
                'init': 0.2,
                'rationale': 'Non-decision time, adaptive upper bound based on RT'
            },
            'noise': {
                'distribution': 'Gamma',
                'parameters': {'alpha': 2.0, 'beta': 6.0},
                'init': 0.35,
                'rationale': 'Diffusion noise scaling, typically around 0.3-0.5'
            }
        },
        'hierarchical_integrated': {
            'integration_weight': {
                'distribution': 'Beta',
                'parameters': {'alpha': 2, 'beta': 2},
                'init': 0.7,
                'rationale': 'Weakly informative, slight bias toward channel evidence'
            },
            'threshold_mean': {
                'distribution': 'Gamma',
                'parameters': {'alpha': 2.5, 'beta': 3.0},
                'init': 0.8,
                'rationale': 'Hierarchical threshold center'
            },
            'threshold_std': {
                'distribution': 'HalfNormal',
                'parameters': {'sigma': 0.2},
                'rationale': 'Individual threshold variation around mean'
            },
            'thresholds': {
                'distribution': 'Normal',
                'parameters': {'mu': 'threshold_mean', 'sigma': 'threshold_std'},
                'shape': 4,
                'rationale': 'Individual accumulator thresholds'
            }
        }
    }
    
    # Print detailed specifications
    for model_type, model_priors in priors.items():
        print(f"\n🔧 {model_type.upper()} MODEL PRIORS:")
        for param, spec in model_priors.items():
            print(f"   {param}:")
            print(f"     Distribution: {spec['distribution']}({spec['parameters']})")
            if 'init' in spec:
                print(f"     Initial value: {spec['init']}")
            if 'adaptive_init' in spec:
                print(f"     Adaptive init: {spec['adaptive_init']}")
            print(f"     Rationale: {spec['rationale']}")
            print()
    
    return priors

# ============================================================================
# Prior Predictive Sampling
# ============================================================================

def sample_individual_lba_priors(n_samples: int = 1000, accuracy_level: float = 0.6) -> Dict:
    """
    Sample from individual LBA model priors to understand prior implications
    """
    
    print(f"🎲 SAMPLING INDIVIDUAL LBA PRIORS")
    print(f"   Samples: {n_samples}")
    print(f"   Assumed accuracy level: {accuracy_level:.1%}")
    print("-" * 40)
    
    # Adaptive initialization based on accuracy
    if accuracy_level < 0.40:
        drift_correct_init = 1.0
        drift_incorrect_init = 0.8
    elif accuracy_level > 0.75:
        drift_correct_init = 2.2
        drift_incorrect_init = 0.4
    else:
        drift_correct_init = 1.6
        drift_incorrect_init = 0.6
    
    # Build prior-only model
    with pm.Model() as prior_model:
        # Sample from priors
        drift_correct = pm.Gamma('drift_correct', alpha=2.0, beta=1.5, initval=drift_correct_init)
        drift_incorrect = pm.Gamma('drift_incorrect', alpha=2.0, beta=3.5, initval=drift_incorrect_init)
        threshold = pm.Gamma('threshold', alpha=2.5, beta=3.0, initval=0.8)
        start_var = pm.Uniform('start_var', lower=0.1, upper=0.6, initval=0.3)
        ndt = pm.Uniform('ndt', lower=0.05, upper=0.5, initval=0.2)  # Fixed upper bound for prior sampling
        noise = pm.Gamma('noise', alpha=2.0, beta=6.0, initval=0.35)
        
        # Sample from prior
        prior_samples = pm.sample_prior_predictive(samples=n_samples, return_inferencedata=True)
    
    # Extract samples
    samples = {}
    param_names = ['drift_correct', 'drift_incorrect', 'threshold', 'start_var', 'ndt', 'noise']
    
    for param in param_names:
        samples[param] = prior_samples.prior[param].values.flatten()
    
    # Compute derived quantities
    samples['drift_difference'] = samples['drift_correct'] - samples['drift_incorrect']
    samples['drift_ratio'] = samples['drift_correct'] / (samples['drift_incorrect'] + 1e-6)
    
    # Summary statistics
    print("📊 PRIOR SAMPLE STATISTICS:")
    for param in param_names + ['drift_difference', 'drift_ratio']:
        values = samples[param]
        print(f"   {param}:")
        print(f"     Mean: {np.mean(values):.3f} ± {np.std(values):.3f}")
        print(f"     Range: [{np.min(values):.3f}, {np.max(values):.3f}]")
        print(f"     Quantiles: [{np.quantile(values, 0.05):.3f}, {np.quantile(values, 0.95):.3f}]")
    
    # Check for problematic prior combinations
    print("\n⚠️  PRIOR CONSTRAINT VIOLATIONS:")
    
    # Check drift ordering constraint
    drift_violations = np.sum(samples['drift_correct'] <= samples['drift_incorrect'])
    print(f"   Drift ordering violations: {drift_violations}/{n_samples} ({drift_violations/n_samples:.1%})")
    
    # Check extreme parameter values
    extreme_threshold = np.sum(samples['threshold'] > 3.0)
    print(f"   Extreme thresholds (>3.0): {extreme_threshold}/{n_samples} ({extreme_threshold/n_samples:.1%})")
    
    extreme_ndt = np.sum(samples['ndt'] > 0.4)
    print(f"   Extreme NDT (>0.4s): {extreme_ndt}/{n_samples} ({extreme_ndt/n_samples:.1%})")
    
    return {
        'samples': samples,
        'param_names': param_names,
        'n_samples': n_samples,
        'accuracy_level': accuracy_level,
        'violations': {
            'drift_ordering': drift_violations,
            'extreme_threshold': extreme_threshold,
            'extreme_ndt': extreme_ndt
        }
    }

def sample_hierarchical_integrated_priors(n_samples: int = 1000) -> Dict:
    """
    Sample from hierarchical integrated model priors
    """
    
    print(f"🎲 SAMPLING HIERARCHICAL INTEGRATED PRIORS")
    print(f"   Samples: {n_samples}")
    print("-" * 40)
    
    with pm.Model() as hierarchical_prior_model:
        # Integration weight
        integration_weight = pm.Beta('integration_weight', alpha=2, beta=2, initval=0.7)
        
        # Hierarchical thresholds
        threshold_mean = pm.Gamma('threshold_mean', alpha=2.5, beta=3.0, initval=0.8)
        threshold_std = pm.HalfNormal('threshold_std', sigma=0.2)
        thresholds = pm.Normal('thresholds', mu=threshold_mean, sigma=threshold_std, shape=4, initval=0.8)
        
        # Other parameters
        start_var = pm.Uniform('start_var', lower=0.1, upper=0.6, initval=0.3)
        ndt = pm.Uniform('ndt', lower=0.05, upper=0.5, initval=0.2)
        noise = pm.Gamma('noise', alpha=2.0, beta=6.0, initval=0.35)
        
        # Sample from prior
        prior_samples = pm.sample_prior_predictive(samples=n_samples, return_inferencedata=True)
    
    # Extract samples
    samples = {}
    param_names = ['integration_weight', 'threshold_mean', 'threshold_std', 'start_var', 'ndt', 'noise']
    
    for param in param_names:
        samples[param] = prior_samples.prior[param].values.flatten()
    
    # Extract individual thresholds
    thresholds_samples = prior_samples.prior['thresholds'].values  # Shape: (1, n_samples, 4)
    for i in range(4):
        samples[f'threshold_{i}'] = thresholds_samples[0, :, i]
    
    # Derived quantities
    samples['threshold_range'] = np.max(thresholds_samples[0], axis=1) - np.min(thresholds_samples[0], axis=1)
    samples['channel_evidence_weight'] = samples['integration_weight']
    samples['additional_source_weight'] = 1 - samples['integration_weight']
    
    # Summary statistics
    print("📊 HIERARCHICAL PRIOR STATISTICS:")
    all_params = param_names + [f'threshold_{i}' for i in range(4)] + ['threshold_range', 'channel_evidence_weight', 'additional_source_weight']
    
    for param in all_params:
        values = samples[param]
        print(f"   {param}:")
        print(f"     Mean: {np.mean(values):.3f} ± {np.std(values):.3f}")
        print(f"     Range: [{np.min(values):.3f}, {np.max(values):.3f}]")
    
    # Integration weight analysis
    print(f"\n🔗 INTEGRATION WEIGHT IMPLICATIONS:")
    high_channel = np.sum(samples['integration_weight'] > 0.7)
    balanced = np.sum((samples['integration_weight'] >= 0.3) & (samples['integration_weight'] <= 0.7))
    high_additional = np.sum(samples['integration_weight'] < 0.3)
    
    print(f"   High channel weight (>70%): {high_channel}/{n_samples} ({high_channel/n_samples:.1%})")
    print(f"   Balanced integration (30-70%): {balanced}/{n_samples} ({balanced/n_samples:.1%})")
    print(f"   High additional weight (<30%): {high_additional}/{n_samples} ({high_additional/n_samples:.1%})")
    
    return {
        'samples': samples,
        'param_names': param_names,
        'n_samples': n_samples,
        'integration_patterns': {
            'high_channel': high_channel,
            'balanced': balanced, 
            'high_additional': high_additional
        }
    }

# ============================================================================
# Prior Sensitivity Analysis
# ============================================================================

def prior_sensitivity_analysis(param_name: str, prior_variations: List[Dict], 
                               n_samples: int = 500) -> Dict:
    """
    Test sensitivity to different prior specifications for a given parameter
    """
    
    print(f"🔬 PRIOR SENSITIVITY ANALYSIS: {param_name}")
    print(f"   Testing {len(prior_variations)} prior specifications")
    print("-" * 50)
    
    sensitivity_results = {}
    
    for i, prior_spec in enumerate(prior_variations):
        print(f"   Prior {i+1}: {prior_spec}")
        
        try:
            with pm.Model() as sensitivity_model:
                # Create parameter with specified prior
                if prior_spec['distribution'] == 'Gamma':
                    param = pm.Gamma(param_name, 
                                   alpha=prior_spec['alpha'], 
                                   beta=prior_spec['beta'])
                elif prior_spec['distribution'] == 'Normal':
                    param = pm.Normal(param_name,
                                    mu=prior_spec['mu'],
                                    sigma=prior_spec['sigma'])
                elif prior_spec['distribution'] == 'Uniform':
                    param = pm.Uniform(param_name,
                                     lower=prior_spec['lower'],
                                     upper=prior_spec['upper'])
                elif prior_spec['distribution'] == 'Beta':
                    param = pm.Beta(param_name,
                                  alpha=prior_spec['alpha'],
                                  beta=prior_spec['beta'])
                else:
                    print(f"     ❌ Unsupported distribution: {prior_spec['distribution']}")
                    continue
                
                # Sample from this prior
                samples = pm.sample_prior_predictive(samples=n_samples, return_inferencedata=True)
                
                # Extract samples
                param_samples = samples.prior[param_name].values.flatten()
                
                # Compute statistics
                sensitivity_results[f"prior_{i+1}"] = {
                    'specification': prior_spec,
                    'samples': param_samples,
                    'mean': np.mean(param_samples),
                    'std': np.std(param_samples),
                    'median': np.median(param_samples),
                    'q05': np.quantile(param_samples, 0.05),
                    'q95': np.quantile(param_samples, 0.95),
                    'min': np.min(param_samples),
                    'max': np.max(param_samples)
                }
                
                print(f"     Mean: {np.mean(param_samples):.3f} ± {np.std(param_samples):.3f}")
                print(f"     95% Range: [{np.quantile(param_samples, 0.05):.3f}, {np.quantile(param_samples, 0.95):.3f}]")
                
        except Exception as e:
            print(f"     ❌ Error: {e}")
            sensitivity_results[f"prior_{i+1}"] = {'error': str(e)}
    
    # Compare priors
    print(f"\n📊 PRIOR COMPARISON SUMMARY:")
    valid_results = {k: v for k, v in sensitivity_results.items() if 'error' not in v}
    
    if len(valid_results) > 1:
        print(f"   Range of means: [{min(r['mean'] for r in valid_results.values()):.3f}, {max(r['mean'] for r in valid_results.values()):.3f}]")
        print(f"   Range of std: [{min(r['std'] for r in valid_results.values()):.3f}, {max(r['std'] for r in valid_results.values()):.3f}]")
        
        # Compute pairwise overlap
        overlaps = []
        prior_keys = list(valid_results.keys())
        
        for i in range(len(prior_keys)):
            for j in range(i+1, len(prior_keys)):
                prior1 = valid_results[prior_keys[i]]
                prior2 = valid_results[prior_keys[j]]
                
                # Approximate overlap using 95% ranges
                overlap_lower = max(prior1['q05'], prior2['q05'])
                overlap_upper = min(prior1['q95'], prior2['q95'])
                
                if overlap_upper > overlap_lower:
                    range1 = prior1['q95'] - prior1['q05']
                    range2 = prior2['q95'] - prior2['q05']
                    overlap_length = overlap_upper - overlap_lower
                    
                    overlap_pct = overlap_length / min(range1, range2)
                    overlaps.append(overlap_pct)
                    
                    print(f"   {prior_keys[i]} vs {prior_keys[j]}: {overlap_pct:.1%} overlap")
                else:
                    overlaps.append(0.0)
                    print(f"   {prior_keys[i]} vs {prior_keys[j]}: No overlap")
        
        mean_overlap = np.mean(overlaps) if overlaps else 0
        print(f"   Average overlap: {mean_overlap:.1%}")
        
        if mean_overlap > 0.8:
            print(f"   ✅ Low sensitivity - priors are similar")
        elif mean_overlap > 0.5:
            print(f"   ⚖️  Moderate sensitivity")
        else:
            print(f"   ⚠️  High sensitivity - prior choice matters!")
    
    return sensitivity_results

# ============================================================================
# Prior Robustness Testing
# ============================================================================

def test_prior_robustness_individual_lba() -> Dict:
    """
    Test robustness of individual LBA model to prior specifications
    """
    
    print(f"🛡️  TESTING INDIVIDUAL LBA PRIOR ROBUSTNESS")
    print("="*60)
    
    robustness_results = {}
    
    # Test drift_correct prior sensitivity
    print(f"\n1. DRIFT_CORRECT SENSITIVITY:")
    drift_correct_priors = [
        {'distribution': 'Gamma', 'alpha': 1.5, 'beta': 1.0},  # More dispersed
        {'distribution': 'Gamma', 'alpha': 2.0, 'beta': 1.5},  # Current
        {'distribution': 'Gamma', 'alpha': 3.0, 'beta': 2.0},  # More concentrated
        {'distribution': 'Gamma', 'alpha': 2.0, 'beta': 1.0},  # Higher mean
    ]
    
    robustness_results['drift_correct'] = prior_sensitivity_analysis(
        'drift_correct', drift_correct_priors)
    
    # Test integration_weight prior sensitivity (for hierarchical model)
    print(f"\n2. INTEGRATION_WEIGHT SENSITIVITY:")
    integration_priors = [
        {'distribution': 'Beta', 'alpha': 1, 'beta': 1},     # Uniform
        {'distribution': 'Beta', 'alpha': 2, 'beta': 2},     # Current (weakly informative)
        {'distribution': 'Beta', 'alpha': 3, 'beta': 2},     # Biased toward channel
        {'distribution': 'Beta', 'alpha': 2, 'beta': 3},     # Biased toward additional
        {'distribution': 'Beta', 'alpha': 5, 'beta': 5},     # Concentrated at 0.5
    ]
    
    robustness_results['integration_weight'] = prior_sensitivity_analysis(
        'integration_weight', integration_priors)
    
    # Test threshold prior sensitivity
    print(f"\n3. THRESHOLD SENSITIVITY:")
    threshold_priors = [
        {'distribution': 'Gamma', 'alpha': 2.0, 'beta': 2.0},  # Lower thresholds
        {'distribution': 'Gamma', 'alpha': 2.5, 'beta': 3.0},  # Current
        {'distribution': 'Gamma', 'alpha': 3.0, 'beta': 4.0},  # Higher thresholds
        {'distribution': 'Uniform', 'lower': 0.2, 'upper': 2.0},  # Non-informative
    ]
    
    robustness_results['threshold'] = prior_sensitivity_analysis(
        'threshold', threshold_priors)
    
    return robustness_results

# ============================================================================
# Prior Visualization
# ============================================================================

def plot_prior_distributions(prior_samples: Dict, model_type: str = 'individual', 
                            save_plots: bool = True) -> None:
    """
    Visualize prior distributions and their implications
    """
    
    print(f"📊 PLOTTING {model_type.upper()} PRIOR DISTRIBUTIONS")
    print("-" * 50)
    
    samples = prior_samples['samples']
    param_names = prior_samples['param_names']
    
    # Set up plotting
    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    # Plot each parameter
    for i, param in enumerate(param_names):
        ax = axes[i]
        values = samples[param]
        
        # Histogram
        ax.hist(values, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        
        # Statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        q05, q95 = np.quantile(values, [0.05, 0.95])
        
        ax.axvline(mean_val, color='red', linestyle='-', alpha=0.8, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
        ax.axvline(q05, color='gray', linestyle=':', alpha=0.6, label=f'5%: {q05:.2f}')
        ax.axvline(q95, color='gray', linestyle=':', alpha=0.6, label=f'95%: {q95:.2f}')
        
        ax.set_title(f'{param} Prior Distribution', fontsize=12)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_prior_distributions_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Prior distributions saved: {filename}")
    
    plt.show()
    
    # Plot parameter relationships for key pairs
    if model_type == 'individual' and 'drift_correct' in samples and 'drift_incorrect' in samples:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.scatter(samples['drift_incorrect'], samples['drift_correct'], 
                  alpha=0.6, s=20, color='blue')
        
        # Add diagonal line (drift_correct = drift_incorrect)
        max_val = max(np.max(samples['drift_correct']), np.max(samples['drift_incorrect']))
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, 
                label='drift_correct = drift_incorrect')
        
        ax.set_xlabel('Drift Incorrect')
        ax.set_ylabel('Drift Correct')
        ax.set_title('Prior Relationship: Drift Rates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Count violations
        violations = np.sum(samples['drift_correct'] <= samples['drift_incorrect'])
        total = len(samples['drift_correct'])
        ax.text(0.05, 0.95, f'Violations: {violations}/{total} ({violations/total:.1%})', 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        if save_plots:
            filename = f"{model_type}_drift_relationship_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Drift relationship saved: {filename}")
        
        plt.show()

# ============================================================================
# Main Prior Analysis Function
# ============================================================================

def comprehensive_prior_analysis(n_samples: int = 1000, save_outputs: bool = True) -> Dict:
    """
    Comprehensive prior analysis for all LBA models
    """
    
    print("🔬 COMPREHENSIVE PRIOR ANALYSIS")
    print("="*80)
    
    # Document current priors
    prior_specs = get_current_prior_specifications()
    
    # Sample from individual LBA priors
    print("\n" + "="*60)
    individual_prior_samples = sample_individual_lba_priors(n_samples)
    
    # Sample from hierarchical integrated priors  
    print("\n" + "="*60)
    hierarchical_prior_samples = sample_hierarchical_integrated_priors(n_samples)
    
    # Prior robustness testing
    print("\n" + "="*60)
    robustness_results = test_prior_robustness_individual_lba()
    
    # Generate visualizations
    if save_outputs:
        plot_prior_distributions(individual_prior_samples, 'individual', save_outputs)
        plot_prior_distributions(hierarchical_prior_samples, 'hierarchical', save_outputs)
    
    # Save comprehensive results
    if save_outputs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save prior samples to CSV
        individual_df = pd.DataFrame(individual_prior_samples['samples'])
        individual_filename = f"individual_lba_prior_samples_{timestamp}.csv"
        individual_df.to_csv(individual_filename, index=False)
        
        hierarchical_df = pd.DataFrame(hierarchical_prior_samples['samples'])
        hierarchical_filename = f"hierarchical_lba_prior_samples_{timestamp}.csv"
        hierarchical_df.to_csv(hierarchical_filename, index=False)
        
        print(f"\n💾 Prior samples saved:")
        print(f"   Individual: {individual_filename}")
        print(f"   Hierarchical: {hierarchical_filename}")
    
    print(f"\n🎉 COMPREHENSIVE PRIOR ANALYSIS COMPLETE")
    
    return {
        'prior_specifications': prior_specs,
        'individual_samples': individual_prior_samples,
        'hierarchical_samples': hierarchical_prior_samples,
        'robustness_results': robustness_results,
        'analysis_complete': True
    }

if __name__ == "__main__":
    # Run comprehensive prior analysis
    prior_analysis_results = comprehensive_prior_analysis(n_samples=1000, save_outputs=True)
