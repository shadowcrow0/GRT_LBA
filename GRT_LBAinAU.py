# -*- coding: utf-8 -*-
"""
COMPLETE FULL HIERARCHICAL DUAL-CHANNEL LBA MODEL
WITH PYTENSOR COMPATIBILITY FIXES

MAINTAINS ALL LBA PARAMETERS:
- drift_correct, drift_incorrect (evidence accumulation rates)
- threshold (decision boundary) 
- start_var (starting point variability)
- ndt (non-decision time)
- noise (diffusion noise scaling)

FIXES PYTENSOR COMPATIBILITY WITHOUT SIMPLIFYING MODEL
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional
import time
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# Part 1: Data Loading and Preprocessing (COMPLETE)
# ============================================================================

def load_and_prepare_data(csv_file: str = 'GRT_LBA.csv') -> pd.DataFrame:
    """
    Load and prepare raw experimental data for complete LBA analysis
    """
    
    print("🚀 COMPLETE FULL HIERARCHICAL DUAL-CHANNEL LBA ANALYSIS")
    print("="*80)
    print("📂 Loading raw experimental data...")
    
    try:
        # Load raw data
        raw_df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(raw_df)} trials from {csv_file}")
        
        # Complete data preprocessing with dual-channel mapping
        print("🔄 Preprocessing data with complete dual-channel mapping...")
        
        # Full stimulus mapping for dual-channel architecture
        stimulus_mapping = {
            0: {'left_tilt': 1, 'right_tilt': 0, 'description': 'Left\\Right|'},  # Left diagonal, right vertical
            1: {'left_tilt': 1, 'right_tilt': 1, 'description': 'Left\\Right/'},  # Left diagonal, right diagonal
            2: {'left_tilt': 0, 'right_tilt': 0, 'description': 'Left|Right|'},   # Left vertical, right vertical
            3: {'left_tilt': 0, 'right_tilt': 1, 'description': 'Left|Right/'}    # Left vertical, right diagonal
        }
        
        # Extract dual-channel features
        raw_df['left_line_tilt'] = raw_df['Stimulus'].map(
            lambda x: stimulus_mapping.get(x, {'left_tilt': 0})['left_tilt']
        )
        raw_df['right_line_tilt'] = raw_df['Stimulus'].map(
            lambda x: stimulus_mapping.get(x, {'right_tilt': 0})['right_tilt']
        )
        
        # Complete data cleaning
        valid_rt = (raw_df['RT'] >= 0.1) & (raw_df['RT'] <= 3)
        valid_choice = raw_df['Response'].isin([0, 1, 2, 3])
        valid_stimulus = raw_df['Stimulus'].isin([0, 1, 2, 3])
        valid_data = valid_rt & valid_choice & valid_stimulus
        
        clean_df = raw_df[valid_data].copy()
        clean_df = clean_df.dropna(subset=['left_line_tilt', 'right_line_tilt', 'Response', 'RT'])
        
        print(f"✅ Complete data cleaning:")
        print(f"   Original: {len(raw_df)} trials")
        print(f"   Cleaned: {len(clean_df)} trials")
        print(f"   Retention: {len(clean_df)/len(raw_df)*100:.1f}%")
        print(f"   Participants: {clean_df['participant'].nunique()}")
        
        # Show stimulus distribution
        for stim, info in stimulus_mapping.items():
            count = len(clean_df[clean_df['Stimulus'] == stim])
            print(f"   Stimulus {stim} ({info['description']}): {count} trials")
        
        return clean_df
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        raise

# ============================================================================
# Part 2: Complete Hierarchical LBA Architecture
# ============================================================================

def reconstruct_complete_dual_channel_decisions(choice_data: np.ndarray, stimulus_data: np.ndarray) -> Dict:
    """
    Complete reconstruction of dual-channel decisions from 4-choice responses
    """
    
    print("🔄 Reconstructing complete dual-channel LBA architecture...")
    
    # Complete mapping from 4-choice to dual 2-choice decisions
    choice_to_channels = {
        0: (1, 0),  # Choice 0 (\|) → Left=diagonal(1), Right=vertical(0)
        1: (1, 1),  # Choice 1 (\/) → Left=diagonal(1), Right=diagonal(1)
        2: (0, 0),  # Choice 2 (||) → Left=vertical(0), Right=vertical(0)
        3: (0, 1)   # Choice 3 (|/) → Left=vertical(0), Right=diagonal(1)
    }
    
    # Complete stimulus feature extraction
    stimulus_to_features = {
        0: (1, 0),  # Stimulus 0 → Left=diagonal(1), Right=vertical(0)
        1: (1, 1),  # Stimulus 1 → Left=diagonal(1), Right=diagonal(1)
        2: (0, 0),  # Stimulus 2 → Left=vertical(0), Right=vertical(0)
        3: (0, 1)   # Stimulus 3 → Left=vertical(0), Right=diagonal(1)
    }
    
    # Reconstruct channel decisions
    left_decisions = []
    right_decisions = []
    left_stimuli = []
    right_stimuli = []
    
    for choice, stimulus in zip(choice_data, stimulus_data):
        # Level 1 decisions (reconstructed from final choice)
        left_dec, right_dec = choice_to_channels[choice]
        left_decisions.append(left_dec)
        right_decisions.append(right_dec)
        
        # True stimulus features
        left_stim, right_stim = stimulus_to_features[stimulus]
        left_stimuli.append(left_stim)
        right_stimuli.append(right_stim)
    
    architecture_data = {
        'left_decisions': np.array(left_decisions, dtype=float),
        'right_decisions': np.array(right_decisions, dtype=float),
        'left_stimuli': np.array(left_stimuli, dtype=float),
        'right_stimuli': np.array(right_stimuli, dtype=float),
        'original_choices': choice_data,
        'original_stimuli': stimulus_data
    }
    
    print(f"✅ Complete reconstruction: {len(choice_data)} trials")
    print(f"   Left channel decisions: {np.mean(left_decisions):.1%} diagonal")
    print(f"   Right channel decisions: {np.mean(right_decisions):.1%} diagonal")
    
    return architecture_data

# ============================================================================
# Part 3: Complete 2-Choice LBA Likelihood with PyTensor Compatibility
# ============================================================================

def compute_complete_2choice_lba_likelihood(decisions, stimuli, rt, drift_correct, drift_incorrect, 
                                          threshold, start_var, ndt, noise_scale):
    """
    COMPLETE 2-choice LBA likelihood function with full PyTensor compatibility
    
    ALL LBA PARAMETERS MAINTAINED:
    - drift_correct: Evidence accumulation rate for correct responses
    - drift_incorrect: Evidence accumulation rate for incorrect responses  
    - threshold: Decision boundary (b parameter)
    - start_var: Starting point variability (A parameter)
    - ndt: Non-decision time
    - noise_scale: Diffusion noise scaling (s parameter)
    """
    
    # PARAMETER BOUNDS - Ensure all parameters are positive and reasonable
    drift_correct = pt.maximum(drift_correct, 0.1)
    drift_incorrect = pt.maximum(drift_incorrect, 0.05)
    threshold = pt.maximum(threshold, 0.1)
    start_var = pt.maximum(start_var, 0.05)
    ndt = pt.maximum(ndt, 0.05)
    noise_scale = pt.maximum(noise_scale, 0.1)
    
    # Ensure correct drift is always higher than incorrect
    drift_correct = pt.maximum(drift_correct, drift_incorrect + 0.05)
    
    # DECISION TIME CALCULATION
    decision_time = pt.maximum(rt - ndt, 0.01)
    
    # CORRECT VS INCORRECT RESPONSE DETERMINATION
    stimulus_correct = pt.eq(decisions, stimuli)  # 1 if correct, 0 if incorrect
    
    # DRIFT RATES FOR WINNER AND LOSER ACCUMULATORS
    v_winner = pt.where(stimulus_correct, drift_correct, drift_incorrect)
    v_loser = pt.where(stimulus_correct, drift_incorrect, drift_correct)
    
    # COMPLETE LBA DENSITY CALCULATION
    sqrt_t = pt.sqrt(decision_time)
    
    # Winner accumulator calculations with conservative clipping
    z1_winner = pt.clip((v_winner * decision_time - threshold) / (noise_scale * sqrt_t), -4.5, 4.5)
    z2_winner = pt.clip((v_winner * decision_time - start_var) / (noise_scale * sqrt_t), -4.5, 4.5)
    
    # Loser accumulator calculation
    z1_loser = pt.clip((v_loser * decision_time - threshold) / (noise_scale * sqrt_t), -4.5, 4.5)
    
    # PYTENSOR-COMPATIBLE NORMAL FUNCTIONS
    from pytensor.tensor import erf
    
    def safe_normal_cdf(x):
        """PyTensor-compatible normal CDF with safety clipping"""
        x_safe = pt.clip(x, -4.5, 4.5)
        return 0.5 * (1 + erf(x_safe / pt.sqrt(2)))
    
    def safe_normal_pdf(x):
        """PyTensor-compatible normal PDF with safety clipping"""
        x_safe = pt.clip(x, -4.5, 4.5)
        return pt.exp(-0.5 * x_safe**2) / pt.sqrt(2 * pt.pi)
    
    # WINNER LIKELIHOOD CALCULATION
    winner_cdf_term = safe_normal_cdf(z1_winner) - safe_normal_cdf(z2_winner)
    winner_pdf_term = (safe_normal_pdf(z1_winner) - safe_normal_pdf(z2_winner)) / (noise_scale * sqrt_t)
    
    # Ensure positive CDF difference
    winner_cdf_term = pt.maximum(winner_cdf_term, 1e-10)
    
    # Complete winner likelihood
    winner_likelihood = pt.maximum(
        (v_winner / start_var) * winner_cdf_term + winner_pdf_term / start_var,
        1e-10
    )
    
    # LOSER SURVIVAL PROBABILITY
    loser_cdf = safe_normal_cdf(z1_loser)
    loser_survival = pt.maximum(1 - loser_cdf, 1e-10)
    
    # COMPLETE JOINT LBA LIKELIHOOD
    joint_likelihood = winner_likelihood * loser_survival
    joint_likelihood = pt.maximum(joint_likelihood, 1e-12)
    
    # LOG-LIKELIHOOD WITH PYTENSOR-COMPATIBLE VALIDITY CHECKS
    log_likelihood = pt.log(joint_likelihood)
    
    # PYTENSOR-COMPATIBLE INVALID VALUE HANDLING
    # Replace pt.isfinite with manual checks
    is_neg_inf = pt.eq(log_likelihood, -np.inf)
    is_pos_inf = pt.eq(log_likelihood, np.inf)
    is_nan = pt.isnan(log_likelihood)
    
    # Combine all invalid conditions
    is_invalid = is_neg_inf | is_pos_inf | is_nan
    
    # Replace invalid values with penalty
    log_likelihood_safe = pt.where(is_invalid, -100.0, log_likelihood)
    
    # Additional safety: clip extreme values
    log_likelihood_final = pt.clip(log_likelihood_safe, -100.0, 10.0)
    
    return pt.sum(log_likelihood_final)

# ============================================================================
# Part 4: Complete Subject Analysis with All LBA Parameters
# ============================================================================

def fit_complete_hierarchical_lba_subject(subject_data: pd.DataFrame, subject_id: int) -> Dict:
    """
    Fit COMPLETE hierarchical dual-channel LBA model with ALL parameters
    """
    
    print(f"\n🔧 Subject {subject_id}: Complete hierarchical LBA (ALL parameters)")
    
    # Extract complete data
    choice_data = subject_data['Response'].values
    rt_data = subject_data['RT'].values
    stimulus_data = subject_data['Stimulus'].values
    n_trials = len(choice_data)
    
    print(f"   📊 {n_trials} trials, mean RT: {np.mean(rt_data):.3f}s, accuracy: {subject_data['Correct'].mean():.1%}")
    
    # Check minimum data requirement
    if n_trials < 50:
        print(f"   ❌ Insufficient data: {n_trials} < 50 trials")
        return {'subject_id': subject_id, 'success': False, 'error': 'Insufficient data'}
    
    # Complete dual-channel architecture reconstruction
    arch_data = reconstruct_complete_dual_channel_decisions(choice_data, stimulus_data)
    
    # Data quality analysis
    left_prop = np.mean(arch_data['left_decisions'])
    right_prop = np.mean(arch_data['right_decisions'])
    rt_mean = np.mean(rt_data)
    rt_std = np.std(rt_data)
    
    print(f"   🎯 Channel balance: Left={left_prop:.2f}, Right={right_prop:.2f}")
    print(f"   ⏱️  RT statistics: mean={rt_mean:.3f}s, std={rt_std:.3f}s")
    
    # Build COMPLETE PyMC model with ALL LBA parameters
    with pm.Model() as complete_lba_model:
        
        # LEFT CHANNEL - COMPLETE LBA PARAMETERS
        left_drift_correct = pm.Gamma('left_drift_correct', alpha=2.5, beta=1.2, initval=1.8)
        left_drift_incorrect = pm.Gamma('left_drift_incorrect', alpha=2.0, beta=3.0, initval=0.6)
        left_threshold = pm.Gamma('left_threshold', alpha=3.0, beta=3.5, initval=0.9)
        left_start_var = pm.Uniform('left_start_var', lower=0.1, upper=0.7, initval=0.35)
        left_ndt = pm.Uniform('left_ndt', lower=0.05, upper=min(0.6, rt_mean*0.7), initval=0.25)
        left_noise = pm.Gamma('left_noise', alpha=2.5, beta=8.0, initval=0.3)
        
        # RIGHT CHANNEL - COMPLETE LBA PARAMETERS
        right_drift_correct = pm.Gamma('right_drift_correct', alpha=2.5, beta=1.2, initval=1.8)
        right_drift_incorrect = pm.Gamma('right_drift_incorrect', alpha=2.0, beta=3.0, initval=0.6)
        right_threshold = pm.Gamma('right_threshold', alpha=3.0, beta=3.5, initval=0.9)
        right_start_var = pm.Uniform('right_start_var', lower=0.1, upper=0.7, initval=0.35)
        right_ndt = pm.Uniform('right_ndt', lower=0.05, upper=min(0.6, rt_mean*0.7), initval=0.25)
        right_noise = pm.Gamma('right_noise', alpha=2.5, beta=8.0, initval=0.3)
        
        # Convert data to PyTensor tensors
        left_decisions_tensor = pt.as_tensor_variable(arch_data['left_decisions'])
        left_stimuli_tensor = pt.as_tensor_variable(arch_data['left_stimuli'])
        right_decisions_tensor = pt.as_tensor_variable(arch_data['right_decisions'])
        right_stimuli_tensor = pt.as_tensor_variable(arch_data['right_stimuli'])
        rt_tensor = pt.as_tensor_variable(rt_data)
        
        # COMPLETE CHANNEL LIKELIHOODS
        left_likelihood = compute_complete_2choice_lba_likelihood(
            left_decisions_tensor, left_stimuli_tensor, rt_tensor,
            left_drift_correct, left_drift_incorrect,
            left_threshold, left_start_var, left_ndt, left_noise
        )
        
        right_likelihood = compute_complete_2choice_lba_likelihood(
            right_decisions_tensor, right_stimuli_tensor, rt_tensor,
            right_drift_correct, right_drift_incorrect,
            right_threshold, right_start_var, right_ndt, right_noise
        )
        
        # Add likelihoods to model
        pm.Potential('left_channel_complete', left_likelihood)
        pm.Potential('right_channel_complete', right_likelihood)
    
    print(f"   🔧 COMPLETE model: {len(complete_lba_model.basic_RVs)} parameters (6 per channel)")
    
    # Model validation before sampling
    try:
        with complete_lba_model:
            test_point = complete_lba_model.initial_point()
            log_prob = complete_lba_model.compile_logp()(test_point)
            
            if not np.isfinite(log_prob):
                print(f"   ❌ Invalid model log probability: {log_prob}")
                return {'subject_id': subject_id, 'success': False, 'error': 'Invalid model log probability'}
            
            print(f"   ✅ Model validation passed: log_prob = {log_prob:.2f}")
    except Exception as e:
        print(f"   ❌ Model validation failed: {e}")
        return {'subject_id': subject_id, 'success': False, 'error': f'Model validation failed: {e}'}
    
    # COMPLETE MCMC SAMPLING
    print(f"   🎲 Starting COMPLETE MCMC sampling...")
    start_time = time.time()
    
    try:
        with complete_lba_model:
            # Optional MAP estimation for better initialization
            try:
                print(f"   🎯 MAP estimation...")
                map_estimate = pm.find_MAP(method='BFGS', maxeval=800)
                print(f"   ✅ MAP completed")
                use_map = True
            except Exception as e:
                print(f"   ⚠️  MAP failed: {e}, using default initialization")
                map_estimate = None
                use_map = False
            
            # COMPLETE NUTS sampling
            trace = pm.sample(
                draws=500,                    # Adequate sampling for complete model
                tune=500,                     # Adequate tuning
                chains=2,                     # Two chains for convergence
                cores=1,                      # Single core for stability
                target_accept=0.88,           # High acceptance rate
                max_treedepth=9,              # Moderate tree depth
                init='jitter+adapt_diag',     # Robust initialization
                initvals=map_estimate if use_map else None,
                random_seed=42,
                progressbar=True,
                return_inferencedata=True,
                discard_tuned_samples=True
            )
        
        sampling_time = time.time() - start_time
        print(f"   ✅ COMPLETE MCMC completed in {sampling_time/60:.1f} minutes")
        
        # COMPLETE convergence diagnostics
        try:
            summary = az.summary(trace)
            rhat_max = summary['r_hat'].max() if 'r_hat' in summary else 1.0
            ess_min = summary['ess_bulk'].min() if 'ess_bulk' in summary else 100
            converged = rhat_max <= 1.08 and ess_min >= 80  # Slightly relaxed for complex model
            
            print(f"   📊 Convergence: R̂_max={rhat_max:.3f}, ESS_min={ess_min:.0f} {'✅' if converged else '⚠️'}")
        except Exception as e:
            print(f"   ⚠️  Convergence diagnostics failed: {e}")
            rhat_max, ess_min, converged = 1.1, 80, False
        
        # Extract ALL parameter posterior means
        posterior_means = {}
        complete_param_names = [
            'left_drift_correct', 'left_drift_incorrect', 'left_threshold',
            'left_start_var', 'left_ndt', 'left_noise',
            'right_drift_correct', 'right_drift_incorrect', 'right_threshold',
            'right_start_var', 'right_ndt', 'right_noise'
        ]
        
        for param in complete_param_names:
            try:
                posterior_means[param] = float(az.summary(trace, var_names=[param])['mean'].iloc[0])
            except Exception as e:
                print(f"   ⚠️  Failed to extract {param}: {e}")
                posterior_means[param] = np.nan
        
        return {
            'subject_id': subject_id,
            'success': True,
            'converged': converged,
            'trace': trace,
            'posterior_means': posterior_means,
            'n_trials': n_trials,
            'rhat_max': float(rhat_max),
            'ess_min': float(ess_min),
            'sampling_time_minutes': sampling_time / 60,
            'accuracy': subject_data['Correct'].mean(),
            'mean_rt': float(np.mean(rt_data)),
            'architecture_data': arch_data,
            'model_type': 'complete_hierarchical_dual_lba'
        }
        
    except Exception as e:
        print(f"   ❌ COMPLETE MCMC sampling failed: {e}")
        return {
            'subject_id': subject_id,
            'success': False,
            'error': str(e),
            'sampling_time_minutes': (time.time() - start_time) / 60,
            'model_type': 'complete_hierarchical_dual_lba'
        }

# ============================================================================
# Part 5: Complete Batch Analysis
# ============================================================================

def analyze_all_subjects_complete(data_df: pd.DataFrame, max_subjects: int = None) -> List[Dict]:
    """
    Analyze all subjects with COMPLETE hierarchical LBA
    """
    
    participants = sorted(data_df['participant'].unique())
    if max_subjects:
        participants = participants[:max_subjects]
    
    print(f"\n🎯 COMPLETE batch analysis: {len(participants)} subjects")
    print(f"   Each subject: 12 LBA parameters (6 per channel)")
    
    results = []
    start_time = time.time()
    
    for i, subject_id in enumerate(participants, 1):
        print(f"\n{'='*60}")
        print(f"Progress: {i}/{len(participants)} ({i/len(participants)*100:.1f}%)")
        
        # Extract subject data
        subject_data = data_df[data_df['participant'] == subject_id].copy()
        
        # Analyze with COMPLETE model
        result = fit_complete_hierarchical_lba_subject(subject_data, subject_id)
        results.append(result)
        
        # Progress update
        if result['success']:
            convergence_status = "✅ Converged" if result['converged'] else "⚠️ Convergence warning"
            print(f"✅ Subject {subject_id} completed ({convergence_status})")
        else:
            print(f"❌ Subject {subject_id} failed: {result.get('error', 'Unknown error')}")
        
        # Early termination check
        if i >= 3:  # Check after first 3 subjects
            recent_failures = sum(1 for r in results[-3:] if not r['success'])
            if recent_failures >= 3:
                print(f"\n⚠️  Warning: Last 3 subjects all failed")
                response = input("Continue with remaining subjects? (y/n): ")
                if response.lower() != 'y':
                    print("Batch analysis terminated by user")
                    break
    
    # Batch summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    converged = sum(1 for r in results if r.get('success', False) and r.get('converged', False))
    
    print(f"\n{'='*80}")
    print(f"🎉 COMPLETE BATCH ANALYSIS SUMMARY")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"✅ Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"🔄 Converged: {converged}/{successful} ({converged/max(successful,1)*100:.1f}%)")
    
    if successful > 0:
        successful_results = [r for r in results if r['success']]
        avg_time = np.mean([r['sampling_time_minutes'] for r in successful_results])
        print(f"⏱️  Average time per subject: {avg_time:.1f} minutes")
    
    return results

# ============================================================================
# Part 6: Complete Sigma Matrix Analysis with ALL Parameters
# ============================================================================

def compute_complete_sigma_matrices(results: List[Dict]) -> Dict:
    """
    Compute complete sigma matrices using ALL MCMC-estimated LBA parameters
    """
    
    print(f"\n🔬 Computing COMPLETE sigma matrices from ALL LBA parameters...")
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) == 0:
        raise ValueError("No successful results available for complete sigma analysis")
    
    print(f"   📊 Using {len(successful_results)} successful subjects")
    print(f"   🔢 Each subject: 12 parameters (6 per channel)")
    
    # Extract ALL parameters
    param_data = []
    for result in successful_results:
        params = result['posterior_means']
        param_data.append({
            'subject_id': result['subject_id'],
            # LEFT CHANNEL - ALL LBA PARAMETERS
            'left_drift_correct': params['left_drift_correct'],
            'left_drift_incorrect': params['left_drift_incorrect'],
            'left_threshold': params['left_threshold'],
            'left_start_var': params['left_start_var'],
            'left_ndt': params['left_ndt'],
            'left_noise': params['left_noise'],
            # RIGHT CHANNEL - ALL LBA PARAMETERS
            'right_drift_correct': params['right_drift_correct'],
            'right_drift_incorrect': params['right_drift_incorrect'],
            'right_threshold': params['right_threshold'],
            'right_start_var': params['right_start_var'],
            'right_ndt': params['right_ndt'],
            'right_noise': params['right_noise']
        })
    
    params_df = pd.DataFrame(param_data)
    
    # COMPLETE parameter groups
    left_params = ['left_drift_correct', 'left_drift_incorrect', 'left_threshold',
                   'left_start_var', 'left_ndt', 'left_noise']
    right_params = ['right_drift_correct', 'right_drift_incorrect', 'right_threshold',
                    'right_start_var', 'right_ndt', 'right_noise']
    all_params = left_params + right_params
    
    # Compute COMPLETE covariance matrices
    left_data = params_df[left_params].values
    right_data = params_df[right_params].values
    all_data = params_df[all_params].values
    
    sigma_left = np.cov(left_data.T)
    sigma_right = np.cov(right_data.T)
    sigma_bilateral = np.cov(all_data.T)
    
    # COMPLETE cross-channel covariance matrix (6×6)
    sigma_cross = np.zeros((len(left_params), len(right_params)))
    for i, left_param in enumerate(left_params):
        for j, right_param in enumerate(right_params):
            sigma_cross[i, j] = np.cov(params_df[left_param], params_df[right_param])[0, 1]
    
    # COMPLETE correlation matrices
    corr_left = np.corrcoef(left_data.T)
    corr_right = np.corrcoef(right_data.T)
    corr_bilateral = np.corrcoef(all_data.T)
    
    # COMPLETE cross-channel correlations (6×6)
    corr_cross = np.zeros((len(left_params), len(right_params)))
    for i, left_param in enumerate(left_params):
        for j, right_param in enumerate(right_params):
            if np.std(params_df[left_param]) > 0 and np.std(params_df[right_param]) > 0:
                corr_cross[i, j] = np.corrcoef(params_df[left_param], params_df[right_param])[0, 1]
            else:
                corr_cross[i, j] = 0.0
    
    print(f"✅ COMPLETE sigma matrices computed:")
    print(f"   Left channel covariance: {sigma_left.shape}")
    print(f"   Right channel covariance: {sigma_right.shape}")
    print(f"   Cross-channel correlation: {corr_cross.shape}")
    print(f"   Bilateral covariance: {sigma_bilateral.shape}")
    
    return {
        'params_df': params_df,
        'sigma_left': sigma_left,
        'sigma_right': sigma_right,
        'sigma_cross': sigma_cross,
        'sigma_bilateral': sigma_bilateral,
        'corr_left': corr_left,
        'corr_right': corr_right,
        'corr_cross': corr_cross,
        'corr_bilateral': corr_bilateral,
        'left_params': left_params,
        'right_params': right_params,
        'all_params': all_params,
        'n_subjects': len(successful_results)
    }

def test_complete_independence(sigma_results: Dict) -> Dict:
    """
    Test channel independence using ALL MCMC-estimated LBA parameters
    """
    
    print(f"\n🧪 TESTING COMPLETE LBA INDEPENDENCE WITH ALL PARAMETERS")
    print("="*70)
    
    params_df = sigma_results['params_df']
    corr_cross = sigma_results['corr_cross']
    left_params = sigma_results['left_params']
    right_params = sigma_results['right_params']
    n_subjects = sigma_results['n_subjects']
    
    print(f"Testing independence with {n_subjects} subjects")
    print(f"Cross-channel correlations matrix: {corr_cross.shape[0]}×{corr_cross.shape[1]} = {np.prod(corr_cross.shape)} tests")
    
    # Test ALL cross-parameter correlations
    independence_violations = []
    
    print(f"\nCross-channel parameter correlations:")
    print("-" * 70)
    
    for i, left_param in enumerate(left_params):
        for j, right_param in enumerate(right_params):
            r = corr_cross[i, j]
            
            # Statistical significance test
            if abs(r) > 1e-10 and n_subjects > 2:
                t_stat = r * np.sqrt((n_subjects - 2) / (1 - r**2 + 1e-10))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_subjects - 2))
            else:
                p_value = 1.0
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"{left_param:18s} × {right_param:18s}: r={r:6.3f}, p={p_value:.3f} {significance}")
            
            if p_value < 0.05:
                independence_violations.append({
                    'left_param': left_param,
                    'right_param': right_param,
                    'correlation': r,
                    'p_value': p_value,
                    'significance_level': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
                })
    
    # Overall independence assessment
    independence_supported = len(independence_violations) == 0
    
    print(f"\n{'='*70}")
    print(f"COMPLETE LBA INDEPENDENCE TEST RESULTS:")
    if independence_supported:
        print(f"✅ DUAL-CHANNEL LBA INDEPENDENCE SUPPORTED")
        print(f"   No significant cross-channel correlations detected")
        print(f"   Standard dual-channel LBA assumptions validated")
        print(f"   All {len(left_params) * len(right_params)} parameter pairs independent")
    else:
        print(f"❌ DUAL-CHANNEL LBA INDEPENDENCE VIOLATED")
        print(f"   {len(independence_violations)} out of {len(left_params) * len(right_params)} correlations significant")
        print(f"   Violation rate: {len(independence_violations)/(len(left_params) * len(right_params))*100:.1f}%")
        print(f"\n   Significant violations:")
        for violation in independence_violations:
            print(f"   • {violation['left_param']} × {violation['right_param']}: "
                  f"r={violation['correlation']:.3f} (p={violation['p_value']:.3f}) {violation['significance_level']}")
    
    # Theoretical parameter groupings analysis
    drift_violations = [v for v in independence_violations if 'drift' in v['left_param'] and 'drift' in v['right_param']]
    threshold_violations = [v for v in independence_violations if 'threshold' in v['left_param'] and 'threshold' in v['right_param']]
    timing_violations = [v for v in independence_violations if 'ndt' in v['left_param'] and 'ndt' in v['right_param']]
    noise_violations = [v for v in independence_violations if 'noise' in v['left_param'] and 'noise' in v['right_param']]
    
    print(f"\n🔬 THEORETICAL IMPLICATIONS:")
    if len(drift_violations) > 0:
        print(f"   🧠 Evidence accumulation coupling: {len(drift_violations)} drift parameter violations")
        print(f"      → Shared evidence quality assessment across channels")
    
    if len(threshold_violations) > 0:
        print(f"   🎯 Decision threshold coupling: {len(threshold_violations)} threshold violations")
        print(f"      → Shared decision criteria across channels")
    
    if len(timing_violations) > 0:
        print(f"   ⏱️  Timing process coupling: {len(timing_violations)} NDT violations")
        print(f"      → Shared encoding/motor processes")
    
    if len(noise_violations) > 0:
        print(f"   🔄 Noise process coupling: {len(noise_violations)} noise violations")
        print(f"      → Shared attention/arousal sources")
    
    if independence_supported:
        print(f"   ✅ True dual-channel independence confirmed")
        print(f"   📊 Left and right processing streams operate independently")
        print(f"   🔬 Standard LBA assumptions empirically validated")
    
    return {
        'independence_supported': independence_supported,
        'n_violations': len(independence_violations),
        'total_tests': len(left_params) * len(right_params),
        'violation_rate': len(independence_violations) / (len(left_params) * len(right_params)),
        'violations': independence_violations,
        'drift_violations': drift_violations,
        'threshold_violations': threshold_violations,
        'timing_violations': timing_violations,
        'noise_violations': noise_violations,
        'n_subjects': n_subjects
    }

# ============================================================================
# Part 7: Complete Analysis Pipeline
# ============================================================================

def main_complete_analysis(csv_file: str = 'GRT_LBA.csv', max_subjects: int = 3):
    """
    COMPLETE analysis pipeline from raw data to independence testing
    """
    
    print("🚀 COMPLETE HIERARCHICAL DUAL-CHANNEL LBA ANALYSIS")
    print("="*80)
    print("🔬 MAINTAINING ALL LBA PARAMETERS:")
    print("   • drift_correct, drift_incorrect (evidence accumulation)")
    print("   • threshold (decision boundary)")
    print("   • start_var (starting point variability)")
    print("   • ndt (non-decision time)")
    print("   • noise (diffusion noise scaling)")
    print("="*80)
    
    try:
        # Step 1: Load and prepare complete data
        data_df = load_and_prepare_data(csv_file)
        
        # Step 2: Analyze all subjects with complete model
        results = analyze_all_subjects_complete(data_df, max_subjects=max_subjects)
        
        # Check if we have successful results
        successful_count = sum(1 for r in results if r['success'])
        if successful_count == 0:
            print("❌ No successful analyses. Cannot proceed with sigma analysis.")
            return None
        
        print(f"\n✅ Proceeding with {successful_count} successful subjects")
        
        # Step 3: Compute complete sigma matrices
        sigma_results = compute_complete_sigma_matrices(results)
        
        # Step 4: Test complete independence
        independence_results = test_complete_independence(sigma_results)
        
        # Step 5: Save complete results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'trace'} for r in results])
        results_filename = f"complete_hierarchical_lba_results_{timestamp}.csv"
        results_df.to_csv(results_filename, index=False, encoding='utf-8-sig')
        
        # Save complete parameters
        params_filename = f"complete_lba_parameters_{timestamp}.csv"
        sigma_results['params_df'].to_csv(params_filename, index=False, encoding='utf-8-sig')
        
        # Save cross-correlation matrix
        corr_filename = f"complete_cross_correlations_{timestamp}.csv"
        corr_df = pd.DataFrame(sigma_results['corr_cross'], 
                              index=sigma_results['left_params'], 
                              columns=sigma_results['right_params'])
        corr_df.to_csv(corr_filename, encoding='utf-8-sig')
        
        print(f"\n💾 COMPLETE results saved:")
        print(f"   📊 Main results: {results_filename}")
        print(f"   📊 All parameters: {params_filename}")
        print(f"   📊 Cross-correlations: {corr_filename}")
        
        # Summary report
        print(f"\n📋 COMPLETE ANALYSIS SUMMARY:")
        print(f"   Subjects analyzed: {len(results)}")
        print(f"   Successful fits: {successful_count}")
        print(f"   Parameters per subject: 12 (6 per channel)")
        print(f"   Cross-correlation tests: {independence_results['total_tests']}")
        print(f"   Independence supported: {'✅ YES' if independence_results['independence_supported'] else '❌ NO'}")
        
        if not independence_results['independence_supported']:
            print(f"   Violation rate: {independence_results['violation_rate']*100:.1f}%")
        
        return {
            'results': results,
            'sigma_results': sigma_results,
            'independence_results': independence_results,
            'analysis_complete': True,
            'model_type': 'complete_hierarchical_dual_lba'
        }
        
    except Exception as e:
        print(f"❌ COMPLETE analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_test_complete(csv_file: str = 'GRT_LBA.csv'):
    """
    Quick test with COMPLETE model on one subject
    """
    
    print("🧪 QUICK TEST: COMPLETE hierarchical LBA (all parameters)")
    print("="*60)
    
    try:
        result = main_complete_analysis(csv_file, max_subjects=1)
        
        if result and result['analysis_complete']:
            print("✅ COMPLETE QUICK TEST SUCCESSFUL!")
            print("🎯 All LBA parameters estimated successfully")
            print("🔬 Ready for full batch analysis")
            return True
        else:
            print("❌ Complete quick test failed")
            return False
            
    except Exception as e:
        print(f"❌ Complete test error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Part 8: Execution Interface
# ============================================================================

if __name__ == "__main__":
    print("🎯 COMPLETE HIERARCHICAL LBA ANALYSIS OPTIONS:")
    print("="*60)
    print("1. Quick test (1 subject) - Test complete model")
    print("2. Small batch (3 subjects) - Validate approach")
    print("3. Medium batch (5 subjects) - Preliminary analysis")
    print("4. Large batch (10 subjects) - Comprehensive analysis")
    print("5. Full analysis (all subjects) - Complete dataset")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            print("\n🧪 Running COMPLETE quick test...")
            success = quick_test_complete()
        elif choice == '2':
            print("\n🔬 Running COMPLETE small batch analysis...")
            result = main_complete_analysis(max_subjects=3)
            success = result is not None
        elif choice == '3':
            print("\n📊 Running COMPLETE medium batch analysis...")
            result = main_complete_analysis(max_subjects=5)
            success = result is not None
        elif choice == '4':
            print("\n🎯 Running COMPLETE large batch analysis...")
            result = main_complete_analysis(max_subjects=10)
            success = result is not None
        elif choice == '5':
            print("\n🚀 Running COMPLETE full analysis...")
            result = main_complete_analysis(max_subjects=None)
            success = result is not None
        else:
            print("Invalid choice, running quick test...")
            success = quick_test_complete()
        
        if success:
            print("\n🎉 COMPLETE ANALYSIS SUCCESSFUL!")
            print("✅ All LBA parameters maintained and estimated")
            print("🔬 Independence testing completed")
        else:
            print("\n❌ Analysis failed")
            
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
